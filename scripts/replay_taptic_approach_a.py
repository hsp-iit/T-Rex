"""CPU inference-only native replay with provisional uncalibrated gate thresholds.

All native reads produce feature/gate diagnostics. Tiny model inference runs
only inside a configurable small number of scheduled slow-cache intervals.
No imitation errors or hardware-frequency/latency claims are produced.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_vla import Qwen3VLVLAModel, split_slow_fast_embeds
from qwen_vla.taptic_dataset import TRexTapticDataset
from taptic.online_contact import ContactNormalization, NativeSample, OnlineContactExtractor
from taptic.inference import ApproachAInference, SlowAnchor
from taptic.split_gate import GateConfig
from taptic.timing import build_quality
from scripts.analyze_taptic_risk import causal_features


def tiny_model(processor):
    torch.manual_seed(321)
    cfg=AutoConfig.from_pretrained(processor,local_files_only=True)
    t,v=cfg.text_config,cfg.vision_config
    t.hidden_size,t.intermediate_size,t.num_hidden_layers=64,128,2
    t.num_attention_heads,t.num_key_value_heads,t.head_dim=2,1,32
    t.rope_scaling={'rope_type':'default','mrope_section':[6,5,5],'mrope_interleaved':True}
    v.hidden_size,v.intermediate_size,v.depth,v.num_heads=32,64,1,2
    v.out_hidden_size,v.deepstack_visual_indexes=64,[0]
    cfg._attn_implementation='eager'
    base=Qwen3VLForConditionalGeneration(cfg).float().eval()
    model=Qwen3VLVLAModel(t,action_dim=21,action_chunk=16,tacf6_dim=360,
                        use_robot_state=True,tactile_intermediate_size=64).float().eval()
    model.initialize_vla_weights()
    model.visual=base.model.visual
    object.__setattr__(model,'_rope_index_fn',base.model.get_rope_index)
    torch.nn.init.normal_(model.final_layer.mlp.fc2.weight,std=.04)
    torch.nn.init.normal_(model.final_layer_tactile.mlp.fc2.weight,std=.08)
    return model


def context(model,batch):
    emb=model.prepare_inputs_embeds(batch['input_ids'],batch['pixel_values'].float(),batch['image_grid_thw'])
    slow,fast=split_slow_fast_embeds(emb,batch['input_ids'],model.image_token_id,int(batch['image_grid_thw'][0].prod()//4))
    pos,_=model.get_rope_index(batch['input_ids'],batch['image_grid_thw'],batch['attention_mask'])
    state=model.state_embedder(batch['state_raw'].float()).unsqueeze(1)
    return dict(inputs_embeds=slow,position_ids=pos[...,:slow.shape[1]],fast_embeds=fast,state_embeds=state,
                noise=torch.randn(1,16,21),attention_mask=batch['attention_mask'])


def episode_samples(source,ep):
    sm,av=source.tactile(ep);frames=source.clocks(ep)
    q=build_quality(sm,av,True)
    keep=(q['read_end_ns']>=frames[0])&(q['read_end_ns']<=frames[-1])
    for i in np.flatnonzero(np.diff(frames)>150_000_000):
        keep &= ~((q['read_end_ns']>frames[i])&(q['read_end_ns']<frames[i+1]))
    idx=np.flatnonzero(keep)
    previous=-2
    samples=[]
    for i in idx:
        row=sm.iloc[i]
        samples.append(NativeSample(int(ep),int(row.read_end_ns),int(av[i]),np.asarray(row.force).reshape(5,3),
            sequence=int(row.sequence),reset_sequence=int(row.reset_sequence),read_errors=int(row.read_errors),
            dropped=int(row.dropped),boundary=bool(q['break_before'][i] or i!=previous+1),
            taxels=np.asarray(row['values']).reshape(5,120,3)))
        previous=i
    return samples


def run(root,processor,out,episodes=(0,48),slow_ticks=2,gate_config=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=SimpleNamespace(lerobot_root=root,taptic_stats='docs/taptic_train_stats.json',action_dim=21,action_chunk=16,
        use_tactile_vec=1,use_tactile_deform=0,use_tactile_code=0,use_tactile_vqvae=0,taptic_tactile_input='taxels',
        use_robot_state=1,use_flare=0,image_size=(56,56),tactile_delay_offsets=[0],cascaded_split_step=6)
    ds=TRexTapticDataset(cfg,AutoProcessor.from_pretrained(processor,local_files_only=True),SimpleNamespace(print=print))
    source=ds.source
    norm=ContactNormalization.from_analysis('docs/taptic_risk_analysis/train_normalization.json',source.episodes,source.identity)
    gate=gate_config or GateConfig(mode='combined',thresholds=(.2,.5),rate_windows=(100,),hysteresis=.03,min_dwell_ns=20_000_000)
    model=tiny_model(processor)
    parameters={k:v.detach().clone() for k,v in model.state_dict().items()}
    rows,actions,action_times,builds,files=[],[],[],[],{}
    contracts=[];offline_errors=[]
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        for ep in episodes:
            meta=source.episodes.loc[ep]
            if bool(meta.tactile_interpolated):
                raise ValueError('native high-frequency episodes required')
            for suffix in ('right','frames'):
                p=source.root/f'tactile_raw/episode_{ep:06d}_{suffix}.parquet'
                files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            samples=episode_samples(source,ep)
            if not all(s.read_ns==s.available_ns for s in samples):
                raise ValueError('this offline-reference replay requires native read-time availability')
            offline=causal_features(np.asarray([s.read_ns for s in samples],np.int64),
                np.stack([s.force for s in samples]),np.asarray([s.boundary for s in samples]))
            reference_index={s.read_ns:i for i,s in enumerate(samples)}
            ex=OnlineContactExtractor(norm)
            pipeline=ApproachAInference(model,ex,gate,ds.action_stats)
            pipeline.start_episode(int(ep));ex.ingest(samples)
            fr=pd.read_parquet(source.root/f'tactile_raw/episode_{ep:06d}_frames.parquet').sort_values('frame_index')
            observed=fr.observation_ns.to_numpy(np.int64)
            # Conservative recorder availability proxy; not camera exposure time.
            available=np.maximum(observed,fr.add_frame_ns.to_numpy(np.int64))
            candidates=[(i,t) for i,(e,t) in enumerate(ds._rows) if e==ep and t+1<len(available)
                        and available[t+1]>available[t]]
            if not candidates:raise ValueError('episode has no eligible training-scope slow anchors')
            first=min(range(len(candidates)),key=lambda j:abs(candidates[j][1]-71))
            selected=candidates[first:first+slow_ticks]
            scheduled={int(available[t]):(i,t) for i,t in selected}
            timeline=sorted(set(int(s.available_ns) for s in samples)|set(scheduled))
            for tick in timeline:
                if tick in scheduled:
                    index,frame=scheduled[tick]
                    started=time.perf_counter()
                    item=ds[index];batch=ds.collate_fn([item])
                    ctx=context(model,batch)
                    context_ms=(time.perf_counter()-started)*1000
                    state=source.frames(ep).iloc[frame]['observation.state']
                    anchor=SlowAnchor(int(ep),tick,int(observed[frame]),tick,tick,
                        min(tick+100_000_000,int(available[frame+1])),tuple(state[34:37]),tuple(state[37:40]))
                    build=pipeline.slow_tick(anchor,**ctx)
                    saved_cache={s.tau:s.cached_kv for s in pipeline.ladder.snapshots}
                    build.update(episode=int(ep),frame=int(frame),context_ms=context_ms)
                    builds.append(build)
                    contracts.append(asdict(anchor))
                result=pipeline.fast_tick(int(ep),tick)
                c,g=result['contact'],result['gate']
                if tick in reference_index:
                    ri=reference_index[tick]
                    for w in (50,100,200):
                        assert c['quality']['feature_valid']['R'+str(w)]==bool(offline['windows'][w]['valid'][ri])
                        error=float(np.max(np.abs(np.asarray(c['positive_variation_rate'][str(w)])-
                                                   offline['windows'][w]['positive_variation_rate'][ri])))
                        offline_errors.append(error)
                        assert error<1e-9
                row=dict(episode=int(ep),task=meta.source_task,object=meta['object'],tick_ns=tick,
                    slow_anchor_ns=result['slow_anchor_ns'],cache_id=result['cache_id'],status=result['status'],
                    age_ms=c['quality']['age_ns']/1e6 if c['quality']['age_ns'] is not None else None,
                    fresh=c['quality']['fresh'],segment=c['quality']['segment'],
                    **c['features'],raw_score=g['raw_score'],score=g['score'],selected_split=g['selected_split'],
                    fallback=g['fallback'],transition=g['transition'],gate_reason=g['reason'],
                    action_steps=result['action_steps'],tactile_steps=result['tactile_steps'],
                    slow_action_euler_updates=result['slow_action_euler_updates'],
                    feature_gate_ms=result['feature_gate_ms'],tactile_ms=result['tactile_ms'],
                    **{'valid_'+k:v for k,v in c['quality']['feature_valid'].items()},
                    selector_state=json.dumps(g['state'],sort_keys=True))
                rows.append(row)
                if result['actions'] is not None:
                    for s in pipeline.ladder.snapshots:
                        for old,new in zip(saved_cache[s.tau].layers,s.cached_kv.layers):
                            assert torch.equal(old.keys,new.keys) and torch.equal(old.values,new.values)
                    actions.append(result['actions']);action_times.append((ep,tick))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in parameters.items())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in files.items())
    table=pd.DataFrame(rows);table.to_csv(out/'ticks.csv',index=False)
    np.savez_compressed(out/'actions.npz',actions=np.stack(actions) if actions else np.empty((0,1,16,21)),
                        episode_tick=np.asarray(action_times,np.int64).reshape(-1,2))
    (out/'gate_config.json').write_text(json.dumps(asdict(gate),indent=2))
    (out/'contact_normalization.json').write_text(json.dumps(asdict(norm),indent=2))
    (out/'slow_anchors.json').write_text(json.dumps(contracts,indent=2))
    summary=dict(provisional_uncalibrated=True,random_model=True,action_error_supported=False,
        context_availability='max(observation_ns, recorder add_frame_ns), proxy only',
        ready_time='logical replay ready=S; CPU computation latency not added to recorded time',
        gate_config=asdict(gate),train_normalization_episodes=list(norm.train_episodes),source_identity=source.identity,
        episodes=[int(e) for e in episodes],ticks=len(rows),predicted_chunks=len(actions),
        builds=builds,slow_cache_builds=len(builds),weights_unchanged=True,source_files_unchanged=True,caches_preserved=True,
        source_sha256=files,transitions=int(table.transition.sum()),fallbacks=int(table.fallback.sum()),
        offline_max_abs_rate_error=max(offline_errors,default=0.),offline_compared_window_cutoffs=len(offline_errors),
        selection_counts={str(k):int(v) for k,v in table.selected_split.value_counts().items()},
        status_counts={str(k):int(v) for k,v in table.status.value_counts().items()},
        actions_shape=list(np.stack(actions).shape) if actions else [],
        feature_ranges={k:[float(table[k].min()),float(table[k].max())] for k in ('M','R50','R100','R200')})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plot(table,out,gate)
    print(json.dumps({k:summary[k] for k in ('episodes','ticks','predicted_chunks','slow_cache_builds','transitions','fallbacks','selection_counts','actions_shape')},indent=2))
    return summary


def plot(table,out,gate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for ep,g in table.groupby('episode'):
        t=(g.tick_ns-g.tick_ns.iloc[0])/1e9
        broken=(g.segment.diff().fillna(0)!=0)|(g.tick_ns.diff().fillna(0)>50_000_000)
        fig,axes=plt.subplots(4,1,sharex=True,figsize=(13,10))
        for feature in ('M','R50','R100','R200'):axes[0].plot(t,g[feature].where(~broken),label=feature)
        axes[0].legend();axes[0].set_ylabel('Normalized descriptors')
        axes[1].plot(t,g.score.where(~broken),label='EMA scalar score')
        for threshold in gate.thresholds:axes[1].axhline(threshold,ls='--',color='red',alpha=.5)
        axes[1].legend();axes[1].set_ylabel('Provisional score')
        axes[2].step(t,g.selected_split.where(~broken),where='post');axes[2].set_yticks([.2,.4,.6]);axes[2].set_ylabel('Selected tau')
        for number,(_,active) in enumerate(g[g.status=='ok'].groupby('cache_id')):
            axes[2].axvspan((active.tick_ns.min()-g.tick_ns.iloc[0])/1e9,(active.tick_ns.max()-g.tick_ns.iloc[0])/1e9,
                           alpha=.15,color='green',label='Actual model inference windows' if number==0 else None)
        axes[2].legend()
        axes[3].plot(t,g.age_ms,label='SDK-read age ms')
        axes[3].scatter(t[g.fallback],g.age_ms[g.fallback],s=8,color='red',label='Gate fallback')
        axes[3].legend();axes[3].set_ylabel('Age [ms]');axes[3].set_xlabel('Physical recorded time [s]')
        for ax in axes:
            ax.grid(alpha=.2)
            tt=t.to_numpy();gaps=np.flatnonzero(g.tick_ns.diff().fillna(0).to_numpy()>50_000_000)
            for i in gaps:ax.axvspan(tt[i-1],tt[i],alpha=.1,color='gray')
        fig.suptitle(f"Episode {ep}: {g.task.iloc[0]} — PROVISIONAL, UNCALIBRATED thresholds; random model; no safety labels")
        fig.tight_layout();fig.savefig(out/f'episode_{ep:06d}.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True);p.add_argument('--processor',required=True)
    p.add_argument('--out',default='docs/taptic_approach_a_strategy1/replay')
    p.add_argument('--episodes',type=int,nargs='+',default=[0,48]);p.add_argument('--slow-ticks',type=int,default=2)
    p.add_argument('--gate-config',help='JSON GateConfig; defaults are a provisional engineering hypothesis')
    args=p.parse_args();torch.set_num_threads(2)
    config=GateConfig(**json.loads(Path(args.gate_config).read_text())) if args.gate_config else None
    run(args.root,args.processor,args.out,args.episodes,args.slow_ticks,config)
