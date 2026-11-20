"""Episode-59-only offline command-reference replay. Tiny RANDOM weights, CPU.

Recorded-row timestamps are reference labels, not verified actuation deadlines.
Latency is measured CPU wall duration mapped into the replay clock; it does NOT
recreate historical latency or establish a real-time bound. Limits below are
permissive diagnostics only. No hardware APIs are imported or called.
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
from scipy.spatial.transform import Rotation
from transformers import AutoProcessor

from qwen_vla.taptic_dataset import TRexTapticDataset
from scripts.replay_taptic_approach_a import tiny_model, context, episode_samples
from taptic.action_executor import TimedActionChunk, PhysicalTrajectory, ActionExecutor, Limits, Transition
from taptic.actions import rotation_angle_error, matrix_to_rotvec_gd
from taptic.data import denormalize
from taptic.inference import SlowAnchor, TaxelObservation
from taptic.online_contact import ContactNormalization, OnlineContactExtractor

MS=1_000_000


def diagnostic_limits():
    """Not Panda/XHand safety limits. Allows viewing arbitrary random predictions."""
    return Limits((-100.,)*3,(100.,)*3,(-100.,)*12,(100.,)*12,
                  10000.,10000.,10000.,1e14,1e14,1e14,100.,4.,100.)


def cache_digest(ladder):
    h=hashlib.sha256()
    for snap in ladder.snapshots:
        for layer in snap.cached_kv.layers:
            for v in (layer.keys,layer.values):h.update(v.float().numpy().tobytes())
        h.update(snap.integration_state_fp32.numpy().tobytes())
    return h.hexdigest()


def run(root,processor,out,make_plots=True):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    cfg=SimpleNamespace(lerobot_root=root,taptic_stats='docs/taptic_train_stats.json',action_dim=21,action_chunk=16,
        use_tactile_vec=1,use_tactile_deform=0,use_tactile_code=0,use_tactile_vqvae=0,taptic_tactile_input='taxels',
        use_robot_state=1,use_flare=0,image_size=(56,56),tactile_delay_offsets=[0],cascaded_split_step=6)
    ds=TRexTapticDataset(cfg,AutoProcessor.from_pretrained(processor,local_files_only=True),SimpleNamespace(print=print))
    src=ds.source;ep=59
    meta=src.episodes.loc[ep]
    if meta.source_task!='dosing_pipette' or bool(meta.tactile_interpolated):
        raise ValueError('expected verified native episode 59 dosing_pipette')
    files=[src.root/f'tactile_raw/episode_{ep:06d}_{suffix}.parquet' for suffix in ('right','frames')]
    files+=list((src.root/'data').rglob('*.parquet'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    frames=src.frames(ep);state=np.stack(frames['observation.state']);raw=np.stack(frames.action)
    clocks=src.clocks(ep)
    fr=pd.read_parquet(files[1]).sort_values('frame_index')
    available=np.maximum(clocks,fr.add_frame_ns.to_numpy(np.int64))
    # Exactly 200ms scheduler; latest causally AVAILABLE frame, not nearest future.
    first=71;s0=int(available[first]);scheduled=[s0,s0+200*MS]
    indices={t:i for i,(e,t) in enumerate(ds._rows) if e==ep}
    samples=episode_samples(src,ep)
    norm=ContactNormalization.from_analysis('docs/taptic_risk_analysis/train_normalization.json',src.episodes,src.identity)
    extractor=OnlineContactExtractor(norm);extractor.start_episode(ep);extractor.ingest(samples)
    model=tiny_model(processor)
    weight_hash=lambda:hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in model.state_dict().values())).hexdigest()
    weights_before=weight_hash()
    model_id='tiny-random-seed321:'+weights_before
    config_id=hashlib.sha256(json.dumps(vars(cfg),sort_keys=True).encode()).hexdigest()
    norm_id=hashlib.sha256(Path(cfg.taptic_stats).read_bytes()).hexdigest()
    physical_clock='episode59:recorded_CLOCK_MONOTONIC'
    executor=ActionExecutor(diagnostic_limits(),Transition(60*MS))
    executor.reset(ep)
    chunks=[];gt=[];commands=[];events=[];anchors=[];roundtrips=[];cache_ok=True
    latest_event=-1;future=0;model_predictions=[]
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        for k,S in enumerate(scheduled):
            t=int(np.searchsorted(available,S,side='right')-1)
            if t not in indices:raise ValueError('scheduled anchor has no full valid training window')
            item=ds[indices[t]];batch=ds.collate_fn([item])
            np.testing.assert_array_equal(item['meta']['action_timestamps_ns'],clocks[t:t+16])
            extractor.at(S)
            if k == 0:
                executor.reset(ep, extractor.segment)
            started=time.perf_counter_ns()
            ctx=context(model,batch)
            ladder=model.forward_flow_action_ladder(**ctx)
            ready=S+time.perf_counter_ns()-started
            digest=cache_digest(ladder)
            anchor=SlowAnchor(ep,S,int(clocks[t]),int(available[t]),ready,int(clocks[t+15]),
                              tuple(state[t,34:37]),tuple(state[t,37:40]))
            anchors.append(asdict(anchor)|{'frame':t,'target_ns':clocks[t:t+16].tolist()})
            def timed(a,avail,split):
                return TimedActionChunk(a,S,int(clocks[t]),ready,avail,tuple(clocks[t:t+16]),
                    anchor.ref_position,anchor.ref_rotvec_gd,ep,f'59:{S}:{k}',split,norm_id,config_id,
                    model_id,'recorded_row',physical_clock,sensor_segment=extractor.segment)
            # Ground truth is only a retrospective label comparison, not inference.
            truth=PhysicalTrajectory(timed(item['action_chunk'],ready,.4));gt.append(truth)
            reconstructed=np.concatenate([truth.position,matrix_to_rotvec_gd(truth.rotation),truth.joints],axis=1)
            roundtrips.append((float(np.max(np.abs(reconstructed[:,:3]-raw[t:t+16,:3]))),
                float(rotation_angle_error(reconstructed[:,3:6],raw[t:t+16,3:6]).max()),
                float(np.max(np.abs(truth.as_actions21()-item['action_chunk'])))))
            # Physical 50ms ticks independent of Euler-step count. Missed deadlines
            # are explicit; do not shift prediction/waypoint times to hide latency.
            for j in range(4):
                F=S+(j+1)*50*MS
                if k==0 and F>=scheduled[1]:continue
                if F<ready or F<latest_event:
                    events.append({'tick_ns':F,'status':'missed_compute_deadline','cache':k});continue
                contact=extractor.at(F)
                last_read=contact['quality']['latest_read_ns']
                future+=int(last_read>F or contact['quality']['latest_available_ns']>F)
                obs=TaxelObservation()(extractor,1,torch.device('cpu'))
                if not obs or not contact['quality']['fresh']:
                    events.append({'tick_ns':F,'status':'missing_tactile','cache':k});continue
                split=(.2,.4,.6)[len(chunks)%3]
                start_prediction=time.perf_counter_ns()
                prediction=model.tactile_flow_continue_from_ladder(ladder,split,**obs)
                completed=F+time.perf_counter_ns()-start_prediction
                # Model start/finish mapped onto the historical clock explicitly.
                result=dict(status='ok',actions=denormalize(prediction.float().numpy(),ds.action_stats),
                    tick_ns=F,slow_anchor_ns=S,cache_identity=f'59:{S}:{k}',gate={'selected_split':split},
                    reference={'observation_ns':anchor.observation_ns,'position':anchor.ref_position,'rotvec_gd':anchor.ref_rotvec_gd})
                c=TimedActionChunk.from_prediction(result,anchor,prediction_available_ns=completed,
                    target_ns=tuple(clocks[t:t+16]),normalization_id=norm_id,config_id=config_id,
                    model_version=model_id,timing_semantics='recorded_row',clock_id=physical_clock,
                    sensor_segment=extractor.segment)
                chunks.append(c);model_predictions.append(result['actions'][0])
                accepted=executor.submit(c,completed)
                cmd=executor.command(completed) if accepted.accepted else None
                if cmd and cmd.accepted:commands.append(cmd.command)
                events.append({'tick_ns':F,'available_ns':completed,'cache':k,'split':split,
                               'status':accepted.status,'command_status':cmd.status if cmd else None,
                               'last_tactile_read_ns':last_read,'tactile_steps':round(split*10)})
                latest_event=completed
                # More frequent reference queries do NOT create new measurements
                # or run additional tactile denoising. Keep them before next tick.
                until = F+50*MS-1
                if k == 0:
                    until = min(until, scheduled[1]-1)
                for app in range(completed+10*MS, until+1, 10*MS):
                    ref=executor.command(app)
                    if ref.accepted:commands.append(ref.command)
                    else:events.append({'tick_ns':app,'status':'reference_rejected:'+ref.status,'cache':k})
                    latest_event=app
                cache_ok &= cache_digest(ladder)==digest
            # Command selected just before the next scheduled slow anchor, if valid.
            end=scheduled[1]-1 if k==0 else S+250*MS
            if end>latest_event:
                c=executor.command(end)
                if c.accepted:commands.append(c.command)
                latest_event=end
    if not chunks:raise RuntimeError('no predictions; inspect deadline/quality events')
    r=np.array(roundtrips)
    # An empirical episode envelope is a diagnostic counterexample, not hardware
    # limits and never a fitted deployment safety boundary.
    outside_envelope=0
    for c in chunks:
        p=PhysicalTrajectory(c)
        outside_envelope+=int(np.any(p.position < raw[:,:3].min(0)-.02) or
                              np.any(p.position > raw[:,:3].max(0)+.02) or
                              np.any(p.joints < raw[:,6:].min(0)-.05) or
                              np.any(p.joints > raw[:,6:].max(0)+.05))
    result={'episode':59,'task':str(meta.source_task),'object':str(meta.object),'source_episode':int(meta.source_episode_index),
        'frames':len(frames),'native_tactile':True,'frame_interval_ms_min_median_max':np.quantile(np.diff(clocks)/MS,[0,.5,1]).tolist(),
        'roundtrip_position_max_m':float(r[:,0].max()),'roundtrip_angle_max_rad':float(r[:,1].max()),
        'roundtrip_action21_max_abs':float(r[:,2].max()),'random_predictions':len(chunks),
        'random_prediction_shape':list(np.stack(model_predictions).shape),
        'selected_splits':sorted(set(c.selected_split for c in chunks)),
        'cache_builds':len(anchors),'cache_immutable':bool(cache_ok),'future_tactile_accesses':future,
        'command_references':len(commands),'random_plan_rejections':sum(e['status'] not in ('plan_accepted','missed_compute_deadline','missing_tactile') for e in events),
        'missed_deadlines':sum(e['status']=='missed_compute_deadline' for e in events),
        'weights_unchanged':weights_before==weight_hash(),
        'random_chunks_outside_recorded_episode_envelope':outside_envelope,
        'source_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
        'model_id':model_id,'normalization_id':norm_id,'config_id':config_id,
        'limits_status':'PERMISSIVE DIAGNOSTIC LIMITS; NOT HARDWARE SAFETY LIMITS',
        'limits':asdict(diagnostic_limits()),'transition_ms':60,
        'latency_convention':'measured CPU wall duration mapped into historical monotonic clock; NOT historical/real-time latency',
        'timing_convention':'recorded_row labels; no measured Panda actuator application deadline'}
    if commands:
        p=np.stack([c.position for c in commands]);q=np.stack([c.quaternion_xyzw for c in commands]);h=np.stack([c.joints for c in commands])
        dt=np.diff([c.t_ns for c in commands])/1e9
        result['command_position_step_max_m']=float(np.linalg.norm(np.diff(p,axis=0),axis=1).max())
        result['command_cartesian_speed_max_m_s']=float((np.linalg.norm(np.diff(p,axis=0),axis=1)/dt).max())
        result['command_joint_speed_max_rad_s']=float((np.abs(np.diff(h,axis=0))/dt[:,None]).max())
        result['command_angular_speed_max_rad_s']=float(((Rotation.from_quat(q[1:])*Rotation.from_quat(q[:-1]).inv()).magnitude()/dt).max())
        np.savez_compressed(out/'references.npz',t_ns=np.array([c.t_ns for c in commands]),position=p,quaternion_xyzw=q,joints=h,
            actions21=np.stack(model_predictions),target_ns=np.array([c.target_ns for c in chunks]),
            prediction_available_ns=np.array([c.prediction_available_ns for c in chunks]),
            prediction_split=np.array([c.selected_split for c in chunks]),
            command_split=np.array([c.selected_split for c in commands]),
            command_cache_id=np.array([c.cache_id for c in commands]))
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'anchors.json').write_text(json.dumps(anchors,indent=2))
    (out/'events.json').write_text(json.dumps(events,indent=2))
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    if make_plots:plot(out,gt,chunks,commands,scheduled)
    return result


def plot(out,gt,chunks,commands,scheduled):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin=gt[0].chunk.observation_ns
    sec=lambda t:(np.asarray(t,dtype=np.int64)-origin)/1e9
    # All components, separate panels; no inference-error metric implied.
    for kind in ('position','orientation','joints'):
        n={'position':3,'orientation':3,'joints':12}[kind]
        fig,axes=plt.subplots(n,1,figsize=(12,max(6,n*1.3)),sharex=True,squeeze=False)
        def value(p):
            if kind=='orientation':return Rotation.from_matrix(p.rotation).as_rotvec()
            return getattr(p,kind)
        for ki,g in enumerate(gt):
            for j,ax in enumerate(axes[:,0]):
                ax.plot(sec(g.chunk.target_ns),value(g)[:,j],'k.-',alpha=.6,label='recorded targets (row-time labels)' if ki==0 else None)
        for ki,c in enumerate(chunks):
            tr=PhysicalTrajectory(c)
            grid=np.linspace(max(c.prediction_available_ns,tr.start_ns),tr.end_ns,70).astype(np.int64)
            dense=[tr.at(int(t)) for t in grid]
            densevalue=(Rotation.from_quat(np.stack([v.quaternion_xyzw for v in dense])).as_rotvec() if kind=='orientation'
                        else np.stack([getattr(v,kind) for v in dense]))
            for j,ax in enumerate(axes[:,0]):
                ax.plot(sec(c.target_ns),value(tr)[:,j],'x',color='tab:orange',alpha=.2,label='RANDOM-model waypoints' if ki==0 else None)
                ax.plot(sec(grid),densevalue[:,j],color='tab:orange',alpha=.15,label='time interpolation, not measurements' if ki==0 else None)
        if commands:
            v=(Rotation.from_quat(np.stack([c.quaternion_xyzw for c in commands])).as_rotvec() if kind=='orientation'
               else np.stack([getattr(c,kind) for c in commands]))
            for j,ax in enumerate(axes[:,0]):ax.plot(sec([c.t_ns for c in commands]),v[:,j],'b.-',label='emitted offline references')
        for j,ax in enumerate(axes[:,0]):
            ax.set_ylabel(f'{kind} {j}'+(' (m)' if kind=='position' else ' (rad)'));ax.grid(alpha=.2)
            for s in scheduled:ax.axvline(sec(s),color='green',ls='--',alpha=.5)
        axes[0,0].legend(fontsize=7);axes[-1,0].set_xlabel('physical seconds from original observation')
        fig.suptitle(f'Episode 59 — {kind}; random weights, diagnostic bounds; green = slow cache (200ms)')
        fig.tight_layout();fig.savefig(out/f'{kind}.png',dpi=130);plt.close(fig)
    if len(commands) < 2:
        return
    fig,ax=plt.subplots(4,1,figsize=(11,9),sharex=True)
    t=sec([c.t_ns for c in commands]);p=np.stack([c.position for c in commands]);dt=np.diff(t)
    ax[0].step(t,[c.selected_split for c in commands],where='post');ax[0].set_ylabel('selected tau')
    ax[1].plot(t[1:],np.linalg.norm(np.diff(p,axis=0),axis=1),'o-');ax[1].set_ylabel('position step (m)')
    ax[2].plot(t[1:],np.linalg.norm(np.diff(p,axis=0),axis=1)/dt,'o-');ax[2].set_ylabel('Cartesian m/s')
    h=np.stack([c.joints for c in commands]);ax[3].plot(t[1:],np.max(np.abs(np.diff(h,axis=0)),axis=1)/dt,'o-');ax[3].set_ylabel('max joint rad/s')
    for a in ax:
        for s in scheduled:a.axvline(sec(s),color='green',ls='--')
        a.grid(alpha=.2)
    ax[-1].set_xlabel('physical seconds from original observation');fig.suptitle('Offline references — split/cache transitions, no physical safety claim')
    fig.tight_layout();fig.savefig(out/'transitions.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--processor',required=True)
    p.add_argument('--out',default='docs/taptic_action_executor/replay')
    args=p.parse_args();print(json.dumps(run(args.root,args.processor,args.out),indent=2))
