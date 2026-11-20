"""Online causality, provisional gate state, replay and real-model integration."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.analyze_taptic_risk import causal_features
from taptic.online_contact import ContactNormalization, NativeSample, OnlineContactExtractor
from taptic.split_gate import GateConfig, SplitSelector, select_split
from taptic.inference import ApproachAInference, SlowAnchor, SDKForceObservation

NS=1_000_000
SCALES=ContactNormalization((10,)*5,((100,)*5,)*3)


def sample(ms,value=1,**kwargs):
    force=np.zeros((5,3));force[:,0]=value
    return NativeSample(0,ms*NS,kwargs.pop('available_ns',ms*NS),force,
                        sequence=kwargs.pop('sequence',ms),taxels=np.ones((5,120,3)),**kwargs)


def extractor(samples=(),**kwargs):
    e=OnlineContactExtractor(SCALES,**kwargs);e.start_episode(0);e.ingest(samples)
    return e


def test_online_offline_exact_irregular_and_future_cutoffs():
    rng=np.random.default_rng(91)
    t=np.cumsum(rng.integers(8,19,size=70))*NS
    f=rng.integers(-9,20,size=(70,5,3)).astype(float)
    f[4:8]=f[3]  # real repeated observations remain valid
    breaks=np.zeros(70,bool);breaks[[0,31]]=True
    offline=causal_features(t,f,breaks)
    e=extractor([NativeSample(0,int(t[i]),int(t[i]),f[i],sequence=i,boundary=bool(breaks[i])) for i in range(70)])
    errors=[]
    for i,tick in enumerate(t):
        out=e.at(int(tick))
        assert out['quality']['fresh']
        expected=float(np.clip(offline['magnitude'][i]/10,0,1).max())
        assert out['features']['M']==expected
        for w in (50,100,200):
            ref=offline['windows'][w]
            assert out['quality']['feature_valid']['R'+str(w)]==bool(ref['valid'][i])
            delta=np.max(np.abs(np.asarray(out['positive_variation_rate'][str(w)])-ref['positive_variation_rate'][i]))
            errors.append(float(delta));assert delta<1e-9
    print('ONLINE_OFFLINE_AGREEMENT',json.dumps({'max_absolute_counts_per_s':max(errors),'read_cutoffs':len(t)}))
    # A cutoff between reads agrees with an offline held-value diagnostic row.
    q=extractor([sample(i,float(i%7),sequence=j) for j,i in enumerate(range(0,301,10))])
    out=q.at(295*NS)
    tt=np.r_[np.arange(0,291,10)*NS,295*NS]
    ff=np.stack([sample(int(x//NS),float((x//NS)%7)).force for x in tt[:-1]]+[sample(290,float(290%7)).force])
    off=causal_features(tt,ff,[True]+[False]*(len(tt)-1))
    np.testing.assert_allclose(out['positive_variation_rate']['100'],off['windows'][100]['positive_variation_rate'][-1],atol=1e-10)


def test_repeats_partial_invalid_stale_and_late():
    e=extractor([sample(i,2,sequence=j) for j,i in enumerate(range(0,221,10))])
    a=e.at(20*NS)
    assert a['quality']['fresh'] and not a['quality']['feature_valid']['R50']
    assert a['quality']['coverage']['50']==.4
    b=e.at(220*NS)
    assert all(b['quality']['feature_valid'].values())
    assert b['features']['R200']==0
    assert not e.at(271*NS)['quality']['fresh']
    with pytest.raises(ValueError):e.at(270*NS)
    delayed=extractor([sample(10,9,sequence=0,available_ns=40*NS),sample(20,2,sequence=1)])
    assert delayed.at(19*NS)['quality']['latest_read_ns'] is None
    assert delayed.at(20*NS)['features']['M']==.2
    assert delayed.at(40*NS)['quality']['late_discarded']==1
    assert delayed.at(40*NS)['features']['M']==.2
    delayed.ingest([sample(50,np.nan,sequence=2)])
    assert not delayed.at(50*NS)['quality']['fresh']
    delayed.ingest([sample(60,1,sequence=3)])
    assert not delayed.at(60*NS)['quality']['feature_valid']['R50']
    with pytest.raises(ValueError):delayed.ingest([NativeSample(1,70*NS,70*NS,np.ones((5,3)))])
    delayed.start_episode(1)
    assert delayed.at(100*NS)['quality']['latest_read_ns'] is None


@pytest.mark.parametrize('change',[dict(reset_sequence=1),dict(sequence=500),dict(read_errors=1),dict(dropped=1),dict(boundary=True),dict(valid=False),dict(native=False)])
def test_boundaries_clear_derivatives(change):
    e=extractor([sample(i,1,sequence=j) for j,i in enumerate(range(0,211,10))])
    old=e.at(210*NS)['quality']['segment']
    args=dict(sequence=22);args.update(change)
    e.ingest([sample(220,100,**args)])
    out=e.at(220*NS)
    assert out['quality']['segment']!=old
    assert not out['quality']['feature_valid']['R50']
    assert out['positive_variation_rate']['50']==[0.]*5


def test_gap_duplicate_and_force_history_padding():
    e=extractor([sample(0,sequence=0),sample(10,sequence=1),sample(80,sequence=2)])
    a=e.at(10*NS)
    e.ingest([sample(10,sequence=1)])
    assert e.at(10*NS)['quality']['duplicates']==1
    with pytest.raises(ValueError):e.ingest([sample(10,99,sequence=1)])
    b=e.at(80*NS)
    assert b['quality']['segment']!=a['quality']['segment']
    assert b['quality']['boundary_reason']=='gap'
    hist,n=e.force_history(16)
    assert hist.shape==(16,5,3) and n==1
    assert not b['quality']['feature_valid']['R50']
    with pytest.raises(ValueError):sample(0,available_ns=-1)
    with pytest.raises(ValueError):e.at(80.5*NS)


def features(v):
    return dict(M=v,R50=v,R100=v,R200=v)


def quality(valid=True):
    return dict(fresh=valid,feature_valid={k:valid for k in features(0)},segment=1)


def test_gate_mapping_ema_hysteresis_dwell_fallback_and_determinism():
    c=GateConfig(mode='magnitude',thresholds=(.3,.7),levels=(.6,.4,.2))
    a,b=SplitSelector(c),SplitSelector(c)
    for i,v in enumerate([.1,.5,.9]):
        x=a.select_split(features(v),quality(),i)
        assert x==b.select_split(features(v),quality(),i)
        assert x['selected_split']==(.6,.4,.2)[i]
        assert a.select_split(features(v),quality(),i)==x
    gate=SplitSelector(GateConfig(mode='magnitude',ema_alpha=.5,hysteresis=.05,min_dwell_ns=10))
    assert gate.select_split(features(.1),quality(),0)['selected_split']==.2
    assert gate.select_split(features(1),quality(),1)['reason']=='dwell'
    assert gate.select_split(features(1),quality(),11)['selected_split']==.6
    fb=gate.select_split(features(0),quality(False),12)
    assert fb['fallback'] and fb['selected_split']==.4 and fb['state']['ema'] is None
    assert SplitSelector(GateConfig(fixed_split=.6)).select_split(features(0),quality(False),0)['selected_split']==.6
    with pytest.raises(ValueError):gate.select_split(features(0),quality(),11)
    with pytest.raises(ValueError):gate.select_split(features(.9),quality(),12)
    for kw in (dict(mode='learned'),dict(thresholds=(.8,.2)),dict(ema_alpha=0),dict(levels=(.2,.2,.4))):
        with pytest.raises(ValueError):GateConfig(**kw)
    state=SplitSelector(c)
    assert select_split(features(.9),quality(),c,tick_ns=0,state=state)==.2
    assert state.last_result['raw_score']==.9
    with pytest.raises(ValueError):select_split(features(0),quality(),GateConfig(),tick_ns=1,state=state)
    # A partial unused R200 window does not invalidate an M-only selector.
    q=quality();q['feature_valid']['R200']=False
    assert not SplitSelector(c).select_split(features(.9),q,0)['fallback']


class FakeModel:
    def __init__(self):self.builds=0;self.continuations=0
    def forward_flow_action_ladder(self,**kwargs):
        self.builds+=1
        return {s:SimpleNamespace(_x_tau=torch.zeros(1,16,21)) for s in (.2,.4,.6)}
    def tactile_flow_continue_from_ladder(self,ladder,split,**kwargs):
        self.continuations+=1
        return torch.full((1,16,21),split)


def anchor(s=210,ready=None):
    return SlowAnchor(0,s*NS,s*NS,s*NS,(ready or s)*NS,(s+100)*NS,(.1,.2,.3),(0,0,0))


def engine(config=GateConfig(),ready=None):
    e=extractor([sample(i,1,sequence=j) for j,i in enumerate(range(0,211,10))])
    model=FakeModel();p=ApproachAInference(model,e,config,{'min':[-1]*21,'max':[1]*21})
    p.slow_tick(anchor(ready=ready))
    return p,model


def test_replay_expiry_readiness_missing_reset_and_no_rebuild():
    p,m=engine(GateConfig(mode='magnitude',thresholds=(.2,.7)))
    a=p.fast_tick(0,210*NS)
    b=p.fast_tick(0,220*NS,[sample(220,9,sequence=22)])
    assert a['gate']['selected_split']==.2 and b['gate']['selected_split']==.6
    assert a['cache_id']==b['cache_id'] and m.builds==1
    assert b['actions'].shape==(1,16,21) and np.isfinite(b['actions']).all()
    assert b['reference']==a['reference'] and not b['action_error_supported']
    assert p.fast_tick(0,275*NS)['status']=='missing_or_stale_tactile'
    assert p.fast_tick(0,310*NS)['status']=='cache_expired'
    with pytest.raises(ValueError):p.fast_tick(0,300*NS)
    with pytest.raises(ValueError):p.fast_tick(1,320*NS)
    with pytest.raises(NotImplementedError):p.action_error(None)
    p,m=engine(ready=220)
    assert p.fast_tick(0,215*NS)['status']=='cache_not_ready'
    assert p.fast_tick(0,220*NS,[sample(220,1,sequence=22)])['status']=='ok'
    out=p.fast_tick(0,230*NS,[sample(230,1,sequence=23,reset_sequence=1)])
    assert out['actions'] is None and out['status'].startswith('sensor_boundary') and m.builds==1
    assert p.fast_tick(0,240*NS,[sample(240,1,sequence=24,reset_sequence=1)])['actions'] is None
    p.start_episode(1)
    assert p.fast_tick(1,250*NS)['status']=='no_cache'


def test_pipeline_future_payloads_and_stable_repeated_diagnostics():
    p,m=engine(GateConfig(mode='combined'))
    p.extractor.ingest([sample(230,100,sequence=23,available_ns=250*NS)])
    a=p.fast_tick(0,220*NS,[sample(220,1,sequence=22)])
    assert a['contact']['quality']['latest_read_ns']==220*NS
    assert a['gate']['raw_score']<.2
    b=p.fast_tick(0,220*NS)
    assert a['contact']==b['contact'] and a['gate']==b['gate']
    assert np.array_equal(a['actions'],b['actions']) and a['reference']==b['reference']
    late=p.fast_tick(0,250*NS)
    assert late['contact']['quality']['latest_read_ns']==230*NS
    assert late['contact']['quality']['available_age_ns']==0
    assert late['contact']['quality']['age_ns']==20*NS
    assert m.builds==1


def test_existing_sdk_force_observation_and_partial_fallback():
    e=extractor([sample(i,2,sequence=j) for j,i in enumerate(range(0,101,10))])
    e.at(100*NS)
    stats=SimpleNamespace(normalize=lambda x:x/10)
    obs=SDKForceObservation(stats,16,True)(e,2,'cpu')
    assert obs['tactile_f6'].shape==(2,5,3)
    assert obs['tactile_f6_history'].shape==(2,16,5,3)
    assert obs['tactile_f6'][0,0,0]==pytest.approx(.2)
    out=e.at(100*NS)
    decision=SplitSelector(GateConfig(mode='rate',rate_windows=(200,))).select_split(out['features'],out['quality'],100*NS)
    assert decision['fallback'] and decision['selected_split']==.4
    p,m=engine(GateConfig(mode='magnitude'))
    missing=NativeSample(0,220*NS,220*NS,np.ones((5,3)),sequence=22)
    out=p.fast_tick(0,220*NS,[missing])
    assert out['gate']['fallback'] and out['gate']['selected_split']==.4
    assert out['status']=='missing_model_tactile' and m.continuations==0


def test_normalizer_training_provenance_and_immutability():
    import pandas as pd
    p=Path('docs/taptic_risk_analysis/train_normalization.json')
    if p.exists():
        summary=json.loads((p.parent/'summary.json').read_text())
        ids=json.loads(p.read_text())['train_episodes']
        meta=pd.DataFrame({'suggested_split':['train']*len(ids),'tactile_interpolated':[False]*len(ids)},index=ids)
        n=ContactNormalization.from_analysis(p,meta,summary['source_identity'])
        assert len(n.train_episodes)==512
        meta.loc[ids[0],'suggested_split']='val'
        with pytest.raises(ValueError):ContactNormalization.from_analysis(p,meta,summary['source_identity'])
        with pytest.raises(ValueError):ContactNormalization.from_analysis(p,meta,'wrong')
    with pytest.raises(ValueError):ContactNormalization((0,)*5,((1,)*5,)*3)


def test_real_model_end_to_end_and_fixed_regression():
    spec=importlib.util.spec_from_file_location('real_ladder_test',Path(__file__).with_name('test_cache_ladder.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model,b,slow,fast,pos,state,ladder,updated=mod.case.__wrapped__()
    # Current input is real native raw taxels; causal force descriptors use a
    # synthetic-controlled prior history here. The replay script tests both real.
    e=extractor([sample(i,1,sequence=j) for j,i in enumerate(range(0,211,10))])
    raw=b['tactile_taxels'][0].numpy()
    p=ApproachAInference(model,e,GateConfig(),{'min':[-1]*21,'max':[1]*21})
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        p.slow_tick(anchor(),inputs_embeds=slow,position_ids=pos,noise=b['eps_r'].float(),fast_embeds=fast,state_embeds=state)
        reference={s.tau:s.cached_kv for s in p.ladder.snapshots}
        out=p.fast_tick(0,220*NS,[NativeSample(0,220*NS,220*NS,np.ones((5,3)),sequence=22,taxels=raw)])
        fixed=model.tactile_flow_continue_from_ladder(p.ladder,.4,tactile_f6=b['tactile_f6s'])
        assert torch.equal(out['norm_actions'],fixed)
        selected=[]
        # A new selector state for an explicit engineering mapping experiment.
        p.selector=SplitSelector(GateConfig(mode='magnitude',thresholds=(.2,.7)))
        for k,(value,wanted) in enumerate(((1,.2),(5,.4),(9,.6))):
            tick=(230+k*10)*NS
            out=p.fast_tick(0,tick,[sample(230+k*10,value,sequence=23+k)])
            assert out['gate']['selected_split']==wanted and out['actions'].shape==(2,16,21)
            selected.append(out['tactile_steps'])
        assert selected==[2,4,6]
        for s in p.ladder.snapshots:mod.cache_equal(reference[s.tau],s.cached_kv)
        assert p.cache_id==1
        # A parameter update invalidates cache identity; it never silently reuses
        # KV from another weight version, nor launches an automatic replan.
        next(model.parameters()).add_(.001)
        assert p.fast_tick(0,270*NS)['status']=='model_changed_requires_scheduled_slow_tick'
    print('APPROACH_A_REAL_MODEL',json.dumps(dict(fixed_bitwise_equal=True,selected_tactile_steps=selected,cache_builds=p.cache_id,shape=list(out['actions'].shape))))
