"""CPU cache-ladder validation using original MoT classes and real TAPTIC data."""
import ast
import copy
from dataclasses import FrozenInstanceError
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from qwen_vla.cache_ladder import ActionCacheLadder, handover_for_split
from qwen_vla.flow_matching import flow_time
from taptic.timing import build_quality


def load_helpers():
    spec = importlib.util.spec_from_file_location('ladder_baseline_tests', Path(__file__).with_name('test_multisplit.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def head_method(name):
    """Read-only reference from the commit preceding this implementation."""
    import qwen_vla.modeling_vla as module
    source = subprocess.check_output(['git', 'show', '05c9328:qwen_vla/modeling_vla.py'], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3VLVLAModel')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = vars(module).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 '<pre-ladder:'+name+'>', 'exec'), namespace)
    return namespace[name]


def cache_equal(a, b):
    assert len(a.layers) == len(b.layers)
    for x, y in zip(a.layers, b.layers):
        assert torch.equal(x.keys, y.keys)
        assert torch.equal(x.values, y.values)


@pytest.fixture(scope='module')
def case():
    if not os.environ.get('TAPTIC_ROOT') or not os.environ.get('TAPTIC_PROCESSOR'):
        pytest.skip('set TAPTIC_ROOT and TAPTIC_PROCESSOR for real-data ladder tests')
    helpers = load_helpers()
    real = helpers.real.__wrapped__()
    source, ds, ids = real
    model, _ = helpers.make_model(real)
    # Two layers exercise complete-cache cloning, not only layer zero.
    second = copy.deepcopy(model.model.layers[0])
    second.self_attn.layer_idx = 1
    model.model.layers.append(second)
    model.config.num_hidden_layers = 2
    torch.nn.init.normal_(model.final_layer.mlp.fc2.weight, std=.04)
    torch.nn.init.normal_(model.final_layer_tactile.mlp.fc2.weight, std=.08)
    ep = ids[0]
    samples, available = source.tactile(ep)
    quality = build_quality(samples, available, True)
    times = source.clocks(ep)
    raw = np.stack(samples['values']).astype(np.float32).reshape(-1,5,120,3)
    best = None
    # Offline fixture selection finds a real changing payload within one slow
    # interval. It is not a gate and does not expose future tactile to slow flow.
    for index, (episode, frame) in enumerate(ds._rows):
        if episode != ep or frame+1 >= len(times):
            continue
        tick = int(times[frame])
        if times[frame+1]-tick > 150_000_000:
            continue
        i = int(np.searchsorted(available, tick, side='right')-1)
        j = int(np.searchsorted(available, min(tick+90_000_000, int(times[frame+1])-1), side='right')-1)
        if i < 0 or j <= i or quality['segment_id'][i] != quality['segment_id'][j] or tick-available[i] > 50_000_000:
            continue
        score = float(np.linalg.norm(raw[j]-raw[i]))
        if best is None or score > best[0]:
            best = score, index, i, j
    assert best is not None and best[0] > 0
    _, index, i, j = best
    item = ds[index]
    assert item['tactile_valid'].all()
    # Duplicate one unpadded anchor to exercise B=2 without unequal text lengths.
    b = ds.collate_fn([item, item])
    assert (b['attention_mask'] == 1).all()
    updated = torch.from_numpy((raw[j]/np.asarray([128.,128.,255.], np.float32)).reshape(5,360)).unsqueeze(0).repeat(2,1,1)
    assert not torch.equal(b['tactile_f6s'], updated)
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        slow, fast, pos, state = helpers.context(model, b)
        times_seen, forwards, updates = [], [], []
        hooks = [model.t_embedder.register_forward_pre_hook(lambda m,a: times_seen.append(a[0].clone())),
                 model.model.register_forward_pre_hook(lambda m,a: forwards.append(1)),
                 model.final_layer.register_forward_hook(lambda m,a,o: updates.append(o.clone()))]
        ladder = model.forward_flow_action_ladder(slow,pos,b['eps_r'].float(),
            attention_mask=b['attention_mask'],fast_embeds=fast,state_embeds=state)
        for hook in hooks:
            hook.remove()
    assert len(forwards) == 9 and len(updates) == 8
    assert [float(t[0]) for t in times_seen] == [flow_time(i).item() for i in range(9)]
    assert all(t.dtype == torch.float32 for t in times_seen)
    assert ladder.levels == (.6,.4,.2)
    # Reconstruct the solver accumulation to catch accidental rounding at capture.
    x = b['eps_r'].float()
    for step, v in enumerate(updates, start=1):
        x = x + torch.tensor(-.1, dtype=torch.float32)*v.float()
        if step in (4,6,8):
            assert torch.equal(x,ladder[(10-step)/10].integration_state_fp32)
    print('LADDER_BUILD',json.dumps(dict(episode=int(ep),frame=item['meta']['frame'],
        slow_tick_ns=item['meta']['observation_ns'],fast_tick_ns=int(available[j]),
        elapsed_to_fast_ms=(int(available[j])-item['meta']['observation_ns'])/1e6,
        initial_read_ns=int(available[i]),action_forwards=len(forwards),euler_updates=len(updates),
        cache_layers=len(ladder[.4].cached_kv.layers),token_counts=ladder[.4].token_counts,
        kv_bytes_per_snapshot=sum(layer.keys.numel()*layer.keys.element_size()+
                                  layer.values.numel()*layer.values.element_size()
                                  for layer in ladder[.4].cached_kv.layers),
        x_shape=list(ladder[.4].x_tau.shape),tactile_shape=list(updated.shape))))
    return model,b,slow,fast,pos,state,ladder,updated


@pytest.mark.parametrize('split', [.6,.4,.2])
@torch.no_grad()
@torch.autocast('cpu',dtype=torch.bfloat16)
def test_ladder_equivalence_coherence_and_endpoints(case,split):
    model,b,slow,fast,pos,state,ladder,updated = case
    s = ladder[split]
    kwargs = dict(attention_mask=b['attention_mask'],fast_embeds=fast,state_embeds=state,
                  split_step=s.handover.action_steps)
    # Reference does not share the new trajectory helper.
    x, cache, nact, tau = head_method('forward_flow_action_partial')(
        model,slow,pos,b['eps_r'].float(),**kwargs)
    assert tau == s.tau == split and nact == s.n_action_in_cache
    assert torch.equal(x,s.x_tau)
    cache_equal(cache,s.cached_kv)
    assert s.integration_state_fp32.dtype == torch.float32 and s.x_tau.dtype == torch.bfloat16
    assert not s.x_tau.requires_grad and not s.integration_state_fp32.requires_grad
    metadata = s.cache_metadata
    assert metadata['coherent'] and metadata['precision']=='fp32' and metadata['dt']==-.1
    assert metadata['fp32_time_grid'] == tuple(flow_time(i).item() for i in range(11))
    assert metadata['time_grid'][-1] == 0 and flow_time(10).item() == 0
    assert s.position_ids.shape == (3,2,metadata['length'])
    assert torch.equal(s.latent_position_ids,pos)
    expected_pos=model.model._extend_position_ids(pos,nact,0)
    assert torch.equal(s.position_ids,expected_pos)
    assert torch.equal(s.cache_position,torch.arange(metadata['length']))
    assert s.token_counts['total']==slow.shape[1]+nact
    assert nact==fast.shape[1]+state.shape[1]+1+16
    # Independently rebuild the action suffix from exported x_tau and tau.
    fresh = s.cached_kv
    fresh.crop(-nact)
    seq=torch.cat([fast,state,model.t_embedder(flow_time(s.handover.action_steps).expand(2)).unsqueeze(1),
                   model.x_embedder(s.x_tau)],dim=1)
    out=model.model(inputs_embeds=seq,position_ids=s.action_position_ids,past_key_values=fresh,use_cache=True,
        latent_indexes=torch.arange(0),action_indexes=torch.arange(nact),tactile_indexes=torch.arange(0))
    cache_equal(out.past_key_values,s.cached_kv)
    seen=[];velocities=[]
    hook=model.t_embedder.register_forward_pre_hook(lambda m,a:seen.append(a[0].clone()))
    velocity_hook=model.final_layer_tactile.register_forward_hook(lambda m,a,o:velocities.append(o.clone()))
    result=model.tactile_flow_continue_from_ladder(ladder,split,tactile_f6=b['tactile_f6s'])
    hook.remove()
    velocity_hook.remove()
    assert len(seen)==s.handover.tactile_steps==round(split*10)
    assert all(t.dtype==torch.float32 for t in seen)
    assert [float(t[0]) for t in seen]==[flow_time(i).item() for i in range(s.handover.action_steps,10)]
    endpoint=s.x_tau.float()
    for v in velocities:
        endpoint=endpoint+torch.tensor(-.1,dtype=torch.float32)*v.float()
    assert torch.equal(endpoint.to(torch.bfloat16),result)
    reference=head_method('tactile_flow_continue')(model,cache,pos,nact,x,tau,
                         tactile_f6=b['tactile_f6s'],split_step=s.handover.action_steps)
    assert torch.equal(result,reference)
    assert result.shape==(2,16,21) and torch.isfinite(result).all()
    print('LADDER_EQUIVALENCE',json.dumps(dict(tau=split,action_steps=s.handover.action_steps,
        tactile_steps=len(seen),state_max_abs=float((x.float()-s.x_tau.float()).abs().max()),
        output_max_abs=float((result.float()-reference.float()).abs().max()),
        output_rms=float(result.float().square().mean().sqrt()),
        kv_max_abs=0.,cache_length=metadata['length'])))


@torch.no_grad()
@torch.autocast('cpu',dtype=torch.bfloat16)
def test_immutability_and_updated_physical_fast_ticks(case):
    model,b,slow,fast,pos,state,ladder,updated=case
    before={s.tau:(s.x_tau,s.integration_state_fp32,s.cached_kv,s.position_ids,s.cache_metadata) for s in ladder.snapshots}
    outputs={}
    # Reverse and repeat split order on the same slow-tick ladder.
    for split in (.2,.6,.4,.2,.4,.6):
        a=model.tactile_flow_continue_from_ladder(ladder,split,tactile_f6=b['tactile_f6s'])
        changed=model.tactile_flow_continue_from_ladder(ladder,split,tactile_f6=updated)
        again=model.tactile_flow_continue_from_ladder(ladder,split,tactile_f6=b['tactile_f6s'])
        assert torch.equal(a,again)
        assert not torch.equal(a,changed), 'random model must actually respond to new real tactile'
        outputs[split]=float((a.float()-changed.float()).abs().max())
    # Mutate all public mutable exports, including KV contents and length.
    for s in ladder.snapshots:
        s.x_tau.zero_();s.integration_state_fp32.zero_()
        s.latent_position_ids.zero_();s.action_position_ids.zero_();s.position_ids.zero_()
        s.cache_position.zero_();s.cache_metadata['tau']=99;s.token_counts['total']=0
        exported=s.cached_kv
        exported.layers[0].keys.zero_();exported.layers[0].values.zero_();exported.crop(1)
        x,fp,cache,positions,metadata=before[s.tau]
        assert torch.equal(x,s.x_tau) and torch.equal(fp,s.integration_state_fp32)
        assert torch.equal(positions,s.position_ids) and metadata==s.cache_metadata
        cache_equal(cache,s.cached_kv)
        with pytest.raises(FrozenInstanceError):
            s.n_chunk=1
    with pytest.raises(FrozenInstanceError):
        ladder.snapshots=()
    # No tensor storage shared between levels or with original caller positions.
    assert len({s._x_tau.data_ptr() for s in ladder.snapshots})==3
    for layer in range(2):
        assert len({s._cache.layers[layer].keys.data_ptr() for s in ladder.snapshots})==3
        assert len({s._cache.layers[layer].values.data_ptr() for s in ladder.snapshots})==3
    assert all(s._latent_positions.data_ptr()!=pos.data_ptr() for s in ladder.snapshots)
    print('LADDER_FAST_TICKS',json.dumps(dict(updated_tactile_output_max_abs=outputs,
                                            all_snapshots_bitwise_preserved=True)))


@pytest.mark.parametrize('value',[0.,1.,.3,.5,.2001,float('nan'),float('inf'),True,None,'0.4'])
def test_invalid_split_rejection(value):
    with pytest.raises(ValueError,match='selected_split'):
        handover_for_split(value)


def test_fp32_split_selection():
    for split in (.2,.4,.6):
        assert handover_for_split(np.float32(split)).tau==split
    with pytest.raises(ValueError):
        ActionCacheLadder(())


@pytest.mark.parametrize('with_fast,with_state',[(False,False),(True,False),(False,True)])
@torch.no_grad()
@torch.autocast('cpu',dtype=torch.bfloat16)
def test_optional_tokens_and_1d_positions(case,with_fast,with_state):
    model,b,slow,fast,pos,state,_,updated=case
    one_d=pos[0]
    kwargs=dict(fast_embeds=fast if with_fast else None,state_embeds=state if with_state else None)
    ladder=model.forward_flow_action_ladder(slow,one_d,b['eps_r'].float(),**kwargs)
    for split in (.2,.4,.6):
        s=ladder[split]
        x,cache,nact,tau=head_method('forward_flow_action_partial')(
            model,slow,one_d,b['eps_r'].float(),split_step=s.handover.action_steps,**kwargs)
        assert torch.equal(s.x_tau,x)
        cache_equal(s.cached_kv,cache)
        assert s.position_ids.shape==(2,slow.shape[1]+nact)
        assert s.n_fast==(fast.shape[1] if with_fast else 0)
        assert s.n_state==(1 if with_state else 0)
        a=model.tactile_flow_continue_from_ladder(ladder,split,tactile_f6=updated)
        c=model.tactile_flow_continue(cache,one_d,nact,x,tau,tactile_f6=updated,split_step=s.handover.action_steps)
        assert torch.equal(a,c)


@torch.no_grad()
@torch.autocast('cpu',dtype=torch.bfloat16)
def test_fixed_api_regression_and_ladder_input_validation(case):
    model,b,slow,fast,pos,state,ladder,updated=case
    weights={k:v.clone() for k,v in model.state_dict().items()}
    kwargs=dict(attention_mask=b['attention_mask'],fast_embeds=fast,state_embeds=state)
    for precision in ('fp32','legacy_bf16'):
        old=head_method('forward_flow_action_partial')(model,slow,pos,b['eps_r'].float(),flow_time_precision=precision,**kwargs)
        new=model.forward_flow_action_partial(slow,pos,b['eps_r'].float(),flow_time_precision=precision,**kwargs)
        assert torch.equal(old[0],new[0]) and old[2:]==new[2:]
        cache_equal(old[1],new[1])
        a=head_method('tactile_flow_continue')(model,*[old[1],pos,old[2],old[0],old[3]],
              tactile_f6=b['tactile_f6s'],flow_time_precision=precision)
        c=model.tactile_flow_continue(new[1],pos,new[2],new[0],new[3],
              tactile_f6=b['tactile_f6s'],flow_time_precision=precision)
        assert torch.equal(a,c)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    for extra in (dict(num_steps_total=20),dict(flow_time_precision='legacy_bf16')):
        with pytest.raises(ValueError):
            model.forward_flow_action_ladder(slow,pos,b['eps_r'].float(),**extra)
    with pytest.raises(ValueError,match='position'):
        model.forward_flow_action_ladder(slow,pos[...,:-1],b['eps_r'].float())
    with pytest.raises(ValueError,match='noise shape'):
        model.forward_flow_action_ladder(slow,pos,b['eps_r'][:,:8].float())
    padded=b['attention_mask'].clone();padded[:,0]=0
    with pytest.raises(ValueError,match='unpadded'):
        model.forward_flow_action_ladder(slow,pos,b['eps_r'].float(),attention_mask=padded)
    with pytest.raises(TypeError):
        model.tactile_flow_continue_from_ladder({},.4,tactile_f6=updated)
    for invalid in (.3,float('nan')):
        with pytest.raises(ValueError):
            model.tactile_flow_continue_from_ladder(ladder,invalid,tactile_f6=updated)
    model.train()
    try:
        with pytest.raises(ValueError,match='eval'):
            model.forward_flow_action_ladder(slow,pos,b['eps_r'].float())
        with pytest.raises(ValueError,match='eval'):
            model.tactile_flow_continue_from_ladder(ladder,.4,tactile_f6=updated)
    finally:
        model.eval()
    print('LADDER_FIXED_REGRESSION',json.dumps(dict(fp32_bitwise_equal=True,legacy_bf16_bitwise_equal=True,
                                                  weights_unchanged=True)))
