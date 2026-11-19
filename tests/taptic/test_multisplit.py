"""CPU numerical, causal-data and multi-split forward/backward regression tests."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from qwen_vla.flow_matching import Handover, flow_time, select_handover, tactile_training_pair
from taptic.timing import build_quality, latest_quality
from taptic.data import tactile_at, tactile_stream_between


@pytest.fixture(scope='module')
def real():
    if not os.environ.get('TAPTIC_ROOT') or not os.environ.get('TAPTIC_PROCESSOR'):
        pytest.skip('set TAPTIC_ROOT and TAPTIC_PROCESSOR for real-data tests')
    spec = importlib.util.spec_from_file_location('baseline_tests', Path(__file__).with_name('test_adapter.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.setup.__wrapped__()


def test_integer_time_grid_and_batch_selection():
    times = torch.stack([flow_time(i) for i in range(11)])
    assert times.dtype == torch.float32
    torch.testing.assert_close(times, torch.tensor([1,.9,.8,.7,.6,.5,.4,.3,.2,.1,0]), atol=1e-7, rtol=0)
    cfg = SimpleNamespace(cascaded_split_mode='fixed', cascaded_split_step=6)
    assert {select_handover(cfg).tau for _ in range(10)} == {.4}
    cfg.cascaded_split_mode = 'multi'
    gen = torch.Generator().manual_seed(123)
    draws = [select_handover(cfg, generator=gen) for _ in range(120)]
    assert {h.tau for h in draws} == {.2,.4,.6}
    assert all(h.action_steps+h.tactile_steps == 10 for h in draws)
    assert all(isinstance(h.action_steps, int) for h in draws)
    for bad in [3, 5, 10]:
        with pytest.raises(ValueError):
            Handover(bad)


def _rank_select(rank, init, output):
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=2)
    cfg = SimpleNamespace(cascaded_split_mode='multi')
    gen = torch.Generator().manual_seed(rank+3)
    choices = [select_handover(cfg, generator=gen).action_steps for _ in range(12)]
    Path(output+str(rank)).write_text(json.dumps(choices))
    dist.destroy_process_group()


def test_split_selection_is_synchronized_across_cpu_ranks(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp
    if not dist.is_gloo_available():
        pytest.skip('Gloo unavailable')
    mp.start_processes(_rank_select, args=('file://'+str(tmp_path/'init'), str(tmp_path/'rank')),
                       nprocs=2, join=True, start_method='spawn')
    assert (tmp_path/'rank0').read_text() == (tmp_path/'rank1').read_text()


def test_repetition_staleness_resets_gaps_and_future_rows():
    times = np.array([0,13,26,400,413,426], np.int64)*1_000_000
    sm = pd.DataFrame({'read_end_ns':times, 'sequence':np.arange(6),
        'reset_sequence':[0,0,1,1,1,1], 'read_errors':[0,0,0,0,0,1], 'dropped':[0]*6,
        'force':[np.ones(15)]*6, 'values':[np.ones(1800)]*6})
    q = build_quality(sm, times, True)
    assert q['break_before'].tolist() == [True,False,True,True,False,True]
    stale = latest_quality(q,350_000_000,0,True)
    assert stale['age_ns'] == 324_000_000 and not stale['fresh']
    _, _, valid, force_valid, _ = tactile_at(sm,times,350_000_000,0,True,quality=q)
    assert not valid and not force_valid
    stream = tactile_stream_between(sm,times,430_000_000,0,True,quality=q)
    assert len(stream['sdk_force']) == 5  # repeated rows retained
    assert stream['history_valid'].tolist() == [False,False,False,False,True]
    assert stream['force_unchanged_from_previous_read'].tolist() == [True,False,False,True,False]
    assert stream['age_ns'][-1] == 4_000_000
    assert not stream['derivative_valid'].any()
    assert latest_quality(q,10_000_000,13_000_000,True)['source_index'] == -1
    # Changing a future payload/reset must not change the current observation.
    altered = sm.copy(deep=True)
    altered.loc[5,'reset_sequence'] = 9
    altered.at[5,'force'] = np.full(15,100.)
    before = tactile_at(sm,times,413_000_000,0,True)
    after = tactile_at(altered,times,413_000_000,0,True)
    np.testing.assert_array_equal(before[1],after[1])
    # The same timestamp rules do not manufacture 76 Hz legacy observations.
    legacy_q = build_quality(sm.iloc[:3],times[:3],False)
    assert latest_quality(legacy_q,120_000_000,0,False)['fresh']
    assert not latest_quality(legacy_q,200_000_000,0,False)['fresh']


def test_real_timestamp_alignment_and_episode_boundaries(real):
    src, ds, ids = real
    for ep in [0,48,98,197,553,554,771,297]:
        sm, avail = src.tactile(ep)
        clock = src.clocks(ep)
        native = not bool(src.episodes.loc[ep,'tactile_interpolated'])
        q = build_quality(sm,avail,native)
        first = tactile_stream_between(sm,avail,int(clock[0]),None,native,quality=q)
        assert len(first['available_ns']) == 1
        follow = tactile_stream_between(sm,avail,int(clock[1]),int(clock[0]),native,quality=q)
        assert np.all(follow['available_ns'] > clock[0])
        assert np.all(follow['available_ns'] <= clock[1])
        for tick in clock[[0,len(clock)//2,-1]]:
            quality = latest_quality(q,int(tick),int(clock[0]),native)
            assert quality['fresh'] and quality['age_ns'] >= 0
        if not native:
            assert len(sm) == len(clock)
            assert not sm.interpolated.any()
            states = np.stack(src.frames(ep)['observation.state'])
            force = np.stack([tactile_at(sm,avail,int(t),int(clock[0]),False)[1].ravel() for t in clock])
            np.testing.assert_array_equal(force,states[:,19:34])
            assert not follow['taxel_valid'].any()
    # Actual loader exposes age/quality alongside the current observation.
    item = ds[1]
    b = ds.collate_fn([item])
    assert b['tactile_quality']['fresh'].all()
    assert b['tactile_quality']['age_ns'].shape == (1,)
    assert b['tactile_stream']['age_ns'].shape == b['tactile_stream']['sample_mask'].shape
    times = src.clocks(ids[0])
    gap_ticks = set((np.flatnonzero(np.diff(times)>150_000_000)+1).tolist())
    idx = next(i for i,(ep,t) in enumerate(ds._rows) if ep==ids[0] and t in gap_ticks)
    paused = ds[idx]
    assert paused['meta']['recording_gap_before_tick']
    assert not paused['tactile_stream']['history_valid'].any()
    assert not paused['tactile_stream']['derivative_valid'].any()
    assert len(paused['tactile_stream']['sdk_force']) > 1


def make_model(real, batch_size=2):
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from qwen_vla import Qwen3VLVLAModel
    torch.manual_seed(321)
    _, ds, ids = real
    other = next(i for i,(ep,_) in enumerate(ds._rows) if ep == ids[1])
    batch = ds.collate_fn([ds[65],ds[other]][:batch_size])
    cfg = AutoConfig.from_pretrained(os.environ['TAPTIC_PROCESSOR'], local_files_only=True)
    t,v = cfg.text_config,cfg.vision_config
    t.hidden_size,t.intermediate_size,t.num_hidden_layers = 64,128,1
    t.num_attention_heads,t.num_key_value_heads,t.head_dim = 2,1,32
    t.rope_scaling = {'rope_type':'default','mrope_section':[6,5,5],'mrope_interleaved':True}
    v.hidden_size,v.intermediate_size,v.depth,v.num_heads = 32,64,1,2
    v.out_hidden_size,v.deepstack_visual_indexes = 64,[0]
    cfg._attn_implementation = 'eager'
    base = Qwen3VLForConditionalGeneration(cfg).float().eval()
    model = Qwen3VLVLAModel(t,action_dim=21,action_chunk=16,tacf6_dim=360,
                            use_robot_state=True,tactile_intermediate_size=64).float().eval()
    model.initialize_vla_weights()
    model.visual = base.model.visual
    model.visual.requires_grad_(False)
    object.__setattr__(model,'_rope_index_fn',base.model.get_rope_index)
    return model,batch


def context(model,batch):
    from qwen_vla import split_slow_fast_embeds
    emb = model.prepare_inputs_embeds(batch['input_ids'],batch['pixel_values'].float(),batch['image_grid_thw'])
    slow,fast = split_slow_fast_embeds(emb,batch['input_ids'],model.image_token_id,int(batch['image_grid_thw'][0].prod()//4))
    pos,_ = model.get_rope_index(batch['input_ids'],batch['image_grid_thw'],batch['attention_mask'])
    state = model.state_embedder(batch['state_raw'].float()).unsqueeze(1)
    return slow,fast,pos[:,:,:slow.shape[1]],state


def action_loss(model,b,slow,fast,pos,state):
    full = torch.cat([slow,fast,state,model.t_embedder(b['timesteps'].float()).unsqueeze(1),
                      model.x_embedder(b['noisy_actions'].float())],1)
    nslow = slow.shape[1]
    out = model.model(inputs_embeds=full,position_ids=pos,attention_mask=b['attention_mask'],use_cache=False,
        latent_indexes=torch.arange(nslow),action_indexes=torch.arange(nslow,full.shape[1]),tactile_indexes=torch.arange(0))
    velocity = model.final_layer(out.last_hidden_state[:,-16:])
    return F.mse_loss(velocity.float(),b['target'].float())


@pytest.mark.parametrize('steps',[8,6,4])
def test_all_splits_forward_cache_times_and_backward(real,steps):
    model,b = make_model(real)
    h = Handover(steps)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    # Mimic the trainer's tactile initialization; the first step may only reach
    # the zero-initialized final norm/bias. A few tiny steps test deeper paths.
    for name,p in model.named_parameters():
        if '_tactile' in name:
            torch.nn.init.xavier_uniform_(p) if p.ndim >= 2 else torch.nn.init.zeros_(p)
    diagnostics = {}
    with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
        for iteration in range(3):
            model.zero_grad(set_to_none=True)
            slow,fast,pos,state = context(model,b)
            loss_act = action_loss(model,b,slow,fast,pos,state)
            seen = []
            hook = model.t_embedder.register_forward_pre_hook(lambda m,a: seen.append(a[0].detach().clone()))
            x,cache,nact,tau = model.forward_flow_action_partial(slow,pos,b['eps_r'].float(),
                attention_mask=b['attention_mask'],fast_embeds=fast,state_embeds=state,split_step=steps)
            hook.remove()
            assert tau == h.tau
            expected = [flow_time(i).item() for i in range(steps+1)]
            assert [float(t[0]) for t in seen] == expected
            assert all(t.dtype == torch.float32 for t in seen)
            assert nact == fast.shape[1]+1+1+16
            length = cache.get_seq_length()
            assert length == slow.shape[1]+nact
            xt,train_tau,target = tactile_training_pair(b['norm_actions'],b['eps_r'],b['time_r'],h)
            assert train_tau.dtype == torch.float32 and (train_tau>0).all() and (train_tau<=h.tau).all()
            assert torch.equal(target,b['eps_r']-b['norm_actions'])
            velocity = model.tactile_flow_train_step(cache,pos,nact,xt,train_tau,tactile_f6=b['tactile_f6s_delayed'])
            assert cache.get_seq_length() == length  # training also preserves snapshot
            loss_tac = F.mse_loss(velocity.float(),target.float())
            assert torch.isfinite(loss_act+loss_tac)
            loss_tac.backward()
            tactile_grads = {n: p.grad.detach().float().abs().sum().item() for n,p in model.named_parameters() if p.grad is not None}
            assert all(np.isfinite(v) for v in tactile_grads.values())
            assert not any('_action' in n and v>0 for n,v in tactile_grads.items())
            assert not any(n.startswith('model.') and '_tactile' not in n and v>0
                           for n,v in tactile_grads.items())
            assert not any(n.startswith('visual.') for n in tactile_grads)
            assert any('_tactile' in n and v>0 for n,v in tactile_grads.items())
            if iteration == 2:
                assert any(n.startswith('tacf6_embedder.') and v>0 for n,v in tactile_grads.items())
                assert any(n.startswith('x_embedder.') and v>0 for n,v in tactile_grads.items())
                assert any(n.startswith('t_embedder.') and v>0 for n,v in tactile_grads.items())
            model.zero_grad(set_to_none=True)
            loss_act.backward()
            assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith('final_layer.'))
            assert not any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '_tactile' in n)
            if iteration == 2:
                assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '_action' in n)
            # Both losses update their expected trainable modules; combine grads.
            for n,p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all()
            # A separate graph for the tiny optimizer step includes both losses.
            model.zero_grad(set_to_none=True)
            slow2,fast2,pos2,state2 = context(model,b)
            la = action_loss(model,b,slow2,fast2,pos2,state2)
            xt2,tt2,target2 = tactile_training_pair(b['norm_actions'],b['eps_r'],b['time_r'],h)
            vt2 = model.tactile_flow_train_step(cache,pos,nact,xt2,tt2,tactile_f6=b['tactile_f6s_delayed'])
            (la+F.mse_loss(vt2.float(),target2.float())).backward()
            optimizer.step()
            diagnostics = {'action_loss':float(loss_act.detach()),'tactile_loss':float(loss_tac.detach()),
                           'cache_length':length,'n_action':nact,'tau':tau}
        # Fresh cache after the optimizer update for repeated fast inference.
        slow,fast,pos,state = context(model,b)
        x,cache,nact,tau = model.forward_flow_action_partial(slow,pos,b['eps_r'].float(),
            attention_mask=b['attention_mask'],fast_embeds=fast,state_embeds=state,split_step=steps)
        length = cache.get_seq_length()
        seen = []
        hook = model.t_embedder.register_forward_pre_hook(lambda m,a: seen.append(a[0].detach().clone()))
        a1 = model.tactile_flow_continue(cache,pos,nact,x,tau,tactile_f6=b['tactile_f6s'],split_step=steps)
        hook.remove()
        assert [float(t[0]) for t in seen] == [flow_time(i).item() for i in range(steps,10)]
        a2 = model.tactile_flow_continue(cache,pos,nact,x,tau,tactile_f6=b['tactile_f6s'],split_step=steps)
        assert torch.equal(a1,a2) and torch.isfinite(a1).all() and a1.shape == (2,16,21)
        assert cache.get_seq_length() == length
        with pytest.raises(ValueError,match='handover'):
            model.tactile_flow_continue(cache,pos,nact,x,.2 if tau!=.2 else .4,tactile_f6=b['tactile_f6s'],split_step=steps)
    print('MULTISPLIT_RESULT',json.dumps({'action_steps':steps,**diagnostics}))


def original_method(name):
    """Load the pre-change implementation read-only for a controlled comparison."""
    import qwen_vla.modeling_vla as module
    code = subprocess.check_output(['git','show','b068f70:qwen_vla/modeling_vla.py'],text=True)
    cls = next(n for n in ast.parse(code).body if isinstance(n,ast.ClassDef) and n.name=='Qwen3VLVLAModel')
    method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace = vars(module).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'<baseline:'+name+'>','exec'),namespace)
    return namespace[name]


@torch.no_grad()
@torch.autocast('cpu',dtype=torch.bfloat16)
def test_fixed_baseline_against_original_and_weight_schema(real):
    model,b = make_model(real,batch_size=1)
    # Random, untrained test head makes integration differences observable;
    # no checkpoint is loaded, changed or written by this test.
    torch.nn.init.normal_(model.final_layer.mlp.fc2.weight,std=.02)
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    slow,fast,pos,state = context(model,b)
    kwargs = dict(attention_mask=b['attention_mask'],fast_embeds=fast,state_embeds=state,split_step=6)
    original = original_method('forward_flow_action_partial')
    old,old_cache,nold,told = original(model,slow,pos,b['eps_r'].float(),**kwargs)
    legacy,legacy_cache,nleg,tleg = model.forward_flow_action_partial(slow,pos,b['eps_r'].float(),
        flow_time_precision='legacy_bf16',**kwargs)
    revised,new_cache,nnew,tnew = model.forward_flow_action_partial(slow,pos,b['eps_r'].float(),**kwargs)
    assert torch.equal(old,legacy) and told == tleg == .39453125
    assert tnew == .4 and nnew == nleg == nold
    for a,c in zip(old_cache.layers,legacy_cache.layers):
        assert torch.equal(a.keys,c.keys) and torch.equal(a.values,c.values)
    old_continue = original_method('tactile_flow_continue')
    end_old = old_continue(model,old_cache,pos,nold,old,told,tactile_f6=b['tactile_f6s'])
    end_legacy = model.tactile_flow_continue(legacy_cache,pos,nleg,legacy,tleg,
        tactile_f6=b['tactile_f6s'],flow_time_precision='legacy_bf16')
    end_new = model.tactile_flow_continue(new_cache,pos,nnew,revised,tnew,tactile_f6=b['tactile_f6s'])
    assert torch.equal(end_old,end_legacy)
    full_kwargs = dict(attention_mask=b['attention_mask'],fast_embeds=fast,state_embeds=state)
    full_old = original_method('forward_flow_action_full')(model,slow,pos,b['eps_r'].float(),**full_kwargs)
    full_legacy = model.forward_flow_action_full(slow,pos,b['eps_r'].float(),flow_time_precision='legacy_bf16',**full_kwargs)
    full_revised = model.forward_flow_action_full(slow,pos,b['eps_r'].float(),**full_kwargs)
    assert torch.equal(full_old,full_legacy)
    assert full_revised.shape == full_old.shape and torch.isfinite(full_revised).all()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())
    src = subprocess.check_output(['git','show','b068f70:qwen_vla/modeling_vla.py'],text=True)
    now = Path('qwen_vla/modeling_vla.py').read_text()
    def constructor(s):
        cls = next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name=='Qwen3VLVLAModel')
        return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__'))
    assert constructor(src) == constructor(now)
    def delta(a,c):
        d = (a.float()-c.float()).abs()
        return {'max_abs':float(d.max()),'rms':float(d.square().mean().sqrt())}
    print('BASELINE_RESULT',json.dumps({'legacy_bitwise_equal':True,'old_tau':told,'revised_tau':tnew,
        'partial_output_difference':delta(old,revised),'final_output_difference':delta(end_old,end_new)}))


def test_training_pair_keeps_velocity_target_and_distribution():
    h = Handover(6)
    torch.manual_seed(31)
    d = torch.distributions.Beta(torch.tensor(1.5),torch.tensor(1.))
    time = d.sample((2000,))*.999+.001
    actions,noise = torch.randn(2000,16,21),torch.randn(2000,16,21)
    x,tau,target = tactile_training_pair(actions,noise,time,h)
    torch.testing.assert_close(tau,time*.4,atol=0,rtol=0)
    assert tau.min()>0 and tau.max()<=.4
    assert torch.equal(target,noise-actions)
    torch.testing.assert_close(x,(1-tau[:,None,None])*actions+tau[:,None,None]*noise)
    assert abs(float((tau/.4).mean())-.6004)<.025


@torch.no_grad()
def test_actual_trainer_validation_loop_uses_one_coherent_split(real,monkeypatch):
    import scripts.midtrain as trainer
    model,b = make_model(real)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:'cpu')
    actual_select = trainer.select_handover
    seen = []
    def choose(*args,**kwargs):
        h = actual_select(*args,**kwargs)
        seen.append(h.tau)
        return h
    monkeypatch.setattr(trainer,'select_handover',choose)
    cfg = SimpleNamespace(cascaded_split_mode='multi',cascaded_total_steps=10,cascaded_split_step=6,
        flow_time_precision='fp32',max_val_batches=1,use_robot_state=1,use_tactile_vec=1,
        use_tactile_deform=0,action_chunk=16)
    accel = SimpleNamespace(unwrap_model=lambda x:x)
    with torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
        for level in [.2,.4,.6]:
            cfg.cascaded_splits = [level]
            seen.clear()
            result = trainer.run_validation(model,[b],accel,cfg,False,False,0,0,-1)
            assert seen == [level]
            assert all(np.isfinite(v) for v in result.values())
            print('VALIDATION_RESULT',json.dumps({'tau':level,**result}))
