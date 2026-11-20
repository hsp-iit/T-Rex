"""Offline time/spatial/continuity tests; all NEW real-data selection is episode 59."""
from dataclasses import replace
import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from taptic.action_executor import (TimedActionChunk, PhysicalTrajectory, Limits,
                                    Transition, ActionExecutor)

MS = 1_000_000


def limits(**updates):
    # Synthetic/diagnostic bounds only, not hardware limits.
    base = dict(position_min=(-100.,)*3, position_max=(100.,)*3,
        joint_min=(-100.,)*12, joint_max=(100.,)*12,
        cartesian_velocity=100., angular_velocity=1000., joint_velocity=1000.,
        cartesian_acceleration=1e9, angular_acceleration=1e9, joint_acceleration=1e9,
        position_jump=100., angular_jump=4., joint_jump=100.)
    return Limits(**(base | updates))


def chunk(**updates):
    a = np.zeros((16,21));a[:,3]=1;a[:,7]=1
    a[:,0] = np.arange(16)*.01;a[:,9:] = np.arange(16)[:,None]*.01
    base = dict(actions=a,slow_ns=1000*MS,observation_ns=1000*MS,
        cache_ready_ns=1010*MS,prediction_available_ns=1020*MS,
        target_ns=tuple(1000*MS+i*100*MS for i in range(16)),
        ref_position=(0.,0.,0.),ref_rotvec_gd=(0.,0.,0.),episode=59,cache_id='59:1',
        selected_split=.4,normalization_id='train-stats-sha',config_id='config-sha',
        model_version='random-test-seed',timing_semantics='recorded_row',clock_id='test-monotonic')
    return TimedActionChunk(**(base | updates))


def executor(**updates):
    e = ActionExecutor(limits(), **updates);e.reset(59);return e


@pytest.mark.parametrize('updates',[
    {'timing_semantics':'unknown'}, {'target_ns':tuple([1000*MS]*16)},
    {'slow_ns':1000.5}, {'prediction_available_ns':999*MS},
    {'cache_ready_ns':1030*MS}, {'prediction_available_ns':2600*MS},
    {'observation_ns':999*MS}, {'model_version':''}, {'selected_split':.3},
    {'frame_id':'world'}, {'actions':np.full((16,21),np.nan)},
])
def test_reject_ambiguous_or_inconsistent_chunk(updates):
    with pytest.raises(ValueError):chunk(**updates)


def test_owned_arrays_and_reference():
    a = chunk().actions.copy();c = chunk(actions=a);a[:]=5
    assert c.actions[0,0] == 0
    with pytest.raises(ValueError):c.actions.setflags(write=True)
    with pytest.raises(ValueError):c.actions[0,0]=8
    p = PhysicalTrajectory(c)
    np.testing.assert_allclose(p.position[:,0],np.arange(16)*.01)
    np.testing.assert_allclose(p.as_actions21(),c.actions,atol=1e-14)
    # Original ref orientation Rx(pi) maps local positive y to base negative y.
    a = c.actions.copy();a[:,1]=.2
    p = PhysicalTrajectory(chunk(actions=a,ref_position=(1.,2.,3.)))
    np.testing.assert_allclose(p.position[0], [1.,1.8,3.],atol=1e-15)
    np.testing.assert_allclose(p.rotation @ p.rotation.transpose(0,2,1),np.tile(np.eye(3),(16,1,1)),atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(p.rotation),1)


def test_exact_interpolation_irregular_grid_and_slerp():
    c=chunk(); a=c.actions.copy()
    r=Rotation.from_euler('z',np.linspace(0,np.pi,16)[:,None]).as_matrix()
    a[:,3:6]=r[:,:,0];a[:,6:9]=r[:,:,1]
    times=tuple(c.observation_ns+i*100*MS+(i%2)*3*MS for i in range(16))
    p=PhysicalTrajectory(chunk(actions=a,target_ns=times))
    for i in range(1,16):
        np.testing.assert_array_equal(p.at(times[i]).position,p.position[i])
    t=(times[2]+times[3])//2
    v=p.at(t)
    np.testing.assert_allclose(v.position,(p.position[2]+p.position[3])/2,atol=1e-12)
    np.testing.assert_allclose(v.joints,.025)
    ra, rb=Rotation.from_matrix(p.rotation[[2,3]])
    rv=Rotation.from_quat(v.quaternion_xyzw)
    assert abs((ra.inv()*rv).magnitude()-(ra.inv()*rb).magnitude()/2)<1e-12
    assert abs(np.linalg.det(rv.as_matrix())-1)<1e-12


@pytest.mark.parametrize('when,reason',[(1010*MS,'prediction_unavailable'),(2501*MS,'expired')])
def test_unavailable_expired(when,reason):
    with pytest.raises(ValueError,match=reason):PhysicalTrajectory(chunk()).at(when)


def test_controller_target_before_horizon():
    c=chunk(timing_semantics='controller_target',target_ns=tuple(1100*MS+i*100*MS for i in range(16)))
    with pytest.raises(ValueError,match='before_horizon'):PhysicalTrajectory(c).at(1050*MS)


@pytest.mark.parametrize('which',[0,1])
def test_degenerate_rotation_rejected(which):
    a=chunk().actions.copy()
    if which==0:a[:,3:6]=0
    else:a[:,6:9]=a[:,3:6]
    with pytest.raises(ValueError,match='degenerate'):PhysicalTrajectory(chunk(actions=a))


def test_fast_updates_do_not_restart_and_missing_prediction():
    e=executor(); c=chunk();assert e.submit(c,1020*MS).accepted
    a=e.command(1030*MS).command
    assert a.position[0]==pytest.approx(.003)
    c2=replace(c,prediction_available_ns=1070*MS,selected_split=.6)
    assert e.submit(c2,1070*MS).accepted
    b=e.command(1080*MS).command
    assert b.position[0]==pytest.approx(.008)
    assert b.selected_split==.6
    for t in [1113,1160,1231,1300]:assert e.command(t*MS).accepted
    assert not e.command(1299*MS).accepted
    assert not e.command(1300*MS).accepted
    assert e.command(2500*MS).accepted
    assert e.command(2501*MS).status=='expired'


def test_late_out_of_order_and_unavailable_predictions():
    e=executor(); c=chunk();assert e.submit(c,1020*MS).accepted
    assert e.submit(replace(c,prediction_available_ns=1040*MS),1030*MS).status=='prediction_unavailable'
    result=e.submit(c,1040*MS)
    assert result.status=='out_of_order_prediction' and result.retained_previous
    assert e.submit(c,1030*MS).status=='nonmonotonic_event'
    assert e.command(1050*MS).accepted


def test_episode_sensor_reset_and_expiry_do_not_hold_forever():
    e=executor();assert e.submit(chunk(),1020*MS).accepted
    assert e.submit(replace(chunk(),episode=60),1030*MS).status=='episode_or_sensor_reset_requires_reset'
    assert e.command(1040*MS).status=='no_plan'
    e.reset(59,1)
    assert e.submit(chunk(sensor_segment=1),1050*MS).accepted
    assert not e.submit(replace(chunk(sensor_segment=1),prediction_available_ns=1040*MS),2600*MS).retained_previous
    e.reset(59,2);assert e.command(2700*MS).status=='no_plan'


@pytest.mark.parametrize('updates',[
    {'clock_id':'other'}, {'model_version':'new'}, {'normalization_id':'new'},
    {'ref_position':(1.,0.,0.)}, {'target_ns':tuple(1000*MS+i*101*MS for i in range(16))},
    {'cache_id':'other'},
])
def test_incompatible_update_preserves_valid_old_plan(updates):
    e=executor();c=chunk();assert e.submit(c,1020*MS).accepted
    result=e.submit(replace(c,prediction_available_ns=1070*MS,**updates),1070*MS)
    assert not result.accepted and result.retained_previous
    assert e.command(1080*MS).command.cache_id==c.cache_id


def test_same_cache_split_and_new_cache_200ms_continuity():
    e=executor(transition=Transition(80*MS));c=chunk()
    assert e.submit(c,1020*MS).accepted
    old=e.command(1060*MS).command
    a=c.actions.copy();a[:,0]+=.002;a[:,9:]+=.003
    c2=replace(c,actions=a,prediction_available_ns=1070*MS,selected_split=.2)
    assert e.submit(c2,1070*MS).accepted
    # At transition start old plan is preserved exactly at the same TIME.
    out=e.command(1070*MS).command
    np.testing.assert_allclose(out.position,PhysicalTrajectory(c).at(1070*MS).position)
    assert e.command(1110*MS).command.blended
    # Second slow anchor 200ms later, independently measured reference changes.
    next_a=c.actions.copy();next_a[:,0]+=.02
    new=replace(c,actions=next_a,slow_ns=1200*MS,observation_ns=1200*MS,
        cache_ready_ns=1210*MS,prediction_available_ns=1220*MS,cache_id='59:2',
        target_ns=tuple(1200*MS+i*100*MS for i in range(16)),selected_split=.6)
    assert e.submit(new,1220*MS).accepted
    assert e.command(1230*MS).command.cache_id=='59:2'
    # Already returned references remain immutable.
    np.testing.assert_allclose(old.position,[.006,0,0])
    assert e.command(1310*MS).accepted


@pytest.mark.parametrize('bound,changes,reason',[
    ({'position_max':(.01,100,100)}, {},'position_limit'),
    ({'joint_max':(.01,)*12}, {},'joint_limit'),
    ({'cartesian_velocity':.01}, {},'velocity_limit'),
    ({'joint_velocity':.01}, {},'velocity_limit'),
    ({'angular_velocity':.01}, {'rotation':True},'velocity_limit'),
    ({'cartesian_acceleration':.01}, {'bend':True},'acceleration_limit'),
    ({'joint_acceleration':.01}, {'hand_bend':True},'acceleration_limit'),
    ({'angular_acceleration':.01}, {'rotation_bend':True},'acceleration_limit'),
])
def test_configured_limits_reject_without_clipping(bound,changes,reason):
    c=chunk();a=c.actions.copy()
    if changes.get('bend'):a[:,0]=np.arange(16)**2*.001
    if changes.get('hand_bend'):a[:,9:]=np.arange(16)[:,None]**2*.001
    if changes.get('rotation') or changes.get('rotation_bend'):
        angle=np.arange(16)*.01
        if changes.get('rotation_bend'):angle=angle**2
        r=Rotation.from_euler('z',angle[:,None]).as_matrix();a[:,3:6]=r[:,:,0];a[:,6:9]=r[:,:,1]
    e=ActionExecutor(limits(**bound));e.reset(59)
    assert e.submit(replace(c,actions=a),1020*MS).status==reason
    np.testing.assert_array_equal(c.actions,chunk().actions)
    assert e.plan is None


def test_rejected_jump_retains_plan_and_checked_command_history():
    e=ActionExecutor(limits(position_jump=.02));e.reset(59);c=chunk()
    assert e.submit(c,1020*MS).accepted
    assert e.command(1030*MS).accepted
    a=c.actions.copy();a[:,0]+=.1
    status=e.submit(replace(c,actions=a,prediction_available_ns=1070*MS),1070*MS)
    assert status.status=='discontinuity_limit' and status.retained_previous
    assert e.command(1080*MS).accepted
    # Bounds also apply to actual emission cadence, not just validation grid.
    e.limits=limits(cartesian_velocity=.01)
    assert e.command(1090*MS).status=='velocity_limit'
    assert e.history[-1].t_ns==1080*MS


def test_transition_rejected_if_not_enough_overlap():
    e=executor(transition=Transition(100*MS));c=chunk();assert e.submit(c,1020*MS).accepted
    r=e.submit(replace(c,prediction_available_ns=2450*MS),2450*MS)
    assert r.status=='insufficient_transition_overlap' and r.retained_previous


def test_real_episode59_roundtrip_and_model_replay(tmp_path):
    root,proc=os.environ.get('TAPTIC_ROOT'),os.environ.get('TAPTIC_PROCESSOR')
    if not root or not proc:pytest.skip('set TAPTIC_ROOT and TAPTIC_PROCESSOR')
    from scripts.replay_taptic_action_executor import run
    report=run(root,proc,tmp_path,make_plots=False)
    assert report['episode']==59 and report['task']=='dosing_pipette'
    assert report['roundtrip_position_max_m']<1e-6
    assert report['roundtrip_angle_max_rad']<1e-6
    assert report['roundtrip_action21_max_abs']<1e-6
    assert report['random_prediction_shape'][1:]==[16,21]
    assert report['random_predictions']>=3
    assert set(report['selected_splits'])=={.2,.4,.6}
    assert report['cache_builds']==2 and report['cache_immutable']
    assert report['source_unchanged'] and report['future_tactile_accesses']==0
    assert report['random_plan_rejections']==0
    assert report['command_references']>0


def test_delivery_age_and_gap_validation():
    c=chunk(); e=ActionExecutor(limits(),max_delivery_delay_ns=20*MS);e.reset(59)
    assert e.submit(c,1050*MS).status=='prediction_delivery_too_late'
    assert e.submit(replace(c,prediction_available_ns=1050*MS),1060*MS).accepted
    with pytest.raises(ValueError,match='waypoint_recording_gap'):
        replace(c,target_ns=tuple(1000*MS+i*200*MS for i in range(16)))


def test_sensor_reset_is_latched_until_explicit_reset():
    e=executor();c=chunk();assert e.submit(c,1020*MS).accepted
    assert not e.submit(replace(c,sensor_segment=1),1030*MS).accepted
    assert not e.submit(replace(c,prediction_available_ns=1040*MS),1040*MS).accepted
    assert e.command(1050*MS).status=='no_plan'
    e.reset(59,1)
    assert e.submit(replace(c,sensor_segment=1,prediction_available_ns=1060*MS),1060*MS).accepted


def test_repeated_overlapping_transitions_and_angular_shortest_path():
    e=executor(transition=Transition(100*MS));c=chunk();assert e.submit(c,1020*MS).accepted
    for i,t in enumerate([1070,1120,1170,1220]):
        a=c.actions.copy();a[:,0]+=.001*i
        assert e.submit(replace(c,actions=a,prediction_available_ns=t*MS,selected_split=(.2,.4,.6)[i%3]),t*MS).accepted
        assert e.command((t+5)*MS).accepted
    assert e.command(1400*MS).accepted
    # Rotation from +170 to -170 degrees crosses the 20-degree shortest arc.
    a=c.actions.copy();angles=np.r_[np.full(3,170.),np.full(13,-170.)]
    r=Rotation.from_euler('z',angles[:,None],degrees=True).as_matrix()
    a[:,3:6]=r[:,:,0];a[:,6:9]=r[:,:,1]
    tr=PhysicalTrajectory(replace(c,actions=a))
    q=tr.at(1250*MS).quaternion_xyzw
    rel=Rotation.from_matrix(tr.rotation[2]).inv()*Rotation.from_quat(q)
    assert rel.magnitude()==pytest.approx(np.deg2rad(10))


def test_unsafe_blended_transition_is_rejected_too():
    e=ActionExecutor(limits(cartesian_velocity=.2),Transition(20*MS));e.reset(59);c=chunk()
    assert e.submit(c,1020*MS).accepted
    a=c.actions.copy();a[:,0]+=.1
    r=e.submit(replace(c,actions=a,prediction_available_ns=1070*MS),1070*MS)
    assert r.status=='velocity_limit' and r.retained_previous


def test_future_handoff_preserves_old_plan_until_first_target():
    e=executor();c=chunk(timing_semantics='controller_target')
    assert e.submit(c,1020*MS).accepted
    # New cache becomes ready at 1210ms, first supplied controller deadline is
    # 1300ms. The already accepted plan must still service 1250ms queries.
    new=replace(c,slow_ns=1200*MS,observation_ns=1200*MS,cache_ready_ns=1210*MS,
        prediction_available_ns=1220*MS,cache_id='59:2',
        target_ns=tuple(1300*MS+i*100*MS for i in range(16)))
    assert e.submit(new,1220*MS).accepted
    before=e.command(1250*MS)
    assert before.accepted and before.command.cache_id==c.cache_id
    assert e.command(1300*MS).command.cache_id==new.cache_id
