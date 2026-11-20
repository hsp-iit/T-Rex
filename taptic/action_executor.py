"""Offline, serialized command-reference executor. No robot/controller API.

All times are integer nanoseconds in ONE monotonic clock. ``recorded_row``
means label-row association, NOT a measured actuator application deadline.
Callers must explicitly choose a timing contract and provide target times.
Limits are application configuration, not certified Panda/XHand limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .actions import rotvec_gd_to_matrix, rot6d_to_matrix


def ns(x, name='timestamp'):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 0:
        raise ValueError(f'{name}: expected nonnegative integer nanoseconds')
    return int(x)


def owned(x, shape, name):
    a = np.asarray(x, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected finite {shape}')
    # Immutable bytes backing prevents even setflags(write=True) alias mutation.
    return np.frombuffer(a.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True)
class TimedActionChunk:
    actions: np.ndarray                         # denormalized [16,21]
    slow_ns: int
    observation_ns: int
    cache_ready_ns: int
    prediction_available_ns: int
    target_ns: tuple[int, ...]
    ref_position: tuple[float, ...]
    ref_rotvec_gd: tuple[float, ...]
    episode: int
    cache_id: str
    selected_split: float
    normalization_id: str
    config_id: str
    model_version: str
    timing_semantics: str                       # recorded_row / controller_target
    clock_id: str                               # e.g. recording CLOCK_MONOTONIC
    frame_id: str = 'panda_urdf_base:flange_link8'
    sensor_segment: int = 0
    max_waypoint_gap_ns: int = 150_000_000       # existing TAPTIC adapter cutoff

    def __post_init__(self):
        object.__setattr__(self, 'actions', owned(self.actions, (16, 21), 'actions'))
        for key in ('slow_ns', 'observation_ns', 'cache_ready_ns', 'prediction_available_ns',
                    'episode', 'sensor_segment'):
            object.__setattr__(self, key, ns(getattr(self, key), key))
        times = tuple(ns(t, 'target_ns') for t in self.target_ns)
        if len(times) != 16 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('target_ns must contain 16 strictly increasing timestamps')
        object.__setattr__(self, 'target_ns', times)
        gap = ns(self.max_waypoint_gap_ns, 'max_waypoint_gap_ns')
        if gap == 0 or any(b-a > gap for a, b in zip(times, times[1:])):
            raise ValueError('waypoint_recording_gap')
        if not self.observation_ns <= self.slow_ns <= self.cache_ready_ns <= self.prediction_available_ns:
            raise ValueError('require observation <= slow <= ready <= prediction availability')
        if self.prediction_available_ns > times[-1] or times[0] < self.observation_ns:
            raise ValueError('expired or pre-observation target horizon')
        if self.timing_semantics not in ('recorded_row', 'controller_target'):
            raise ValueError('explicit verified timing semantics required')
        if self.timing_semantics == 'recorded_row' and times[0] != self.observation_ns:
            raise ValueError('recorded_row action[0] must match observation row time')
        for key in ('ref_position', 'ref_rotvec_gd'):
            object.__setattr__(self, key, tuple(owned(getattr(self, key), (3,), key)))
        for key in ('cache_id', 'normalization_id', 'config_id', 'model_version', 'clock_id', 'frame_id'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f'{key}: nonempty identity required')
        if self.frame_id != 'panda_urdf_base:flange_link8':
            raise ValueError('unsupported/unverified physical frame')
        if self.selected_split not in (.2, .4, .6):
            raise ValueError('selected_split must be .2, .4 or .6')

    @classmethod
    def from_prediction(cls, result, anchor, *, prediction_available_ns, target_ns,
                        normalization_id, config_id, model_version, timing_semantics,
                        clock_id, sensor_segment):
        """Bridge from ApproachAInference, B=1 only; no invented ready/target times.

        Availability is the caller's actual completion time, not inference-start F.
        SlowAnchor.ready_ns must likewise reflect actual cache completion.
        """
        if result.get('status') != 'ok' or result.get('actions') is None:
            raise ValueError('no available prediction')
        prediction_available_ns = ns(prediction_available_ns, 'prediction_available_ns')
        if (result['slow_anchor_ns'] != anchor.slow_ns or
                result.get('tick_ns', -1) < anchor.ready_ns or
                result['tick_ns'] >= anchor.expires_ns or
                result['tick_ns'] > prediction_available_ns):
            raise ValueError('prediction/anchor timing mismatch')
        ref = result.get('reference', {})
        if (ref.get('observation_ns') != anchor.observation_ns or
                not np.array_equal(ref.get('position'), anchor.ref_position) or
                not np.array_equal(ref.get('rotvec_gd'), anchor.ref_rotvec_gd)):
            raise ValueError('prediction/reference mismatch')
        a = np.asarray(result['actions'])
        if a.shape != (1, 16, 21):
            raise ValueError('executor bridge requires B=1')
        return cls(a[0], anchor.slow_ns, anchor.observation_ns, anchor.ready_ns,
                   prediction_available_ns, tuple(target_ns), anchor.ref_position,
                   anchor.ref_rotvec_gd, anchor.episode, result['cache_identity'],
                   result['gate']['selected_split'], normalization_id, config_id,
                   model_version, timing_semantics, clock_id, sensor_segment=sensor_segment)


@dataclass(frozen=True)
class CommandReference:
    t_ns: int
    position: np.ndarray
    quaternion_xyzw: np.ndarray
    joints: np.ndarray
    episode: int
    cache_id: str
    selected_split: float
    blended: bool = False

    def __post_init__(self):
        object.__setattr__(self, 't_ns', ns(self.t_ns))
        for key, shape in [('position', (3,)), ('quaternion_xyzw', (4,)), ('joints', (12,))]:
            object.__setattr__(self, key, owned(getattr(self, key), shape, key))
        if abs(np.linalg.norm(self.quaternion_xyzw)-1) > 1e-7:
            raise ValueError('unit quaternion required')


def blend(a, b, alpha):
    """Same-time interpolation in base coordinates / shortest-arc SO(3)."""
    if a.t_ns != b.t_ns or a.episode != b.episode or not 0 <= alpha <= 1:
        raise ValueError('incompatible blend')
    q = Slerp([0., 1.], Rotation.from_quat([a.quaternion_xyzw, b.quaternion_xyzw]))([alpha]).as_quat()[0]
    return CommandReference(b.t_ns, (1-alpha)*a.position+alpha*b.position, q,
                            (1-alpha)*a.joints+alpha*b.joints, b.episode, b.cache_id,
                            b.selected_split, 0 < alpha < 1)


class Plan(Protocol):
    chunk: TimedActionChunk
    start_ns: int
    end_ns: int

    def at(self, t_ns: int) -> CommandReference: ...


class PhysicalTrajectory:
    def __init__(self, chunk):
        self.chunk = chunk
        self.start_ns, self.end_ns = chunk.target_ns[0], chunk.target_ns[-1]
        a = chunk.actions
        a1, a2 = a[:, 3:6], a[:, 6:9]
        n1 = np.linalg.norm(a1, axis=1)
        if np.any(n1 < 1e-8):
            raise ValueError('degenerate rotation first column')
        proj = a2 - a1 * ((a1*a2).sum(1) / n1**2)[:, None]
        if np.any(np.linalg.norm(proj, axis=1) < 1e-8):
            raise ValueError('degenerate rotation second column')
        r0 = rotvec_gd_to_matrix(chunk.ref_rotvec_gd)
        self.position = owned(np.asarray(chunk.ref_position) + a[:, :3] @ r0.T, (16, 3), 'base position')
        self.rotation = owned(r0 @ rot6d_to_matrix(a[:, 3:9]), (16, 3, 3), 'rotation')
        self.joints = owned(a[:, 9:], (16, 12), 'joints')
        self.quaternion = owned(Rotation.from_matrix(self.rotation).as_quat(), (16, 4), 'quaternion')

    def as_actions21(self):
        """Inverse after Gram-Schmidt projection; exact for valid rotation columns."""
        r0 = rotvec_gd_to_matrix(self.chunk.ref_rotvec_gd)
        dr = r0.T @ self.rotation
        return np.concatenate([(self.position-self.chunk.ref_position) @ r0,
                               dr[:, :, 0], dr[:, :, 1], self.joints], axis=1)

    def at(self, t_ns):
        t_ns = ns(t_ns)
        if t_ns < self.chunk.prediction_available_ns:
            raise ValueError('prediction_unavailable')
        if t_ns < self.start_ns:
            raise ValueError('before_horizon')
        if t_ns > self.end_ns:
            raise ValueError('expired')
        times = self.chunk.target_ns
        i = int(np.searchsorted(times, t_ns, side='right'))-1
        a = CommandReference(t_ns, self.position[i], self.quaternion[i], self.joints[i],
                             self.chunk.episode, self.chunk.cache_id, self.chunk.selected_split)
        if t_ns == times[i]:
            return a
        j = i+1
        b = CommandReference(t_ns, self.position[j], self.quaternion[j], self.joints[j],
                             self.chunk.episode, self.chunk.cache_id, self.chunk.selected_split)
        return blend(a, b, (t_ns-times[i])/(times[j]-times[i]))


@dataclass(frozen=True)
class Limits:
    """Required finite site/controller limits. No invented hardware defaults.

    Angular limits are radians, Cartesian limits metres, derivatives per second.
    Acceleration is checked on sampled command history, not continuous/jerk.
    """
    position_min: tuple[float, ...]
    position_max: tuple[float, ...]
    joint_min: tuple[float, ...]
    joint_max: tuple[float, ...]
    cartesian_velocity: float
    angular_velocity: float
    joint_velocity: float
    cartesian_acceleration: float
    angular_acceleration: float
    joint_acceleration: float
    position_jump: float
    angular_jump: float
    joint_jump: float

    def __post_init__(self):
        for lo, hi, n in [('position_min', 'position_max', 3), ('joint_min', 'joint_max', 12)]:
            l, h = owned(getattr(self, lo), (n,), lo), owned(getattr(self, hi), (n,), hi)
            if np.any(h <= l):
                raise ValueError('invalid bounds')
            object.__setattr__(self, lo, tuple(l)); object.__setattr__(self, hi, tuple(h))
        for key in ('cartesian_velocity', 'angular_velocity', 'joint_velocity', 'cartesian_acceleration',
                    'angular_acceleration', 'joint_acceleration', 'position_jump', 'angular_jump', 'joint_jump'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key}: finite positive bound required')


@dataclass(frozen=True)
class Transition:
    duration_ns: int = 0
    validation_step_ns: int = 5_000_000
    max_validation_points: int = 10000

    def __post_init__(self):
        ns(self.duration_ns); ns(self.validation_step_ns)
        if self.validation_step_ns == 0 or self.max_validation_points < 2:
            raise ValueError('invalid validation grid')


class TransitionPlan:
    def __init__(self, old, new, start, duration):
        self.old, self.new, self.chunk = old, new, new.chunk
        self.start_ns, self.end_ns = old.start_ns, new.end_ns
        self.switch_ns = start
        self.blend_end_ns = start+duration
        if self.blend_end_ns > min(old.end_ns, new.end_ns):
            raise ValueError('insufficient_transition_overlap')

    def at(self, t_ns):
        if t_ns < self.start_ns or t_ns > self.end_ns:
            raise ValueError('outside_transition_horizon')
        if t_ns < self.switch_ns:
            return self.old.at(t_ns)
        if t_ns >= self.blend_end_ns:
            return self.new.at(t_ns)
        return blend(self.old.at(t_ns), self.new.at(t_ns),
                     (t_ns-self.switch_ns)/(self.blend_end_ns-self.switch_ns))

    def compact(self, now):
        # Bound dependency chains after completed transitions (no executed history).
        if now >= self.blend_end_ns:
            return self.new
        if isinstance(self.old, TransitionPlan):
            self.old = self.old.compact(now)
        return self


@dataclass(frozen=True)
class ExecutionResult:
    accepted: bool
    status: str
    command: CommandReference | None = None
    retained_previous: bool = False


def rates(a, b):
    dt = (b.t_ns-a.t_ns)/1e9
    if dt <= 0:
        raise ValueError('nonmonotonic_command')
    w = (Rotation.from_quat(b.quaternion_xyzw)*Rotation.from_quat(a.quaternion_xyzw).inv()).as_rotvec()/dt
    return (b.position-a.position)/dt, w, (b.joints-a.joints)/dt


class ActionExecutor:
    """Serialized receding-horizon references, never hardware actuation.

    ``submit`` is called at receipt time, ``command`` at intended application
    time. Supplied clock/frame identities are checked, not clock-synchronized.
    reset() is mandatory after episode or sensor reset; old plan is invalidated.
    """
    def __init__(self, limits: Limits, transition=Transition(), *, max_delivery_delay_ns=None):
        self.limits, self.transition = limits, transition
        self.max_delivery_delay_ns = (None if max_delivery_delay_ns is None
                                      else ns(max_delivery_delay_ns, 'max_delivery_delay_ns'))
        self.reset(None)

    def reset(self, episode, sensor_segment=0):
        self.episode = None if episode is None else ns(episode, 'episode')
        self.sensor_segment = ns(sensor_segment, 'sensor_segment')
        self.plan = None
        self.last_event_ns = -1
        self.last_prediction_ns = -1
        self.history = []
        self.identity = None
        self.reset_required = False

    def _position(self, c):
        l = self.limits
        if np.any(c.position < l.position_min) or np.any(c.position > l.position_max):
            raise ValueError('position_limit')
        if np.any(c.joints < l.joint_min) or np.any(c.joints > l.joint_max):
            raise ValueError('joint_limit')

    def _jump(self, a, b):
        l = self.limits
        angle = (Rotation.from_quat(a.quaternion_xyzw).inv()*Rotation.from_quat(b.quaternion_xyzw)).magnitude()
        if (np.linalg.norm(b.position-a.position) > l.position_jump or angle > l.angular_jump or
                np.max(np.abs(b.joints-a.joints)) > l.joint_jump):
            raise ValueError('discontinuity_limit')

    def _motion(self, history, c):
        self._position(c)
        if not history:
            return
        self._jump(history[-1], c)
        v = rates(history[-1], c)
        l = self.limits
        if (np.linalg.norm(v[0]) > l.cartesian_velocity or np.linalg.norm(v[1]) > l.angular_velocity or
                np.max(np.abs(v[2])) > l.joint_velocity):
            raise ValueError('velocity_limit')
        if len(history) >= 2:
            prev = rates(history[-2], history[-1])
            # Velocities live at interval midpoints, on potentially irregular grids.
            dt = (c.t_ns-history[-2].t_ns)/2e9
            acc = [(b-a)/dt for a, b in zip(prev, v)]
            if (np.linalg.norm(acc[0]) > l.cartesian_acceleration or
                    np.linalg.norm(acc[1]) > l.angular_acceleration or
                    np.max(np.abs(acc[2])) > l.joint_acceleration):
                raise ValueError('acceleration_limit')

    def _event(self, t):
        t = ns(t)
        if t < self.last_event_ns:
            raise ValueError('nonmonotonic_event')
        self.last_event_ns = t
        if isinstance(self.plan, TransitionPlan):
            self.plan = self.plan.compact(t)
        return t

    def _retained(self, t):
        return bool(self.plan and max(self.plan.start_ns, self.plan.chunk.prediction_available_ns) <= t <= self.plan.end_ns)

    def submit(self, chunk: TimedActionChunk, received_ns: int):
        try:
            t = self._event(received_ns)
        except ValueError as e:
            return ExecutionResult(False, str(e))
        try:
            if self.reset_required:
                raise ValueError('episode_or_sensor_reset_requires_reset')
            if chunk.episode != self.episode or chunk.sensor_segment != self.sensor_segment:
                # Conservative reset boundary: even previous plan is unusable.
                self.plan = None
                self.reset_required = True
                raise ValueError('episode_or_sensor_reset_requires_reset')
            if chunk.prediction_available_ns > t:
                raise ValueError('prediction_unavailable')
            if chunk.prediction_available_ns <= self.last_prediction_ns:
                raise ValueError('out_of_order_prediction')
            if self.max_delivery_delay_ns is not None and t-chunk.prediction_available_ns > self.max_delivery_delay_ns:
                raise ValueError('prediction_delivery_too_late')
            identity = (chunk.normalization_id, chunk.config_id, chunk.model_version,
                        chunk.clock_id, chunk.frame_id, chunk.timing_semantics)
            if self.identity is not None and self.identity != identity:
                raise ValueError('incompatible_plan_identity_requires_reset')
            if self.plan:
                prev = self.plan.chunk
                if chunk.slow_ns < prev.slow_ns:
                    raise ValueError('older_slow_cache')
                if chunk.cache_id == prev.cache_id:
                    fields = ('slow_ns', 'observation_ns', 'cache_ready_ns', 'target_ns',
                              'ref_position', 'ref_rotvec_gd')
                    if any(getattr(chunk, k) != getattr(prev, k) for k in fields):
                        raise ValueError('same_cache_reference_changed')
                elif chunk.slow_ns <= prev.slow_ns:
                    raise ValueError('new_cache_requires_new_slow_anchor')
            new = PhysicalTrajectory(chunk)
            # No retroactive replacement of already emitted command references.
            start = max(t, new.start_ns, self.history[-1].t_ns+1 if self.history else t)
            if start > new.end_ns:
                raise ValueError('expired')
            candidate = new
            if self.plan and self._retained(t) and start > t and not self._retained(start):
                raise ValueError('insufficient_transition_overlap')
            if self.plan and self._retained(start):
                if not self.transition.duration_ns:
                    self._jump(self.plan.at(start), new.at(start))
                if self.transition.duration_ns or start > t:
                    candidate = TransitionPlan(self.plan, new, start, self.transition.duration_ns)
            if (new.end_ns-start)//self.transition.validation_step_ns+1 > self.transition.max_validation_points:
                raise ValueError('validation_horizon_too_large')
            grid = set(range(start, new.end_ns+1, self.transition.validation_step_ns))
            grid.update(t for t in chunk.target_ns if t >= start)
            grid.add(new.end_ns)
            if isinstance(candidate, TransitionPlan):
                grid.add(candidate.blend_end_ns)
                # Include ALL nested trajectory knots; finite grid is not a proof
                # of continuous derivative/acceleration bounds between samples.
                def knots(plan):
                    if isinstance(plan, TransitionPlan):
                        return set(plan.chunk.target_ns) | {plan.switch_ns, plan.blend_end_ns} | knots(plan.old)
                    return set(plan.chunk.target_ns)
                grid.update(x for x in knots(candidate) if start <= x <= new.end_ns)
            if len(grid) > self.transition.max_validation_points:
                raise ValueError('validation_horizon_too_large')
            h = list(self.history)
            for tick in sorted(grid):
                c = candidate.at(tick)
                self._motion(h, c)
                h = (h+[c])[-2:]
            self.plan, self.identity = candidate, identity
            self.last_prediction_ns = chunk.prediction_available_ns
            return ExecutionResult(True, 'plan_accepted')
        except ValueError as e:
            return ExecutionResult(False, str(e), retained_previous=self._retained(t))

    def command(self, t_exec):
        try:
            t = self._event(t_exec)
            if self.history and t <= self.history[-1].t_ns:
                raise ValueError('nonmonotonic_command')
            if not self.plan:
                raise ValueError('no_plan')
            c = self.plan.at(t)
            self._motion(self.history, c)
            self.history = (self.history+[c])[-2:]
            return ExecutionResult(True, 'command_reference', c)
        except ValueError as e:
            return ExecutionResult(False, str(e))
