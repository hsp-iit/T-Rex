"""Touch events: the few moments where fingertip force should change the action.

One definition shared by the dataset (event-weighted tactile loss, --tactile_event_weight), the event-window
evaluation (jobs/eval/event_window.py) and the analysis tools, so "event" means the same thing everywhere.
Documented in docs/TOUCH_MODULES.md.

Inputs are the dataset's 10 fps frames: F = robot fingertip |calc_force| per finger [T, 5] (raw SDK units,
from observation.state[19:34]) and, optionally, the per-frame subtask ids. Event types (any subset):

  grasp_onset  first contact of a grasp: the strongest finger crosses `contact` after >= `quiet_s` with every
               finger below it, and reaches `reach` within 0.3 s
  resistance   while holding (in contact for >= `hold_s` before the rise), the strongest-finger force rises by
               >= `jump` within 0.2 s: the held object meets resistance (card into the wallet, cylinder into the
               hole, pipette squeeze, cap breaking free); re-triggers within `refractory_s` are suppressed.
               `jump` is per task, from the training episodes (jump_threshold)
  release      the strongest finger drops below `contact` after >= `hold_s` in contact
  label        a labelled step starts (subtask id in `label_ids`, e.g. insert / squeeze / unscrew / push);
               only for data with step labels (the re-collected unscrew has none)

A sample at frame t is an EVENT sample when an event lies in [t - post, t + pre]. Touch can only inform the
actions that come after the event (the operator's reaction takes 0.2 to 0.6 s), so the window looks back
(post, default 1 s) more than ahead (pre, default 0.2 s, for sensor timing).
"""
import numpy as np

EVENT_TYPES = ('grasp_onset', 'resistance', 'release', 'label')
DEFAULT_TYPES = ('grasp_onset', 'resistance')
DEFAULT_LABELS = ('insert', 'squeeze', 'unscrew', 'push')
DEFAULT_WINDOW_MS = (200., 1000.)          # (pre, post)
FPS = 10


def finger_force(state):
    """[T, 40] observation.state -> [T, 5] fingertip |calc_force| (raw SDK units)."""
    s = np.asarray(state, np.float64)
    return np.linalg.norm(s[:, 19:34].reshape(-1, 5, 3), axis=-1)


def rise(F, frames=2):
    """Strongest-finger force increase over the last `frames` frames (0 for the first frames)."""
    fm = np.asarray(F, np.float64).max(1)
    r = np.zeros_like(fm)
    r[frames:] = fm[frames:] - fm[:-frames]
    return r


def contact_run(F, contact=2.0):
    """Number of consecutive contact frames (strongest finger > contact) ending at each frame."""
    c = np.asarray(F, np.float64).max(1) > contact
    run = np.zeros(len(c), np.int64)
    for t in range(len(c)):
        run[t] = run[t - 1] + 1 if (c[t] and t) else int(c[t])
    return run


def jump_threshold(force_list, quantile=0.97, contact=2.0, frames=2, floor=5.0):
    """Per-task threshold for `resistance`: the `quantile` of the positive strongest-finger rises during
    contact over the given episodes (pass TRAINING episodes only), never below `floor` units."""
    vals = []
    for F in force_list:
        c = np.asarray(F, np.float64).max(1) > contact
        r = rise(F, frames)
        vals.append(r[(r > 0) & c])
    v = np.concatenate(vals) if vals else np.zeros(0)
    return float(max(np.quantile(v, quantile), floor)) if len(v) else float('inf')


def episode_events(F, *, types=DEFAULT_TYPES, subtask=None, label_ids=(), jump=float('inf'), contact=2.0,
                   reach=5.0, quiet_s=0.5, hold_s=0.5, refractory_s=1.0, fps=FPS):
    """Boolean [T]: True at the frames where an event of `types` happens."""
    unknown = set(types) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f'unknown touch event types {sorted(unknown)}; choose from {EVENT_TYPES}')
    F = np.asarray(F, np.float64)
    T = len(F)
    fm = F.max(1) if T else np.zeros(0)
    c = fm > contact
    run = contact_run(F, contact)
    ev = np.zeros(T, bool)
    q, h, rr = int(round(quiet_s * fps)), int(round(hold_s * fps)), int(round(refractory_s * fps))
    if 'grasp_onset' in types:
        for t in np.flatnonzero(c[1:] & ~c[:-1]) + 1:
            if t >= q and not c[t - q:t].any() and fm[t:t + 3].max() >= reach:
                ev[t] = True
    if 'release' in types:
        for t in np.flatnonzero(~c[1:] & c[:-1]) + 1:
            if run[t - 1] >= h:
                ev[t] = True
    if 'resistance' in types and np.isfinite(jump):
        r = rise(F)
        last = -10 ** 9
        for t in np.flatnonzero((r >= jump) & c):
            if t >= 2 and run[t - 2] >= h and t - last >= rr:
                ev[t] = True
                last = t
    if 'label' in types and subtask is not None and len(label_ids):
        s = np.asarray(subtask)
        ev |= np.r_[False, s[1:] != s[:-1]] & np.isin(s, list(label_ids))
    return ev


def window_flags(events, pre_s=DEFAULT_WINDOW_MS[0] / 1000., post_s=DEFAULT_WINDOW_MS[1] / 1000., fps=FPS):
    """flag[t] = an event lies in frames [t - post, t + pre]."""
    e = np.asarray(events, bool)
    T = len(e)
    pre, post = int(round(pre_s * fps)), int(round(post_s * fps))
    cs = np.r_[0, np.cumsum(e)]
    t = np.arange(T)
    return (cs[np.minimum(T, t + pre + 1)] - cs[np.maximum(0, t - post)]) > 0
