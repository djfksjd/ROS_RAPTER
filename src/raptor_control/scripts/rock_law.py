"""Open-loop rocking + stride position targets, ROS-free (same law as sim/rock_probe.py with lift = 0).

a: seconds since rocking started (after settle). Hip rolls: left = -r, right = +r, r = A*min(1, a/ramp)*sin(2*pi*f*a).
Stride: inside each foot's unloaded window (width `window` cycles, centred at `centers`) the hip pitch swings
forward linearly and returns during stance; the ankle cancels it. Optional ankle pitch feedback on base
pitch and pitch rate, clipped. Position targets only; the caller owns STOP.
"""
import math
from collections import deque

CENTERS = (('left', .28), ('right', .78))
MIRRORED = (('left', .78), ('right', .28))  # left/right roles swapped (with r -> -r): exact y-mirror of the gait


def crouch_pose(joints, hip, knee):
    pose = dict.fromkeys(joints, 0.)
    for side in ('left', 'right'):
        pose |= {f'{side}_hip_pitch_joint': hip, f'{side}_knee_pitch_joint': knee,
                 f'{side}_ankle_pitch_joint': -hip-knee}
    return pose


def targets(pose, a, amplitude, frequency, ramp, stride=0., window=.35, pitch=0., pitch_rate=0., feedback=None,
            mirror=False, abduction=0., cycle=None, ramp_shape='linear', stride_ramp=1.):
    target = dict(pose)
    if a < 0:
        return target
    # cycle: external gait clock in cycles (TouchdownPLL); default is the open-loop time clock a*frequency
    c = a*frequency if cycle is None else cycle
    u = min(1., a/ramp)
    gain = .5*(1-math.cos(math.pi*u)) if ramp_shape == 'cosine' else u  # raised cosine: zero slope at both ends
    r = amplitude*gain*math.sin(2*math.pi*c)*(-1 if mirror else 1)
    # abduction: +delta on both hip rolls moves both feet outward (right axis is -x); ramped in with the gait
    ab = abduction*min(1., a/ramp)
    target |= {'left_hip_roll_joint': -r+ab, 'right_hip_roll_joint': r+ab}
    cycle = c % 1
    for side, center in (MIRRORED if mirror else CENTERS):
        offset = (cycle-center+.5) % 1-.5
        u = (offset+window/2)/window
        swing = -stride*(u-.5) if 0 <= u <= 1 else -stride*(.5-((offset-window/2) % 1)/(1-window))
        swing *= min(1., max(0., (a-ramp)/stride_ramp))
        target[f'{side}_hip_pitch_joint'] = pose[f'{side}_hip_pitch_joint']+swing
        target[f'{side}_ankle_pitch_joint'] = pose[f'{side}_ankle_pitch_joint']-swing
    if feedback:
        kp, kd, limit = feedback
        correction = max(-limit, min(limit, kp*pitch+kd*pitch_rate))
        for side in ('left', 'right'):
            target[f'{side}_ankle_pitch_joint'] += correction
    return target


def tail_targets(r, roll=0., roll_rate=0., pitch=0., pitch_rate=0., gains=(0., 0., 0., 0.), yaw_limit=.6,
                 pitch_limit=.4):
    """Tail balance, position targets only (tail yaw axis +z, tail pitch axis +y; +pitch raises the tail).

    yaw = ky*r + kr*roll (+yaw swings the tail to -y); pitch = kp*pitch + kd*pitch_rate (body pitch, + = nose down).
    `roll_rate` is accepted for a later damping term and unused now. Signs are chosen by experiment (evidence 70).
    """
    ky, kr, kp, kd = gains
    return {'tail_yaw_joint': max(-yaw_limit, min(yaw_limit, ky*r+kr*roll)),
            'tail_pitch_joint': max(-pitch_limit, min(pitch_limit, kp*pitch+kd*pitch_rate))}


class GaitPhase:
    """Gait phase from left-foot contact only: phase 0 at a left touchdown, period and duty by EMA.

    update(t, left_contact) takes debounced booleans (MuJoCo: force hysteresis; Gazebo: contact messages) and returns
    phase in [0, 1) or None until two touchdowns were seen, or when no touchdown came for 1.5 periods.
    """

    def __init__(self, alpha=.3, min_period=.2, max_period=2.):
        self.alpha, self.min_period, self.max_period = alpha, min_period, max_period
        self.previous = self.last_touchdown = self.period = self.duty = None

    def update(self, t, left):
        if self.previous is not None and left != self.previous:
            if left:  # touchdown
                if self.last_touchdown is not None and self.min_period <= t-self.last_touchdown <= self.max_period:
                    p = t-self.last_touchdown
                    self.period = p if self.period is None else (1-self.alpha)*self.period+self.alpha*p
                self.last_touchdown = t
            elif self.last_touchdown is not None and self.period:  # liftoff: stance fraction of the period
                d = (t-self.last_touchdown)/self.period
                if 0 < d < 1:
                    self.duty = d if self.duty is None else (1-self.alpha)*self.duty+self.alpha*d
        self.previous = left
        if self.period is None or self.last_touchdown is None:
            return None
        u = (t-self.last_touchdown)/self.period
        return u % 1 if u < 1.5 else None


def tail_sync_targets(phase, yaw_rate, amp, phi0, k_fb, limit=.6):
    """Gait-synchronised tail yaw (pasted design note, one root segment): theta = A*sin(2*pi*(phase-phi0)) + K*yaw_rate,
    theta + = tail to the LEFT. Our tail_yaw_joint + moves the tail to -y, so the joint target is -theta.
    Tail pitch is held at 0 (stiff). No feedforward while the phase is unknown."""
    theta = (0. if phase is None else amp*math.sin(2*math.pi*(phase-phi0)))+k_fb*yaw_rate
    return {'tail_yaw_joint': max(-limit, min(limit, -theta)), 'tail_pitch_joint': 0.}


class TailSync:
    """tail_sync_targets with a first-order low-pass on the yaw rate and a slew limit on the tail yaw command.

    Gazebo's 50 Hz IMU yaw rate jumps up to 0.3 rad/s between 20 ms samples at touchdowns (evidence 71); unfiltered,
    K_fb turned that into tail commands at the 2 rad/s limit. Same filter in MuJoCo keeps the two engines comparable.
    """

    def __init__(self, amp, phi0, k_fb, tau=.02, max_rate=1.):
        self.amp, self.phi0, self.k_fb, self.tau, self.max_rate = amp, phi0, k_fb, tau, max_rate
        self.t = self.yaw_rate = self.command = None

    def update(self, t, phase, yaw_rate):
        if self.t is None:
            self.yaw_rate, dt = yaw_rate, 0.
        else:
            dt = max(0., t-self.t)
            self.yaw_rate += (yaw_rate-self.yaw_rate)*(dt/(self.tau+dt) if dt else 0.)
        self.t = t
        target = tail_sync_targets(phase, self.yaw_rate, self.amp, self.phi0, self.k_fb)
        yaw = target['tail_yaw_joint']
        if self.command is not None:
            step = self.max_rate*dt
            yaw = min(self.command+step, max(self.command-step, yaw))
        self.command = yaw
        return target | {'tail_yaw_joint': yaw}


def step_roll(a, amp=.08, hold=.5, rate=1.6, second=3.):
    """Experiment F profile (evidence 74): hip-roll 'r' ramps at `rate` to +amp at a = 0, holds `hold` s, returns
    to 0; the mirrored event (-amp) starts at a = `second`. By 3 s MuJoCo has stopped rocking; Gazebo has not.
    Rate-limited to 80 % of the 2 rad/s hip-roll limit. Returns r (left hip roll = -r, right = +r)."""
    rise = amp/rate
    hold = max(hold, rise)  # never a step: the down-ramp starts after the up-ramp ends

    def pulse(t):
        if t < 0:
            return 0.
        if t < rise:
            return rate*t
        if t < hold:
            return amp
        return max(0., amp-rate*(t-hold))
    return pulse(a)-pulse(a-second)


class WindowMax:
    """Max of the samples in the last `window` s: bridges Gazebo's zero-force contact frames (evidence 72/74)."""

    def __init__(self, window=.03):
        self.window, self.samples = window, deque()

    def update(self, t, value):
        self.samples.append((t, value))
        while self.samples and self.samples[0][0] < t-self.window:
            self.samples.popleft()
        return max(v for _, v in self.samples)


class ContactEvents:
    """Debounced foot contact from normal force: on above threshold+hysteresis, off below threshold-hysteresis,
    a change is accepted only after it persisted `hold` s. update() returns 'touchdown', 'liftoff' or None.
    Stage 2 (a) defaults: 34 N (20 % of the 170 N weight), +-10 N, 20 ms."""

    def __init__(self, threshold=34., hysteresis=10., hold=.02):
        self.on, self.off, self.hold = threshold+hysteresis, threshold-hysteresis, hold
        self.loaded, self.candidate_since = None, None

    def update(self, t, force):
        raw = force >= self.on if self.loaded is not True else force >= self.off
        if self.loaded is None:
            self.loaded = raw
            return None
        if raw == self.loaded:
            self.candidate_since = None
            return None
        if self.candidate_since is None:
            self.candidate_since = t
        if t-self.candidate_since+1e-9 >= self.hold:
            self.loaded, self.candidate_since = raw, None
            return 'touchdown' if raw else 'liftoff'
        return None


class FrequencyProfile:
    """Replay a recorded command-frequency history [(t, f), ...] (piecewise constant) as a gait clock (experiment B)."""

    def __init__(self, f0, points):
        self.f0, self.points, self.c, self.t = f0, sorted(points), 0., None
        self.events = []  # same interface as TouchdownPLL (no feedback events)

    def frequency(self, t):
        f = self.f0
        for tp, fp in self.points:
            if tp > t:
                break
            f = fp
        return f

    def update(self, t, forces=None):
        if self.t is not None:
            self.c += self.frequency(self.t)*max(0., t-self.t)
        self.t = t
        return self.c


class TouchdownPLL:
    """Stage 2 (1): touchdown-anchored gait clock (Fable review, evidence 74). The cycle variable c is integrated at
    f; at each debounced touchdown the phase error e = wrap(nominal - c mod 1) (cycles, [-0.5, 0.5)) sets
    f = clip(f0 + k*e, f0 +- clamp). No phase jumps (a 0.15-cycle jump would need > 2 rad/s of hip roll).
    Nominal touchdown phases 0.595 / 0.095 are the kv30 open-loop means at 2.5 Hz. `window` > 0 applies WindowMax
    to the forces first (Gazebo contact messages). Returns c from update(); `events` logs (t, side, e, f).
    Touchdowns of the same foot sooner than `min_gap` cycles after its last accepted one are bounces and ignored;
    no frequency change before `active_after` (caller's clock: ramp end), where contacts are still ambiguous."""

    def __init__(self, f0, k=2.5, clamp=.4, nominal=(.595, .095), window=0., min_gap=.6, active_after=-math.inf,
                 accept='first', cluster=.25, learn=0):
        self.f0, self.k, self.clamp = f0, k, clamp
        self.nominal = {'left': nominal[0], 'right': nominal[1]}
        self.detect = {s: ContactEvents() for s in self.nominal}
        self.smooth = {s: WindowMax(window) for s in self.nominal} if window > 0 else None
        self.f, self.c, self.t, self.events = f0, 0., None, []
        self.min_gap, self.active_after, self.last_touchdown = min_gap/f0, active_after, {}
        # v3 (Fable review 3): accept='last' keeps the LAST touchdown of a burst within `cluster` cycles (committed
        # once the burst is over); learn=N learns each foot's nominal phase as the circular mean of its first N
        # accepted touchdowns after active_after (no retuning until learned)
        self.accept, self.cluster, self.learn = accept, cluster/f0, learn
        self.pending, self.samples = {}, {'left': [], 'right': []}
        if learn:
            self.nominal = {}

    def update(self, t, forces):
        if self.t is not None:
            self.c += self.f*max(0., t-self.t)
        self.t = t
        for side, force in forces.items():
            if self.smooth:
                force = self.smooth[side].update(t, force)
            touchdown = self.detect[side].update(t, force) == 'touchdown'
            if self.accept == 'last':
                if touchdown:
                    self.pending[side] = (t, self.c)
                if side in self.pending and t-self.pending[side][0] >= self.cluster:
                    self._accept(side, *self.pending.pop(side))
                continue
            if touchdown:
                if t-self.last_touchdown.get(side, -math.inf) < self.min_gap or t < self.active_after:
                    self.last_touchdown.setdefault(side, t)
                    continue
                self.last_touchdown[side] = t
                self._accept(side, t, self.c)
        return self.c

    def _accept(self, side, t, c):
        if t < self.active_after:
            return
        if side not in self.nominal:  # learning the nominal phase online
            self.samples[side].append(c % 1)
            if len(self.samples[side]) >= self.learn:
                z = sum(complex(math.cos(2*math.pi*x), math.sin(2*math.pi*x)) for x in self.samples[side])
                self.nominal[side] = (math.atan2(z.imag, z.real)/(2*math.pi)) % 1
            return
        e = (self.nominal[side]-c % 1+.5) % 1-.5
        self.f = min(self.f0+self.clamp, max(self.f0-self.clamp, self.f0+self.k*e))
        self.events.append((round(t, 4), side, round(e, 4), round(self.f, 4)))


def antipump(roll_rate, stance, k_d, sign=1., clamp=.03):
    """Stage 2 (2): single-support damping on the stance hip roll only. The body roll follows -q_left and +q_right
    when that foot is planted (right hip-roll axis is -x), so sign=+1 asks for a body-roll acceleration of
    -k_d*roll_rate; sign=-1 is the opposite (sign is settled by experiment F pulses, evidence 74).
    stance: 'left', 'right' or None (double/no support: no correction). Returns {joint: offset}."""
    if stance is None or k_d == 0:
        return {}
    delta = max(-clamp, min(clamp, sign*k_d*roll_rate))
    return {'left_hip_roll_joint': delta} if stance == 'left' else {'right_hip_roll_joint': -delta}


class SlewLimiter:
    """Per-key rate limit (units/s) on a dict of offsets; keys missing from an update decay toward 0."""

    def __init__(self, rate=.5):
        self.rate, self.t, self.value = rate, None, {}

    def update(self, t, targets):
        dt = 0. if self.t is None else max(0., t-self.t)
        self.t = t
        for key in set(self.value) | set(targets):
            goal, now = targets.get(key, 0.), self.value.get(key, 0.)
            self.value[key] = now+max(-self.rate*dt, min(self.rate*dt, goal-now))
        return dict(self.value)
