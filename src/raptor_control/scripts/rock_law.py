"""Open-loop rocking + stride position targets, ROS-free (same law as sim/rock_probe.py with lift = 0).

a: seconds since rocking started (after settle). Hip rolls: left = -r, right = +r, r = A*min(1, a/ramp)*sin(2*pi*f*a).
Stride: inside each foot's unloaded window (width `window` cycles, centred at `centers`) the hip pitch swings
forward linearly and returns during stance; the ankle cancels it. Optional ankle pitch feedback on base
pitch and pitch rate, clipped. Position targets only; the caller owns STOP.
"""
import math

CENTERS = (('left', .28), ('right', .78))


def crouch_pose(joints, hip, knee):
    pose = dict.fromkeys(joints, 0.)
    for side in ('left', 'right'):
        pose |= {f'{side}_hip_pitch_joint': hip, f'{side}_knee_pitch_joint': knee,
                 f'{side}_ankle_pitch_joint': -hip-knee}
    return pose


def targets(pose, a, amplitude, frequency, ramp, stride=0., window=.35, pitch=0., pitch_rate=0., feedback=None):
    target = dict(pose)
    if a < 0:
        return target
    r = amplitude*min(1., a/ramp)*math.sin(2*math.pi*frequency*a)
    target |= {'left_hip_roll_joint': -r, 'right_hip_roll_joint': r}
    cycle = (a*frequency) % 1
    for side, center in CENTERS:
        offset = (cycle-center+.5) % 1-.5
        u = (offset+window/2)/window
        swing = -stride*(u-.5) if 0 <= u <= 1 else -stride*(.5-((offset-window/2) % 1)/(1-window))
        swing *= min(1., max(0., a-ramp))
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
