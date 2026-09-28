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
