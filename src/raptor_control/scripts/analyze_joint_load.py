#!/usr/bin/env python3
"""Read-only check of recorded gait telemetry against the gz_ros2_control position path.

gz_ros2_control 1.2.20 sets JointVelocityCmd = -gain*(position-command)*update_rate every
physics step; DART clips SERVO commands to the joint velocity limit. A sample at the limit
whose error cannot produce that command (or has the opposite sign) is not explained by
command saturation alone. Samples are sparse (IMU-triggered), so this is not a per-step trace.
"""
import argparse
import json
import math
import numpy as np


def roll_matrix(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def body_rotation(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def analyze(samples, targets, gain_per_s, limits, phase, fresh_s):
    velocity_rows, sole_rows = [], []
    for s in samples:
        if s['phase'] != phase or not s['joints']:
            continue
        for joint, target in targets.items():
            state = s['joints'][joint]
            error = state['position']-target
            command = max(-limits[joint], min(limits[joint], -gain_per_s*error))
            if abs(state['velocity']) >= .96*limits[joint]:
                velocity_rows.append({'sim_time': s['sim_time'], 'joint': joint, 'error_rad': error,
                    'p_command_rad_s': command, 'measured_rad_s': state['velocity'],
                    'explained_by_command': abs(command) >= .96*limits[joint] and command*state['velocity'] > 0,
                    'transmitted_effort': state['transmitted_effort']})
        rotation = body_rotation(s['orientation'])
        row = {'sim_time': s['sim_time'], 'body_roll_rad': math.atan2(rotation[2, 1], rotation[2, 2])}
        # Hip roll axes: left +x, right -x. Sole normal tilt ignores the pitch chain (small effect on roll).
        for side, sign in (('left', 1), ('right', -1)):
            normal = rotation @ roll_matrix(sign*s['joints'][f'{side}_hip_roll_joint']['position']) @ [0, 0, 1]
            points = [p for key, v in s['contacts'].items() if key.startswith(side)
                      and s['sim_time']-v['sim_time'] < fresh_s for p in v['points']]
            row[side] = {'sole_tilt_deg': math.degrees(math.acos(min(1., normal[2]))),
                         'fresh_contact_points': len(points),
                         'contact_y_span_mm': 1e3*float(np.ptp([p[1] for p in points])) if points else None}
        sole_rows.append(row)
    return {'limitation': 'IMU-triggered sparse samples; contact topics publish only non-empty contacts.',
            'limit_samples': len(velocity_rows),
            'not_explained_by_command': sum(not r['explained_by_command'] for r in velocity_rows),
            'velocity_limit_samples': velocity_rows, 'sole_contact': sole_rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('telemetry')
    parser.add_argument('--phase', default='shift_left')
    parser.add_argument('--ankle-target', type=float, default=-.3)
    parser.add_argument('--gain', type=float, default=.3)
    parser.add_argument('--update-rate', type=float, default=100.)
    parser.add_argument('--ankle-velocity-limit', type=float, default=2.5)
    parser.add_argument('--fresh', type=float, default=.03, help='max contact age vs IMU sample, s')
    args = parser.parse_args()
    ankles = [f'{side}_ankle_pitch_joint' for side in ('left', 'right')]
    data = json.load(open(args.telemetry))
    print(json.dumps(analyze(data['samples'], dict.fromkeys(ankles, args.ankle_target), args.gain*args.update_rate,
                             dict.fromkeys(ankles, args.ankle_velocity_limit), args.phase, args.fresh), indent=1))
