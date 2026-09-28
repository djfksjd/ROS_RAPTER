#!/usr/bin/env python3
"""MuJoCo static standing check. Default starts in the crouch pose (like Gazebo crouched_start)
and holds it; --from-zero ramps from the zero pose like the JTC single-point goal.

Static support only. Passing this is not balance control, stepping or walking.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import mujoco
from raptor_servo import GazeboLikeServo

HERE = Path(__file__).resolve().parent


def crouch(hip):
    return {f'{s}_{j}_joint': v for s in ('left', 'right')
            for j, v in (('hip_pitch', hip), ('knee_pitch', .4), ('ankle_pitch', -.4-hip))}


def place_on_floor(model, data):
    mujoco.mj_forward(model, data)
    floor = model.geom('floor').id
    lowest = min(data.geom_xpos[g][2]-model.geom_rbound[g] for g in range(model.ngeom) if g != floor)
    # rbound is conservative; refine with the actual lowest box corner.
    corners = []
    for g in range(model.ngeom):
        if g == floor:
            continue
        half, rot, pos = model.geom_size[g], data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
        corners += [(pos + rot @ (half*np.array(s)))[2] for s in np.array(np.meshgrid(*[[-1, 1]]*3)).T.reshape(-1, 3)]
    data.qpos[2] -= min(corners) if corners else lowest
    mujoco.mj_forward(model, data)


def tilt(quat):
    w = min(1., abs(quat[0]))
    return 2*math.acos(w)  # includes yaw; yaw stays ~0 here and is reported separately


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(HERE/'raptor.xml'))
    parser.add_argument('--crouch-hip', type=float, default=-.15)
    parser.add_argument('--ramp', type=float, default=2.)
    parser.add_argument('--hold', type=float, default=10.)
    parser.add_argument('--from-zero', action='store_true')
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    goal = crouch(args.crouch_hip)
    if not args.from_zero:
        args.ramp = 0.
        for name, value in goal.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    place_on_floor(model, data)
    servo = GazeboLikeServo(model)
    servo.step(data)
    start = servo.read.copy()
    names = list(goal)
    dt = model.opt.timestep
    rows, worst = [], {'tilt_rad': 0., 'ankle_speed': 0.}
    z0 = float(data.qpos[2])
    for k in range(int((args.ramp+args.hold)/dt)):
        t = k*dt
        if args.ramp and t <= args.ramp:  # linear interpolation like the observed JTC ramp
            a = t/args.ramp
            servo.set_target({n: (1-a)*start[servo_index(n)] + a*goal[n] for n in names})
        servo.step(data)
        w = tilt(data.qpos[3:7])
        worst['tilt_rad'] = max(worst['tilt_rad'], w)
        worst['ankle_speed'] = max(worst['ankle_speed'], *(abs(data.qvel[model.jnt_dofadr[model.joint(f'{s}_ankle_pitch_joint').id]]) for s in ('left', 'right')))
        if k % 500 == 0:
            rows.append({'t': round(t, 3), 'base_z': float(data.qpos[2]), 'tilt_rad': w, 'ncon': int(data.ncon)})
    error = max(abs(data.qpos[model.jnt_qposadr[model.joint(n).id]]-goal[n]) for n in names)
    result = {'start': 'zero' if args.from_zero else 'crouch', 'limitation': 'MuJoCo static standing with a Gazebo-like servo approximation; not walking.',
              'crouch_hip': args.crouch_hip, 'initial_base_z': z0, 'final_base_z': float(data.qpos[2]),
              'final_xy': data.qpos[:2].tolist(), 'max_tilt_rad': worst['tilt_rad'],
              'max_ankle_speed_rad_s': worst['ankle_speed'], 'final_tracking_error_rad': float(error),
              'passed_static_standing': bool(worst['tilt_rad'] < .1 and error < .025), 'samples': rows}
    print(json.dumps(result, indent=1))


def servo_index(name):
    from raptor_servo import ACTIVE
    return ACTIVE.index(name)


if __name__ == '__main__':
    main()
