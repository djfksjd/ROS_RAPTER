#!/usr/bin/env python3
"""MuJoCo side of the evidence-73/74 lateral experiments, tail fixed. Gazebo runs use rock_gz_probe.py.

Usage: python lateral_experiments.py OUT.json {A|B|D|E|F} [--kv 20 30 100] [--cycles 150]
A/B rows: fall time and roll_diagnosis.table_metrics over [2 s, fall or end] (60 s gait).
F rows: rate-limited hip-roll steps from the crouch (rock_law.step_roll); per event the lifted foot, liftoff ->
touchdown time and release -> touchdown time from normal forces (off < 5 % weight, on >= 20 % held 20 ms).
"""
import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import roll_diagnosis as rd

SWEEPS = {'A': [('amplitude', a) for a in (.08, .04, .02, .01, 0.)],
          'B': [('window_width', w) for w in (.35, .30, .25)],
          'E': [('frequency', f) for f in (1.0, 1.4, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0)],
          'D': [('delay', d) for d in (0., .02, .04, .06)]}  # command delay injected in MuJoCo (evidence 74)


def one(job):
    kv, key, value, cycles = job
    frequency = value if key == 'frequency' else 2.5
    if key == 'frequency':
        cycles = int(round(60*frequency))  # always 60 s of gait
    series, result = rd.mujoco_series(kv, cycles, **{key: value})
    fall = result['stopped_by_guard_at_s']
    end = fall if fall else float(series['t'][-1])
    return {'kv': kv, key: value, 'fall_s': fall, 'x_m': result['final_xy'][0],
            **rd.table_metrics(series, 2., end, frequency)}


def transitions(t, fz, weight, lo, hi):
    """Debounced off (< 5 % weight) / on (>= 20 %) transitions per foot in [lo, hi) s."""
    out = []
    for side in ('left', 'right'):
        loaded = None
        for ti, f in zip(t, fz[side]):
            if not lo <= ti < hi:
                continue
            state = True if f >= .2*weight else False if f < .05*weight else loaded
            if loaded is not None and state != loaded:
                out.append((round(float(ti)-lo, 3), side, 'on' if state else 'off'))
            loaded = state
    return sorted(out)


def step_events(t, fz, weight, events, hold=.5, on_hold=.02):
    """Foot events after each step start in `events`: first foot below 5 % weight, then its touchdown.
    An event is marked unsettled if both feet were not loaded (>= 20 %) at its start."""
    t = np.asarray(t)
    steps = np.diff(t)
    if len(t) < 2 or not np.any(steps > 0):
        raise ValueError('need at least two increasing timestamps')
    n = max(1, int(round(on_hold/float(np.median(steps[steps > 0])))))  # samples that must stay loaded
    out = []
    for start in events:
        i0 = int(np.searchsorted(t, start))
        if i0 >= len(t):
            out.append({'event_start_s': start, 'settled': False, 'note': 'after the end of the data'})
            continue
        rec = {'event_start_s': start,
               'settled': bool(all(fz[s][i0] >= .2*weight for s in ('left', 'right'))),
               'transitions_1p5s': transitions(t, fz, weight, start, start+1.5)}
        for i in np.nonzero(t >= start)[0]:
            lifted = [s for s in ('left', 'right') if fz[s][i] < .05*weight]
            if lifted:
                side, t_off = lifted[0], float(t[i])
                rec |= {'lifted': side, 'liftoff_s': round(t_off-start, 4)}
                for j in range(i, len(t)-n):
                    if np.all(fz[side][j:j+n] >= .2*weight):
                        rec |= {'touchdown_s': round(float(t[j])-start, 4),
                                'single_support_s': round(float(t[j])-t_off, 4),
                                'release_to_touchdown_s': round(float(t[j])-(start+hold), 4)}
                        break
                break
            if t[i] > start+1.5:
                break
        out.append(rec)
    return out


def experiment_f(kv, amp=.08, hold=.5, settle=1., duration=5.5):
    import mujoco
    sys.path.insert(0, str(rd.HERE.parent/'src/raptor_control/scripts'))
    from raptor_servo import JTCLikeServo
    from rock_law import step_roll
    from rock_probe import foot_forces
    from stand_check import crouch, place_on_floor
    model = mujoco.MjModel.from_xml_path(str(rd.HERE/'raptor_digitigrade.xml'))
    model.actuator_gainprm[:, 0], model.actuator_biasprm[:, 2] = kv, -kv
    data = mujoco.MjData(model)
    for name, value in crouch(-.10, .5).items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    place_on_floor(model, data)
    servo = JTCLikeServo(model, .02, .02)
    servo.step(data)
    weight = float(model.body_mass.sum()*-model.opt.gravity[2])
    force, rows, dt = np.zeros(6), [], model.opt.timestep
    for k in range(int((settle+duration)/dt)):
        a = k*dt-settle
        r = step_roll(a, amp, hold)
        servo.set_target({'left_hip_roll_joint': -r, 'right_hip_roll_joint': r})
        servo.step(data)
        w, x, y, z = data.qpos[3:7]
        f = foot_forces(model, data, force)
        rows.append((a, r, math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)), f['left'], f['right']))
    arr = np.array(rows)
    t, fz = arr[:, 0], {'left': arr[:, 3], 'right': arr[:, 4]}
    events = step_events(t, fz, weight, [0., 3.], hold)
    for ev in events:
        m = (t >= ev['event_start_s']) & (t < ev['event_start_s']+1.)
        ev['roll_peak_rad'] = float(arr[m, 2][np.argmax(np.abs(arr[m, 2]))])
    curve = arr[::10]  # 100 Hz roll curve for the report
    return {'kv': kv, 'events': events,
            'curve': {'t': curve[:, 0].round(3).tolist(), 'r_cmd': curve[:, 1].round(4).tolist(),
                      'roll': curve[:, 2].round(4).tolist(), 'fz_left': curve[:, 3].round(1).tolist(),
                      'fz_right': curve[:, 4].round(1).tolist()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('experiment', choices=sorted(SWEEPS)+['F'])
    parser.add_argument('--kv', type=float, nargs='+', default=[20., 30., 100.])
    parser.add_argument('--cycles', type=int, default=150)
    args = parser.parse_args()
    with ProcessPoolExecutor(8) as pool:
        if args.experiment == 'F':
            rows = list(pool.map(experiment_f, args.kv))
        else:
            jobs = [(kv, key, value, args.cycles) for kv in args.kv for key, value in SWEEPS[args.experiment]]
            rows = list(pool.map(one, jobs))
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=1)
    print('wrote', args.out, len(rows), 'rows')


if __name__ == '__main__':
    main()
