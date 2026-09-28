"""Lateral-slope static stand: 12-DOF (ankle roll) vs 10-DOF, same method (evidence 77).

Gravity is tilted instead of the floor (gravity +y: the right side is uphill by phi). 12 DOF: body kept upright
(rolled phi relative to the floor), hip roll 0, ankle roll levels both soles; the uphill leg is
shortened by bending its knee until both soles touch. 10 DOF: evidence-75 method
(terrain_feasibility.lateral_kinematics, hip roll rolls the body). Hold = fixed joint targets through
the velocity servo (kv 30, gain 30/s) for 5 s; "stands" = tilt change < 0.1 rad and drift < 5 cm.
No controller: a static-support check, not walking on slopes.
"""
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import terrain_feasibility as tf  # noqa: E402
from stand_check import place_on_floor  # noqa: E402

MODEL12 = HERE/'raptor_digitigrade_ankleroll.xml'
G, KV = 9.81, 30.


def load12():
    m = mujoco.MjModel.from_xml_path(str(MODEL12))
    m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 2] = KV, -KV
    return m, mujoco.MjData(m)


def sole(m, d, side):
    g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f'{side}_foot_link').id)
    rot = d.geom_xmat[g].reshape(3, 3)
    return d.geom_xpos[g]-rot[:, 2]*m.geom_size[g][2], rot


def pose12(m, d, phi, hip=-.1):
    """Body rolled phi about x (upright in the tilted-gravity frame), soles levelled by ankle roll."""
    quat = (math.cos(phi/2), math.sin(phi/2), 0, 0)

    def legs(knee_short, short, roll):
        pose = {}
        for s in ('left', 'right'):
            knee = knee_short if s == short else .5
            pose |= {f'{s}_hip_pitch_joint': hip, f'{s}_knee_pitch_joint': knee,
                     f'{s}_ankle_pitch_joint': -(hip+knee), f'{s}_ankle_roll_joint': roll}
        tf.set_pose(m, d, pose, quat)
        return pose

    # ankle roll sign/size that makes the soles parallel to the floor (sole normal = world z here)
    roll = min((r for r in np.linspace(-.5, .5, 201)), key=lambda r: (legs(.5, 'left', r), 1-sole(m, d, 'left')[1][2, 2])[1])
    legs(.5, 'left', roll)
    short = 'left' if sole(m, d, 'left')[0][2] < sole(m, d, 'right')[0][2] else 'right'
    lo, hi = .5, 2.2
    other = 'right' if short == 'left' else 'left'
    for _ in range(40):
        mid = .5*(lo+hi)
        legs(mid, short, roll)
        if sole(m, d, short)[0][2] < sole(m, d, other)[0][2]:
            lo = mid
        else:
            hi = mid
    pose = legs(lo, short, roll)
    level = 1-min(sole(m, d, s)[1][2, 2] for s in ('left', 'right'))
    return pose, quat, {'ankle_roll': round(float(roll), 3), 'short_leg': short, 'short_knee': round(lo, 3),
                        'sole_tilt_rad': round(float(math.acos(1-level)), 4)}


def hold(m, d, pose, quat, gravity, duration=5.):
    m.opt.gravity[:] = gravity
    tf.set_pose(m, d, pose, quat)
    place_on_floor(m, d)
    xy0, t0 = d.qpos[:2].copy(), tf.tilt(d.qpos[3:7])
    act = [(i, m.jnt_qposadr[m.actuator_trnid[i, 0]], pose.get(m.actuator(i).name, 0.)) for i in range(m.nu)]
    vmax, worst = m.actuator_ctrlrange[:, 1], 0.
    while d.time < duration:
        for a, q, want in act:
            d.ctrl[a] = np.clip(-30.*(d.qpos[q]-want), -vmax[a], vmax[a])
        mujoco.mj_step(m, d)
        worst = max(worst, abs(tf.tilt(d.qpos[3:7])-t0))
    final, drift = abs(tf.tilt(d.qpos[3:7])-t0), float(np.linalg.norm(d.qpos[:2]-xy0))
    return {'max_tilt_change_rad': round(worst, 3), 'final_tilt_change_rad': round(final, 3),
            'xy_drift_m': round(drift, 4), 'stands_5s': bool(final < .1 and drift < .05)}


def main():
    rows = []
    for deg in (5, 10, 15, 20, 25):
        ph = math.radians(deg)
        gravity = (0, G*math.sin(ph), -G*math.cos(ph))
        m, d = load12()
        pose, quat, info = pose12(m, d, ph)
        m, d = load12()
        r12 = {'dof': 12, 'deg': deg, **info, **hold(m, d, pose, quat, gravity)}
        m, d = tf.load()
        kin = tf.lateral_kinematics(m, d, ph)
        if kin is None:
            r10 = {'dof': 10, 'deg': deg, 'kinematic': 'infeasible'}
        else:
            m, d = tf.load()
            r10 = {'dof': 10, 'deg': deg, **tf.slope_static(m, d, kin['pose'], kin['quat'], gravity)}
        rows += [r12, r10]
        print(json.dumps(r12)); print(json.dumps(r10), flush=True)
    out = HERE.parent/'docs/evidence/77-ankle-roll/lateral_slopes.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1)+'\n')


if __name__ == '__main__':
    main()
