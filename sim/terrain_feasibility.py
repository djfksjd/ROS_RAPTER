#!/usr/bin/env python3
"""Drop-landing, slope-standing and running-speed bounds for the 10-DOF Raptor MuJoCo model.

No controller, no policy: fixed position targets through the velocity-servo actuators (velocity command
= clip(gain*(target-q), +-ctrlrange); actuator force = clip(kv*(cmd-qvel), +-forcerange), kv=100 in the model).
  drop     : standing crouch (hip -0.10, knee 0.5, ankle -0.4) released from `height` above the floor with the
             servo target fixed; peak torque vs limit, saturation time, peak knee speed, GRF, tilt.
  slopes   : kinematic search for the steepest fore-aft / lateral slope on which both flat soles can lie on the
             ground while the CoM plumb line falls inside the rigid sole (|dx| <= 60 mm, |dy| inside the feet),
             ankle pitch +-0.7, hip roll +-0.5, no ankle roll, body within 0.1 rad of gravity-vertical;
             then a 5 s MuJoCo static hold with gravity tilted instead of the floor.
  running  : analytic hip-swing speed table and a zero-gravity swing-tracking check with the current limits.
Velocity limits (ctrlrange) and effort limits (forcerange) can be scaled; servo stiffness kv defaults to 30 as in
the evidence-74 MuJoCo/Gazebo comparisons (XML default 100). Not a walking/landing certificate.
Usage: .venv-sim/bin/python sim/terrain_feasibility.py [--out-dir docs/evidence/75-feasibility] [--kv 30]
"""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from stand_check import crouch, place_on_floor

HERE = Path(__file__).resolve().parent
MODEL = HERE/'raptor_digitigrade.xml'
LEG = ('hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')
START = crouch(-.10, .5)  # standing crouch used by rock_probe / lateral_experiments
G = 9.81
KV = 30.


def load(v_scale=1., effort_scale=1.):
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    m.actuator_ctrlrange[:] *= v_scale
    m.actuator_forcerange[:] *= effort_scale
    m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 2] = KV, -KV
    return m, mujoco.MjData(m)


def tilt(q):
    w, x, y, z = q
    return math.acos(max(-1., min(1., 1-2*(x*x+y*y))))


class Hold:
    """Fixed joint targets through the model's velocity actuators (gain 30/s like the Gazebo-like servo)."""

    def __init__(self, m, targets, gain=30.):
        self.m, self.gain = m, gain
        self.act = [(m.actuator(n).id, m.jnt_qposadr[m.joint(n).id], targets.get(n, 0.)) for n in
                    [f'{s}_{j}_joint' for s in ('left', 'right') for j in LEG]+['tail_yaw_joint', 'tail_pitch_joint']]
        self.vmax = m.actuator_ctrlrange[:, 1]

    def step(self, d):
        for a, q, want in self.act:
            d.ctrl[a] = np.clip(-self.gain*(d.qpos[q]-want), -self.vmax[a], self.vmax[a])
        mujoco.mj_step(self.m, d)


def set_pose(m, d, joints, quat=(1, 0, 0, 0)):
    d.qpos[:] = 0
    d.qpos[3:7] = quat
    for name, value in joints.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
    mujoco.mj_forward(m, d)


def ground_force(m, d, buf):
    total = 0.
    for i in range(d.ncon):
        c = d.contact[i]
        if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
            mujoco.mj_contactForce(m, d, i, buf)
            total += buf[0]
    return total


# ---------------------------------------------------------------- drop landing
def absorb_bound(heights=(.1, .2, .3, .5), knee_limit=3.):
    """Analytic: knee speed needed to absorb the impact by leg flexion instead of rigid contact. Vertical gain
    dz/dknee is taken from the standing-crouch family crouch(-0.10, knee) (ankle keeps the sole level)."""
    m, d = load()
    hs = []
    for knee in (.45, .55):
        set_pose(m, d, crouch(-.10, knee))
        hs.append(float(d.subtree_com[m.body('base_root').id][2]-sole_bottom(m, d, 'left')[0][2]))
    gain = abs(hs[1]-hs[0])/.1  # m per rad of knee
    rows = []
    for h in heights:
        v = math.sqrt(2*G*h)
        rows.append({'height_m': h, 'impact_speed_mps': round(v, 3), 'knee_speed_needed_rad_s': round(v/gain, 1),
                     'ratio_to_knee_limit': round(v/gain/knee_limit, 1),
                     'note': 'leg can flex at most %.2f m/s at the knee limit; faster impacts are taken rigidly' % (gain*knee_limit)})
    return {'dz_dknee_m_per_rad': round(gain, 4), 'knee_limit_rad_s': knee_limit, 'rows': rows}


def drop(height, v_scale=1., effort_scale=1., duration=2.):
    m, d = load(v_scale, effort_scale)
    set_pose(m, d, START)
    place_on_floor(m, d)
    d.qpos[2] += height
    mujoco.mj_forward(m, d)
    base = m.body('base_root').id
    z_start = float(d.subtree_com[base][2])
    hold = Hold(m, START)
    names = {j: m.actuator(f'left_{j}_joint').id for j in LEG}
    vadr = {j: m.jnt_dofadr[m.joint(f'left_{j}_joint').id] for j in LEG}
    limit = {j: float(m.actuator_forcerange[a, 1]) for j, a in names.items()}
    peak, sat, speed = {j: 0. for j in LEG}, {j: 0 for j in LEG}, {j: 0. for j in LEG}
    buf, grf, impact, zmin, worst = np.zeros(6), 0., None, z_start, 0.
    dt = m.opt.timestep
    while d.time < duration:
        hold.step(d)
        f = ground_force(m, d, buf)
        if impact is None and f > 0:
            impact = d.time
        grf = max(grf, f)
        zmin = min(zmin, float(d.subtree_com[base][2]))
        worst = max(worst, tilt(d.qpos[3:7]))
        for j, a in names.items():
            force = abs(float(d.actuator_force[a]))
            peak[j] = max(peak[j], force)
            speed[j] = max(speed[j], abs(float(d.qvel[vadr[j]])))
            if impact is not None and d.time-impact < .3 and force >= .99*limit[j]:
                sat[j] += 1
    weight = float(m.body_mass.sum()*G)
    final = tilt(d.qpos[3:7])
    return {'height_m': height, 'v_scale': v_scale, 'effort_scale': effort_scale, 'kv': KV,
            'impact_speed_mps_est': round(math.sqrt(2*G*height), 3),
            'peak_torque_nm': {j: round(v, 1) for j, v in peak.items()}, 'torque_limit_nm': limit,
            'saturated_ms_in_first_300ms': {j: round(v*dt*1e3, 1) for j, v in sat.items()},
            'peak_joint_speed_rad_s': {j: round(v, 2) for j, v in speed.items()},
            'peak_grf_over_weight': round(grf/weight, 2), 'com_compression_m': round(z_start-height-zmin, 4),
            'max_tilt_rad': round(worst, 3), 'final_tilt_rad': round(final, 3),
            'final_com_height_m': round(float(d.subtree_com[base][2]), 3),
            'upright': bool(final < .3 and d.subtree_com[base][2] > .5)}


# ---------------------------------------------------------------- slopes
def sole_geom(m, side):
    body = m.body(f'{side}_foot_link').id
    return next(g for g in range(m.ngeom) if m.geom_bodyid[g] == body)


def sole_bottom(m, d, side):
    g = sole_geom(m, side)
    rot = d.geom_xmat[g].reshape(3, 3)
    return d.geom_xpos[g]-rot[:, 2]*m.geom_size[g][2], rot


def foreaft_kinematics(m, d, theta, body_offsets=(-.1, 0., .1), margin=.06):
    """Uphill theta>0 (rad). In the flattened frame gravity is (-g sin, 0, -g cos), the body is pitched theta
    (+offset) about y. Returns the best flat-sole pose whose CoM plumb line lands within `margin` of the sole
    centre, or None."""
    best = None
    lim = {j: m.jnt_range[m.joint(f'left_{j}_joint').id] for j in LEG}
    for off in body_offsets:
        pitch = theta+off
        for hip in np.arange(lim['hip_pitch'][0], lim['hip_pitch'][1]+1e-9, .05):
            for knee in np.arange(lim['knee_pitch'][0], lim['knee_pitch'][1]+1e-9, .05):
                ankle = -(hip+knee)-pitch
                if not lim['ankle_pitch'][0] <= ankle <= lim['ankle_pitch'][1]:
                    continue
                pose = {f'{s}_{j}_joint': v for s in ('left', 'right') for j, v in
                        (('hip_pitch', hip), ('knee_pitch', knee), ('ankle_pitch', ankle))}
                set_pose(m, d, pose, (math.cos(pitch/2), 0, math.sin(pitch/2), 0))
                com = d.subtree_com[m.body('base_root').id]
                bottom, _ = sole_bottom(m, d, 'left')
                h = com[2]-bottom[2]
                dx = com[0]-h*math.tan(theta)-bottom[0]
                if abs(dx) <= margin:
                    score = abs(dx)+.2*abs(h-.75)
                    if best is None or score < best['score']:
                        best = {'score': score, 'body_pitch_rel_floor': round(pitch, 3), 'hip_pitch': round(hip, 3),
                                'knee_pitch': round(knee, 3), 'ankle_pitch': round(ankle, 3),
                                'com_plumb_dx_m': round(dx, 4), 'com_height_m': round(h, 3)}
    return best


def lateral_kinematics(m, d, phi, body_offsets=(-.1, 0., .1), hip=-.1, margin=.05):
    """Left side uphill phi>0. Gravity (0, g sin, -g cos) in the flattened frame; body rolled phi(+offset) about x;
    hip rolls cancel it so both soles are flat; the lower-hip leg keeps knee 0.5, the other knee is solved so both
    soles touch the same floor. Feasible if the CoM plumb line lands between the sole centres (+-margin)."""
    lim_roll = m.jnt_range[m.joint('left_hip_roll_joint').id]
    best = None
    for off in body_offsets:
        roll = phi+off
        if abs(roll) > lim_roll[1]:
            continue
        quat = (math.cos(roll/2), math.sin(roll/2), 0, 0)
        rolls = {'left_hip_roll_joint': -roll, 'right_hip_roll_joint': roll}  # right axis is -x

        def legs(knee_other, long_side):
            pose = dict(rolls)
            for s in ('left', 'right'):
                knee = .5 if s == long_side else knee_other
                pose |= {f'{s}_hip_pitch_joint': hip, f'{s}_knee_pitch_joint': knee, f'{s}_ankle_pitch_joint': -(hip+knee)}
            set_pose(m, d, pose, quat)
            return pose

        # the hip that ends up higher keeps the long (knee 0.5) leg; the other leg is shortened
        legs(.5, 'left')
        zl, zr = sole_bottom(m, d, 'left')[0][2], sole_bottom(m, d, 'right')[0][2]
        long_side, short_side = ('left', 'right') if zl > zr else ('right', 'left')
        lo, hi = .5, 2.2
        for _ in range(40):
            mid = .5*(lo+hi)
            legs(mid, long_side)
            zs, zl_ = sole_bottom(m, d, short_side)[0][2], sole_bottom(m, d, long_side)[0][2]
            if zs < zl_:  # short leg's sole still below the long leg's -> bend more
                lo = mid
            else:
                hi = mid
        pose = legs(lo, long_side)
        bl, rl = sole_bottom(m, d, 'left')
        br, rr = sole_bottom(m, d, 'right')
        if abs(bl[2]-br[2]) > 2e-3 or abs(rl[2, 2]-1) > 1e-3 or abs(rr[2, 2]-1) > 1e-3:
            continue
        com = d.subtree_com[m.body('base_root').id]
        h = com[2]-bl[2]
        y = com[1]+h*math.tan(phi)
        lo_y, hi_y = min(bl[1], br[1])-margin, max(bl[1], br[1])+margin
        if lo_y <= y <= hi_y:
            dist = min(y-lo_y, hi_y-y)
            if best is None or dist > best['margin_m']:
                best = {'margin_m': round(dist, 4), 'body_roll_rel_floor': round(roll, 3), 'hip_roll_abs': round(abs(roll), 3),
                        'short_leg': short_side, 'short_knee': round(lo, 3), 'com_plumb_y_m': round(y, 4),
                        'feet_y_m': [round(bl[1], 3), round(br[1], 3)], 'com_height_m': round(h, 3), 'pose': pose, 'quat': quat}
    return best


def slope_static(m, d, pose, quat, gravity, duration=5.):
    m.opt.gravity[:] = gravity
    set_pose(m, d, pose, quat)
    place_on_floor(m, d)
    xy0, t0 = d.qpos[:2].copy(), tilt(d.qpos[3:7])
    hold = Hold(m, pose)
    worst = 0.
    while d.time < duration:
        hold.step(d)
        worst = max(worst, abs(tilt(d.qpos[3:7])-t0))
    drift = float(np.linalg.norm(d.qpos[:2]-xy0))
    final = abs(tilt(d.qpos[3:7])-t0)
    return {'max_tilt_change_rad': round(worst, 3), 'final_tilt_change_rad': round(final, 3), 'xy_drift_m': round(drift, 4),
            'stands_5s': bool(final < .1 and drift < .05)}


def slopes():
    m, d = load()
    out = {'fore_aft': {}, 'lateral': {}}
    for deg in (5, 10, 15, 20, 25, 30, 35):
        for sign, label in ((1, 'uphill'), (-1, 'downhill')):
            th = math.radians(sign*deg)
            kin = foreaft_kinematics(m, d, th)
            row = {'kinematic': kin}
            if kin and deg <= 25:
                pose = {f'{s}_{j}_joint': kin[j] for s in ('left', 'right') for j in ('hip_pitch', 'knee_pitch', 'ankle_pitch')}
                p = kin['body_pitch_rel_floor']
                ms, ds = load()
                row['static'] = slope_static(ms, ds, pose, (math.cos(p/2), 0, math.sin(p/2), 0),
                                             (-G*math.sin(th), 0, -G*math.cos(th)))
            out['fore_aft'][f'{label}_{deg}deg'] = row
    for deg in (5, 10, 15, 20, 25):
        ph = math.radians(deg)
        kin = lateral_kinematics(m, d, ph)
        row = {'kinematic': None if kin is None else {k: v for k, v in kin.items() if k not in ('pose', 'quat')}}
        if kin:
            ms, ds = load()
            row['static'] = slope_static(ms, ds, kin['pose'], kin['quat'], (0, G*math.sin(ph), -G*math.cos(ph)))
        out['lateral'][f'lateral_{deg}deg'] = row
    return out


# ---------------------------------------------------------------- running
def running(leg_length, v_hip=2.5):
    rows = []
    for f in (1.5, 2., 2.5, 3.):
        for A in (.4, .5, .6):
            peak_sin = 2*math.pi*f*A
            peak_swing40 = math.pi/2*2*A/(.4/f)  # cosine swing across 40 % of the cycle
            rows.append({'stride_hz': f, 'amplitude_rad': A, 'peak_hip_speed_sinusoid_rad_s': round(peak_sin, 2),
                         'peak_hip_speed_40pct_swing_rad_s': round(peak_swing40, 2),
                         'ratio_to_limit_sinusoid': round(peak_sin/v_hip, 2), 'ratio_to_limit_40pct': round(peak_swing40/v_hip, 2)})
    walk = []
    for A in (.2, .3, .4, .5, .6):
        f_sin = v_hip/(2*math.pi*A)
        f_40 = v_hip/(math.pi/2*2*A/.4)
        stride = 2*leg_length*math.sin(A)
        walk.append({'amplitude_rad': A, 'stride_m': round(stride, 3), 'max_f_sinusoid_hz': round(f_sin, 2),
                     'max_speed_sinusoid_mps': round(stride*f_sin, 3), 'max_f_40pct_swing_hz': round(f_40, 2),
                     'max_speed_40pct_swing_mps': round(stride*f_40, 3)})
    froude1 = math.sqrt(G*leg_length)
    return {'leg_length_m': round(leg_length, 3), 'hip_limit_rad_s': v_hip, 'swing_table': rows, 'walking_bound': walk,
            'stance_leg_speed_bound_mps': round(v_hip*leg_length, 3),
            'walk_run_transition_froude_0_5_mps': round(math.sqrt(.5*G*leg_length), 3),
            'running_froude_1_mps': round(froude1, 3),
            'running_2_75hz_0_5rad_peak_hip_rad_s': {'sinusoid': round(2*math.pi*2.75*.5, 2), '40pct_swing': round(math.pi/2*1./(.4/2.75), 2)}}


def swing_check(f, A, v_scale=1., cycles=4):
    """Zero gravity, free base: antiphase hip-pitch sinusoids (ankle cancels) through the velocity servo."""
    m, d = load(v_scale)
    m.opt.gravity[:] = 0
    set_pose(m, d, START)
    d.qpos[2] = 1.5
    hold = Hold(m, dict(START))
    hip = {s: m.jnt_qposadr[m.joint(f'{s}_hip_pitch_joint').id] for s in ('left', 'right')}
    vel = {s: m.jnt_dofadr[m.joint(f'{s}_hip_pitch_joint').id] for s in ('left', 'right')}
    torque = m.actuator('left_hip_pitch_joint').id
    reached, speed, peak_t = 0., 0., 0.
    while d.time < cycles/f:
        targets = {}
        for i, s in enumerate(('left', 'right')):
            sw = (1 if i == 0 else -1)*A*math.sin(2*math.pi*f*d.time)
            targets[f'{s}_hip_pitch_joint'] = START[f'{s}_hip_pitch_joint']+sw
            targets[f'{s}_ankle_pitch_joint'] = START[f'{s}_ankle_pitch_joint']-sw
        hold.act = [(a, q, targets.get(m.actuator(a).name, want)) for a, q, want in hold.act]
        hold.step(d)
        if d.time > 1/f:
            reached = max(reached, abs(float(d.qpos[hip['left']])-START['left_hip_pitch_joint']))
            speed = max(speed, abs(float(d.qvel[vel['left']])))
            peak_t = max(peak_t, abs(float(d.actuator_force[torque])))
    return {'stride_hz': f, 'amplitude_rad': A, 'v_scale': v_scale, 'achieved_amplitude_rad': round(reached, 3),
            'amplitude_ratio': round(reached/A, 3), 'peak_hip_speed_rad_s': round(speed, 2), 'peak_hip_torque_nm': round(peak_t, 1),
            'hip_speed_limit_rad_s': float(m.actuator_ctrlrange[torque, 1])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default=str(HERE.parent/'docs/evidence/75-feasibility'))
    p.add_argument('--parts', nargs='+', default=['drop', 'slopes', 'running'])
    p.add_argument('--kv', type=float, default=30.)
    a = p.parse_args()
    global KV
    KV = a.kv
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if 'drop' in a.parts:
        rows = [drop(h, v, e) for v, e in ((1., 1.), (4., 1.), (4., 2.), (4., 10.)) for h in (.1, .2, .3, .5)]
        for r in rows:
            print(json.dumps({k: r[k] for k in ('height_m', 'v_scale', 'effort_scale', 'peak_torque_nm', 'peak_joint_speed_rad_s',
                                                 'peak_grf_over_weight', 'max_tilt_rad', 'upright')}))
        bound = absorb_bound()
        print('absorb bound', bound)
        (out/'drop_landing.json').write_text(json.dumps({'limitation': 'Fixed servo target, no landing control; effort 10x = '
                                                         'torque the stiff hold would demand.', 'kv': KV, 'absorb_bound': bound,
                                                         'results': rows}, indent=1)+'\n')
    if 'slopes' in a.parts:
        res = slopes()
        for k, v in list(res['fore_aft'].items())+list(res['lateral'].items()):
            print(k, 'kin' if v['kinematic'] else 'NO-KIN', v.get('static'))
        res['kv'] = KV
        (out/'slopes.json').write_text(json.dumps(res, indent=1)+'\n')
    if 'running' in a.parts:
        m, d = load()
        set_pose(m, d, START)
        L = float(d.xpos[m.body('left_thigh_link').id][2]-sole_bottom(m, d, 'left')[0][2])
        res = running(L)
        res['kv'] = KV
        res['swing_check'] = [swing_check(f, A, v) for v in (1., 4.) for f, A in ((1.5, .4), (2., .5), (3., .5))]
        for r in res['swing_check']:
            print(json.dumps(r))
        print('walking bound', res['walking_bound'])
        (out/'running.json').write_text(json.dumps(res, indent=1)+'\n')


if __name__ == '__main__':
    main()
