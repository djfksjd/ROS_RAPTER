#!/usr/bin/env python3
"""Vertical-jump upper bound for the 10-DOF Raptor MuJoCo model under joint velocity/effort limits.

Not a controller and not a policy. Sequence per trial:
  1. start in the standing crouch used by rock_probe/lateral_experiments (hip -0.10, knee 0.5, ankle -0.4);
  2. lower slowly along a *balanced* leg path (whole-body CoM kept ~30 mm behind the sole centre, sole flat,
     body pitch 0.05 rad) to the deepest reachable crouch (ankle limit -0.7 binds); tilt must stay < 0.1 rad;
  3. extend along the same path with coordinated feedforward joint velocities. Profile 'bang' (default, the
     slowest-joint-limited rate): at every instant the binding joint runs *at* its velocity limit. Profile
     'ramp': the binding joint's speed rises linearly and reaches the limit at the end of the stroke. On this
     path the vertical gain dz/du falls as the leg straightens, so the CoM decelerates late in the stroke and
     the feet leave the ground before full extension in either profile; a kinematic bound (dz/dt at the limit
     rate along the path) is reported next to the measured takeoff;
  4. hold the extended pose and measure CoM rise, flight time, takeoff velocity (mj_subtreeVel at the last
     ground-contact step), peak actuator torque and body tilt.
Velocity limits (ctrlrange) and effort limits (forcerange) are scaled to show what spec a jump would need.
Servo stiffness kv (actuator force = kv*(cmd-qvel), clipped to forcerange) defaults to 30 as in the evidence-74
MuJoCo/Gazebo comparisons (the XML default 100 makes foot contact chatter; results at 100 are labelled).
Ground contact = any contact against the world body (sole or passive toes).
Usage: .venv-sim/bin/python sim/jump_feasibility.py [--scales 1 2 4 6 8] [--effort 1 1.5] [--profile bang|ramp] [--kv 30] [--out f.json]
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
START = {'hip_pitch': -.10, 'knee_pitch': .5, 'ankle_pitch': -.4}
COM_DX, BODY_PITCH, SOLE_HALF = -.03, .05, .0175


def leg_path(m, dx=COM_DX, pitch=BODY_PITCH, knees=np.arange(0., 2.2, .01)):
    """Balanced flat-sole poses parametrised by knee angle: for each knee, hip is solved (bisection) so the
    whole-body CoM sits `dx` ahead of the sole centre; ankle = -(hip+knee)-pitch keeps the sole flat.
    Returns arrays (knee, hip, ankle, com_height_above_sole) restricted to poses inside the joint ranges."""
    d = mujoco.MjData(m)
    foot = m.body('left_foot_link').id
    sole = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == foot)
    lim = {j: m.jnt_range[m.joint(f'left_{j}_joint').id] for j in LEG}

    def com_dx(hip, knee):
        ankle = -(hip+knee)-pitch
        d.qpos[:] = 0
        d.qpos[3:7] = [math.cos(pitch/2), 0, math.sin(pitch/2), 0]
        for s in ('left', 'right'):
            for j, v in (('hip_pitch', hip), ('knee_pitch', knee), ('ankle_pitch', ankle)):
                d.qpos[m.jnt_qposadr[m.joint(f'{s}_{j}_joint').id]] = v
        mujoco.mj_forward(m, d)
        com, rot = d.subtree_com[m.body('base_root').id], d.geom_xmat[sole].reshape(3, 3)
        bottom = d.geom_xpos[sole]-rot[:, 2]*SOLE_HALF
        return com[0]-bottom[0], com[2]-bottom[2], ankle

    rows = []
    for knee in knees:
        lo, hi = lim['hip_pitch']
        if (com_dx(lo, knee)[0]-dx)*(com_dx(hi, knee)[0]-dx) > 0:
            continue
        for _ in range(40):
            mid = .5*(lo+hi)
            if (com_dx(mid, knee)[0]-dx)*(com_dx(lo, knee)[0]-dx) <= 0:
                hi = mid
            else:
                lo = mid
        _, h, ankle = com_dx(lo, knee)
        if lim['ankle_pitch'][0] <= ankle <= lim['ankle_pitch'][1]:
            rows.append((knee, lo, ankle, h))
    return np.array(rows).T


def tilt(q):
    w, x, y, z = q
    return math.acos(max(-1., min(1., 1-2*(x*x+y*y))))


class Trial:
    def __init__(self, v_scale=1., effort_scale=1., kv=30., gain=30.):
        m = mujoco.MjModel.from_xml_path(str(MODEL))
        m.actuator_ctrlrange[:] *= v_scale
        m.actuator_forcerange[:] *= effort_scale
        m.actuator_gainprm[:, 0], m.actuator_biasprm[:, 2] = kv, -kv
        self.kv = kv
        self.m, self.d, self.gain = m, mujoco.MjData(m), gain
        self.act = {(s, j): m.actuator(f'{s}_{j}_joint').id for s in ('left', 'right') for j in LEG}
        self.qadr = {k: m.jnt_qposadr[m.actuator_trnid[a, 0]] for k, a in self.act.items()}
        self.vadr = {k: m.jnt_dofadr[m.actuator_trnid[a, 0]] for k, a in self.act.items()}
        self.vmax = {k: float(m.actuator_ctrlrange[a, 1]) for k, a in self.act.items()}
        self.fmax = {k: float(m.actuator_forcerange[a, 1]) for k, a in self.act.items()}
        self.tail = [m.actuator(n).id for n in ('tail_yaw_joint', 'tail_pitch_joint')]
        self.base = m.body('base_root').id
        self.path = leg_path(m)  # knee, hip, ankle, height

    def pose_at(self, u):
        """Interpolate the balanced path; u=0 deepest crouch, u=1 highest pose."""
        knee, hip, ankle, _ = self.path
        s = knee[-1]-(knee[-1]-knee[0])*u
        return {'hip_roll': 0., 'hip_pitch': float(np.interp(s, knee, hip)),
                'knee_pitch': s, 'ankle_pitch': float(np.interp(s, knee, ankle))}

    def step(self, target, feedforward=None):
        d = self.d
        for k, a in self.act.items():
            ff = 0. if feedforward is None else feedforward.get(k[1], 0.)
            d.ctrl[a] = np.clip(ff-self.gain*(d.qpos[self.qadr[k]]-target[k[1]]), -self.vmax[k], self.vmax[k])
        d.ctrl[self.tail] = 0.
        mujoco.mj_step(self.m, d)

    def com_height_static(self, u):
        """Whole-body CoM height above the sole for the path pose u (FK only, body pitch as in the path)."""
        knee, hip, ankle, h = self.path
        return float(np.interp(knee[-1]-(knee[-1]-knee[0])*u, knee, h))

    def on_ground(self):
        return any(0 in (self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]) for c in self.d.contact[:self.d.ncon])

    def com_z(self):
        return float(self.d.subtree_com[self.base][2])

    def run(self, settle=1., lower=2.5, hold=.5, after=1.2, profile='ramp'):
        m, d = self.m, self.d
        for k, v in crouch(START['hip_pitch'], START['knee_pitch']).items():
            d.qpos[m.jnt_qposadr[m.joint(k).id]] = v
        place_on_floor(m, d)
        start = dict(START, hip_roll=0.)
        deep, top = self.pose_at(0.), self.pose_at(1.)
        while d.time < settle:
            self.step(start)
        t0, tilt_lower = d.time, 0.
        while d.time < t0+lower:  # cosine blend from the standing crouch to the deep balanced crouch
            u = .5*(1-math.cos(math.pi*min(1., (d.time-t0)/lower)))
            self.step({j: (1-u)*start[j]+u*deep[j] for j in start})
            tilt_lower = max(tilt_lower, tilt(d.qpos[3:7]))
        t0 = d.time
        while d.time < t0+hold:
            self.step(deep)
            tilt_lower = max(tilt_lower, tilt(d.qpos[3:7]))
        z0, tilt_pre = self.com_z(), tilt(d.qpos[3:7])
        hold_torque = {j: round(max(abs(float(d.actuator_force[self.act[(s, j)]])) for s in ('left', 'right')), 1) for j in LEG}
        # Push schedule: integrate u(t) so that the joint closest to its limit runs exactly at it.
        du, us, ts, t = 1e-3, [0.], [0.], 0.
        while us[-1] < 1.:
            a, b = self.pose_at(us[-1]), self.pose_at(min(1., us[-1]+du))
            rate = max(abs(b[j]-a[j])/du/min(self.vmax[(s, j)] for s in ('left', 'right')) for j in LEG)
            t += du*rate  # seconds needed for this du at the binding joint's limit
            us.append(min(1., us[-1]+du)); ts.append(t)
        T = ts[-1]
        # Kinematic bound: CoM vertical speed if the binding joint sits at its limit, along the path (no dynamics)
        heights = [self.com_height_static(u) for u in us]
        v_kin = [(heights[i+1]-heights[i])/(ts[i+1]-ts[i]) for i in range(len(us)-1)]
        v_kin_max, u_kin = max(zip(v_kin, us[1:]))
        if profile == 'ramp':  # limit-speed time sigma = tau^2/(4T): speed ramps 0 -> limit over 2T, stroke ends at 2T
            ts = [math.sqrt(4*T*sig) for sig in ts]
            T = ts[-1]
        t0 = d.time
        zmax, vz_last, tilt_push, tilt_apex = z0, 0., 0., 0.
        torque, speed = {j: 0. for j in LEG}, {j: 0. for j in LEG}
        airborne_since, apex_time, landed, flights = None, None, False, []  # flights: [liftoff, vz at liftoff, touchdown]
        while d.time < t0+T+after:
            s = d.time-t0
            u = float(np.interp(s, ts, us))
            target = start if landed else self.pose_at(u)  # after the push and touchdown: hold the standing crouch
            if s < T:
                nxt = self.pose_at(float(np.interp(s+1e-3, ts, us)))
                ff = {j: (nxt[j]-target[j])/1e-3 for j in LEG}
            else:
                ff = None
            self.step(target, ff)
            z, w = self.com_z(), tilt(d.qpos[3:7])
            ground = self.on_ground()
            if s < T:
                tilt_push = max(tilt_push, w)
                for (side, j), a in self.act.items():
                    torque[j] = max(torque[j], abs(float(d.actuator_force[a])))
                    speed[j] = max(speed[j], abs(float(d.qvel[self.vadr[(side, j)]])))
            if ground:
                mujoco.mj_subtreeVel(m, d)
                vz_last = float(d.subtree_linvel[self.base][2])
                if airborne_since is not None:
                    flights.append([airborne_since, vz_liftoff, d.time])
                    airborne_since, landed = None, landed or s >= T
            elif airborne_since is None:
                airborne_since, vz_liftoff = d.time, vz_last  # CoM vertical velocity at the last ground-contact step
            if z > zmax:
                zmax, apex_time = z, d.time
            if apex_time is None or d.time <= apex_time:
                tilt_apex = max(tilt_apex, w)
        if airborne_since is not None:
            flights.append([airborne_since, vz_liftoff, d.time])
        # The jump flight is the airborne period containing the CoM apex (a bang-bang start can lift the feet
        # for a few ms at push onset; that is not the jump).
        jump = next((f for f in flights if f[0] <= apex_time <= f[2]), None)
        flight = 0. if jump is None else jump[2]-jump[0]
        vz_take = None if jump is None else jump[1]
        return {'profile': profile, 'kv': self.kv, 'v_scale': self.vmax[('left', 'knee_pitch')]/3., 'effort_scale': self.fmax[('left', 'knee_pitch')]/120.,
                'deep_crouch': {j: round(deep[j], 3) for j in LEG[1:]}, 'top_pose': {j: round(top[j], 3) for j in LEG[1:]},
                'path_stroke_m': round(float(self.path[3][0]-self.path[3][-1]), 4) if self.path[3][0] > self.path[3][-1] else round(float(self.path[3][-1]-self.path[3][0]), 4),
                'tilt_before_push_rad': round(tilt_pre, 4), 'max_tilt_while_lowering_rad': round(tilt_lower, 4),
                'push_time_s': round(T, 3), 'kinematic_takeoff_vz_bound_mps': round(v_kin_max, 3),
                'kinematic_bound_at_u': round(u_kin, 2), 'com_rise_m': round(zmax-z0, 4), 'flight_s': round(flight, 3),
                'takeoff_vz_mps': None if vz_take is None else round(vz_take, 3),
                'ballistic_rise_from_vz_m': None if vz_take is None else round(max(0., vz_take)**2/(2*9.81), 4),
                'max_tilt_during_push_rad': round(tilt_push, 3),
                'peak_joint_speed_rad_s': {j: round(v, 2) for j, v in speed.items()},
                'static_hold_torque_nm': hold_torque,
                'max_tilt_until_apex_rad': round(tilt_apex, 3), 'final_tilt_rad': round(tilt(d.qpos[3:7]), 3),
                'peak_torque_nm': {j: round(v, 1) for j, v in torque.items()},
                'torque_limit_nm': {j: self.fmax[('left', j)] for j in LEG},
                'valid': bool(tilt_pre < .1 and tilt_apex < .3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scales', type=float, nargs='+', default=[1, 2, 4, 6, 8])
    p.add_argument('--effort', type=float, nargs='+', default=[1.])
    p.add_argument('--profile', default='bang', choices=['bang', 'ramp'])
    p.add_argument('--kv', type=float, default=30.)
    p.add_argument('--out')
    a = p.parse_args()
    rows = []
    for e in a.effort:
        for v in a.scales:
            r = Trial(v, e, a.kv).run(profile=a.profile)
            print(json.dumps({k: r[k] for k in ('profile', 'kv', 'v_scale', 'effort_scale', 'push_time_s', 'kinematic_takeoff_vz_bound_mps', 'com_rise_m', 'flight_s', 'takeoff_vz_mps',
                                                 'tilt_before_push_rad', 'max_tilt_until_apex_rad', 'final_tilt_rad', 'valid')}))
            rows.append(r)
    if a.out:
        Path(a.out).write_text(json.dumps({'limitation': 'Open-loop feedforward extension along a balanced path in MuJoCo; '
                                                         'an upper bound for this model, not a learned jump.',
                                           'results': rows}, indent=1)+'\n')


if __name__ == '__main__':
    main()
