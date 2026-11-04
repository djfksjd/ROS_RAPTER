"""Opt-in jump request and 1 ms measurements. Stage 5a fixed by supervisor 2026-11-04.

No physics writes, no negative rewards. The disabled path never samples or consumes RNG.
Standing follows the get-up/stand state: tilt < .3, height > .8 nominal standing height,
body speed < .5 m/s, each foot load > .15 body weight, continuously for 1 s.
"""
from collections import deque
import math
import mujoco
import numpy as np

FLIGHT_S = .10
RISE_M = .05
LANDING_M = .30
HOLD_S = 1.
READY_S = .10  # standing-start guard only; the required post-landing hold remains 1 s.
DT = .001
SECONDS = 12.
NOMINAL_SEEDS = range(9400, 9408)
DR_SEEDS = range(9500, 9524)
QUICK_NOMINAL_SEEDS = range(9300, 9302)
QUICK_DR_SEEDS = range(9310, 9314)


def physics_sample(env):
    d, m = env.data, env.model
    counts, force = np.zeros(2, int), np.zeros(2)
    hit = False
    for i in range(d.ncon):
        c = d.contact[i]
        other = c.geom2 if c.geom1 in env.ground else c.geom1 if c.geom2 in env.ground else None
        if other is None:
            continue
        side = next((j for j, s in enumerate(('left', 'right')) if other in env.foot_geoms[s]), None)
        if side is None:
            hit = True
        else:
            f = np.zeros(6); mujoco.mj_contactForce(m, d, i, f)
            counts[side] += 1; force[side] += abs(f[0])
    rz = env.rz
    return dict(t=float(d.time), xyz=d.xpos[env.base].copy(), velocity=d.qvel[:3].copy(),
        tilt=float(np.arccos(np.clip(d.xmat[env.base].reshape(3, 3)[2, 2], -1, 1))),
        feet_n=counts, feet_force_n=force, body_hit=hit, ground_z=env.ground_z(),
        torque_nm=d.actuator_force[env.act].copy(),
        current_a=rz.current.copy() if rz is not None and rz.on['motor'] else np.zeros(0),
        current_limit_a=rz.i_lim.copy() if rz is not None and rz.on['motor'] else np.zeros(0))


def standing(s, h_stand, weight):
    return (not s['body_hit'] and s['tilt'] < .3 and s['xyz'][2]-s['ground_z'] > .8*h_stand
            and np.linalg.norm(s['velocity']) < .5 and np.all(s['feet_force_n'] > .15*weight))


class JumpTask:
    def __init__(self, config):
        unknown = set(config)-{'auto_request_s', 'flight_bonus', 'success_bonus'}
        if unknown:
            raise ValueError(f'unknown jump options: {sorted(unknown)}')
        self.auto_request_s = config.get('auto_request_s', 2.)
        self.flight_bonus = float(config.get('flight_bonus', 10.))
        self.success_bonus = float(config.get('success_bonus', 30.))
        if any(not math.isfinite(v) or v < 0 for v in (self.flight_bonus, self.success_bonus)):
            raise ValueError('jump bonuses must be finite and nonnegative')
        if self.auto_request_s is not None and (not math.isfinite(self.auto_request_s) or self.auto_request_s < 0):
            raise ValueError('invalid jump request time')
        self.reset()

    def reset(self):
        self.pending = False; self.request_time = None; self.standing_height = None
        self.ready_heights = deque(maxlen=round(READY_S/DT)); self.air = None; self.best = None; self.landed = None
        self.hold_samples = 0; self.max_hold_samples = 0; self.success_time = None; self.fall_time = None
        self.progress = 0.; self.bonus_due = 0.; self.samples = 0
        self.impact_started = False
        self.impact_peak_n = 0.; self.peak_current_a = 0.; self.peak_current_ratio = 0.; self.peak_torque_nm = 0.
        self.capture = False; self.trace = []; self.last_sample = None

    @property
    def command(self):
        return float(self.request_time is not None and self.success_time is None and self.fall_time is None)

    def request(self):
        self.pending = True

    def _event(self, s, landed):
        return dict(flight_s=self.air['samples']*DT, apex_rise_m=self.air['apex']-self.standing_height,
                    takeoff_s=self.air['t'], takeoff_xy=self.air['xy'].tolist(),
                    touchdown_s=s['t'] if landed else None,
                    landing_error_m=float(np.linalg.norm(s['xyz'][:2]-self.air['xy'])) if landed else None)

    def sample(self, s, h_stand, weight):
        self.samples += 1; self.last_sample = s
        if self.capture:
            self.trace.append(s)
        finite = all(np.isfinite(s[k]).all() for k in ('xyz', 'velocity', 'feet_force_n', 'torque_nm', 'current_a')) and math.isfinite(s['tilt'])
        fallen = not finite or s['body_hit'] or s['tilt'] > .8 or s['xyz'][2]-s['ground_z'] < .5*(h_stand-.03) or np.abs(s['velocity']).max() > 30
        if fallen and self.fall_time is None:
            self.fall_time = s['t']
        self.peak_torque_nm = max(self.peak_torque_nm, float(np.abs(s['torque_nm']).max(initial=0)))
        self.peak_current_a = max(self.peak_current_a, float(s['current_a'].max(initial=0)))
        ratio = s['current_a']/np.maximum(s['current_limit_a'], 1e-12)
        self.peak_current_ratio = max(self.peak_current_ratio, float(ratio.max(initial=0)))
        if self.air is not None and np.any(s['feet_n'] > 0):
            self.impact_started = True
        if self.impact_started:
            self.impact_peak_n = max(self.impact_peak_n, float(np.sum(s['feet_force_n'])))
        stable = standing(s, h_stand, weight)
        if self.fall_time is not None:
            return
        if self.request_time is None:
            if stable:
                self.ready_heights.append(float(s['xyz'][2]))
            else:
                self.ready_heights.clear()
            if self.auto_request_s is not None and s['t'] >= self.auto_request_s:
                self.pending = True
            if self.pending and len(self.ready_heights) == self.ready_heights.maxlen:
                self.request_time = s['t']; self.standing_height = float(np.mean(self.ready_heights))
            return
        if self.success_time is not None:
            return
        off = bool(np.all(s['feet_n'] == 0))
        if self.landed is None:
            if off:
                if self.air is None:
                    self.air = dict(t=s['t'], xy=s['xyz'][:2].copy(), apex=float(s['xyz'][2]), samples=0)
                self.air['samples'] += 1; self.air['apex'] = max(self.air['apex'], float(s['xyz'][2]))
                event = self._event(s, False)
                quality = min(event['flight_s']/FLIGHT_S, 1.)*np.clip(event['apex_rise_m']/RISE_M, 0., 1.)
                gain = max(0., float(quality)-self.progress)
                self.bonus_due += self.flight_bonus*gain; self.progress += gain
                if self.best is None or (event['flight_s'] >= self.best['flight_s'] and event['apex_rise_m'] >= self.best['apex_rise_m']):
                    self.best = event
            elif self.air is not None:
                self.impact_started = True
                self.impact_peak_n = max(self.impact_peak_n, float(np.sum(s['feet_force_n'])))
                event = self._event(s, True)
                if event['flight_s'] >= FLIGHT_S and event['apex_rise_m'] >= RISE_M:
                    self.best = event
                    if event['landing_error_m'] <= LANDING_M:
                        self.landed = event
                self.air = None
        if self.landed is not None:
            self.impact_peak_n = max(self.impact_peak_n, float(np.sum(s['feet_force_n'])))
            self.hold_samples = self.hold_samples+1 if stable else 0
            self.max_hold_samples = max(self.max_hold_samples, self.hold_samples)
            if self.hold_samples >= round(HOLD_S/DT):
                self.success_time = s['t']; self.bonus_due += self.success_bonus

    def consume_bonus(self):
        value = self.bonus_due; self.bonus_due = 0.
        return value

    def result(self):
        event = self.landed or self.best or {}
        return dict(request_time_s=self.request_time, standing_height_m=self.standing_height,
                    flight_s=event.get('flight_s', 0.), apex_rise_m=event.get('apex_rise_m', 0.),
                    landing_error_m=event.get('landing_error_m'), touchdown_s=event.get('touchdown_s'),
                    standing_hold_s=self.max_hold_samples*DT, success_time_s=self.success_time,
                    t_fall=self.fall_time, progress=self.progress, load_samples=self.samples,
                    landing_impact_peak_n=self.impact_peak_n, current_peak_a=self.peak_current_a,
                    current_over_limit_peak=self.peak_current_ratio, torque_peak_nm=self.peak_torque_nm)
