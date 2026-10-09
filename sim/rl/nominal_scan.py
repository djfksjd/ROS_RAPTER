"""Nominal (action-offset) pose for a RunEnv variant by the rule that produced NOMINAL_T1 (pose scan 2026-09-29).

Rule, applied identically to every structure variant (arch8 / arch12 / hip_back): knee fixed at the NOMINAL_T1 knee
angle (1.50 rad, the spring rest angles assume it), toes flat (hip + knee + ankle = -0.5271, the foot mount), and the hip
angle that puts the whole-body CoM over the centre of the toe support (mean x of the foot/toe geoms touching the
ground) in the reset pose. For R-02 this static rule gives hip -0.346 against NOMINAL_T1 -0.375 (found 2026-09-29 as
the boundary between toppling forward and backward with the zero action). The dynamic boundary was not used: with
an ankle motor (arch12) the zero-action hold is actively supported and the boundary no longer marks the CoM over the
toes (scan 2026-10-01: arch12 boundary hip -0.83 with the CoM 0.16 m behind the toes). The zero-action hold time is
reported for information only.
Usage: .venv-sim/bin/python sim/rl/nominal_scan.py '{"arch8": true, "real_mass": true, "hip_back": 0.2, ...}'
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_env import NOMINAL_T1, RunEnv  # noqa: E402

FLAT = sum(NOMINAL_T1)  # -0.5271 (hip + knee + ankle with the toes flat)


def hold(kw, hip, seconds=3.):
    """Hold the zero action from the pose (hip, knee 1.50, flat toes); returns (direction, seconds until the fall).
    direction +1 = forward (nose down), -1 = backward, 0 = still standing at the end with < 5 deg pitch change."""
    knee = NOMINAL_T1[1]
    env = RunEnv(**{**kw, 'nominal': (hip, knee, FLAT-hip-knee)}, randomize=False, seed=0)
    env.reset()
    env.command = np.zeros(3); env.resample_steps = 0
    R0 = env.data.xmat[env.base].reshape(3, 3)[2, 0]
    act = np.zeros(env.action_space.shape)
    t = 0.
    for _ in range(int(seconds/.02)):
        _, _, term, _, _ = env.step(act)
        t += .02
        if term:
            break
    dz = env.data.xmat[env.base].reshape(3, 3)[2, 0]-R0
    if not term and abs(dz) < np.sin(np.radians(5)):
        return 0, t
    return (1 if dz < 0 else -1), t


def static(kw, hip):
    """Reset pose (hip, knee 1.50, flat toes): CoM x minus toe-support x, torso height, CoM x minus hip x, mass."""
    knee = NOMINAL_T1[1]
    env = RunEnv(**{**kw, 'nominal': (hip, knee, FLAT-hip-knee)}, randomize=False, seed=0)
    env.reset()
    m, d = env.model, env.data
    com = d.subtree_com[env.root]
    sup = [d.geom_xpos[g][0] for g in range(m.ngeom) if any(k in m.body(m.geom_bodyid[g]).name for k in ('foot', 'toe'))
           and d.geom_xpos[g][2] < .03]
    return (float(com[0]-np.mean(sup)), float(d.xpos[env.base][2]), float(com[0]-d.xpos[m.body('left_thigh_link').id][0]),
            float(m.body_mass.sum()))


def scan(kw, lo=-1.4, hi=0.2, tol=.002):
    f = lambda h: static(kw, h)[0]
    fa, fb = f(lo), f(hi)
    if np.sign(fa) == np.sign(fb):
        return dict(found=False, at_lo=round(fa, 3), at_hi=round(fb, 3))
    a, b = lo, hi
    while b-a > tol:
        m = .5*(a+b)
        if np.sign(f(m)) == np.sign(fa):
            a = m
        else:
            b = m
    hip = round(.5*(a+b), 4)
    off, h, c_hip, M = static(kw, hip)
    d, t = hold(kw, hip, seconds=6.)
    return dict(found=True, nominal=[hip, NOMINAL_T1[1], round(FLAT-hip-NOMINAL_T1[1], 4)], com_minus_support_m=round(off, 4),
                torso_height_m=round(h, 3), com_ahead_of_hip_m=round(c_hip, 3), mass_kg=round(M, 2),
                zero_action_hold_s=round(t, 2), fall_dir=d)


if __name__ == '__main__':
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps({'variant': kw, **scan(kw)}))
