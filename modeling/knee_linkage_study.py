"""Knee drive transmission study for the 8-axis hybrid leg (evidence 86 §16).

Recommended by Codex gpt-6.1-sol and UNI_AI gpt-6-sol (review_*_linkage.md): keep the motors at the pelvis and drive the
knee (and, through the 1:1 knee-ankle coupling, the ankle) with a linkage whose transmission ratio depends on posture,
then re-check the actuator budget with real torque-speed envelopes.

Model (sagittal, per leg, springs (c) with the ankle spring clutched to stance, spec_40kmh design stride):
- knee drive load   tau_kd = tau_knee - tau_ankle (1:1 coupling, ankle = c - knee), speed w_k
- pelvis motor coaxial with the hip: motor angle = phi(q_knee) + q_hip, so w_m = N(q) w_k + w_hip with N = dphi/dq,
  and by virtual work tau_m = tau_kd / N (x efficiency); the knee motor's reaction adds to the hip: tau_hipmotor = tau_hip - tau_m
  (the biarticular effect the ostrich uses).
- transmissions: 'belt N' (constant N), and a four-bar on the femur: crank a at the hip axis, coupler b, lever c on the
  shin at offset beta, femur 0.22 m; N(q) from the linkage, dead-point guard via the transmission angle (30-150 deg).
- motor envelope per class (peak torque, no-load speed, peak power): full torque up to half the no-load speed, then a
  linear drop to zero at the no-load speed (ASSUMED shape), |tau w| <= P, efficiency 0.85 when motoring, RMS <= 40 % peak.
Utilisation u = max over the stride of |tau| / available torque at that speed (u <= 1 fits); also RMS / (0.4 peak).
"""
import json
import sys
from math import pi
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_r02 as D  # noqa: E402
import spec_40kmh as S40  # noqa: E402

CLASSES = {c[0][0]: dict(tau=c[1], w0=c[2], P=c[3], kg=c[4]) for c in S40.CLASSES}
ETA = .85
FEMUR = D.Params().femur


def loads(mass=11.36, v=11.1, duty=.25, p=None):
    p = p or D.Params()
    g, S, Lo = S40.solve(p, S40.Gait(v=v, duty=duty))
    cases, _ = S40.spring_cases(p, g, S)
    n_st = Lo['J']['n_st']
    stance = np.arange(len(Lo['ang']['knee'])) < n_st

    def spring(j, q, only_stance=False):
        sp = cases['c_ankle_knee_hip'][j]['spring']; k, q0, mode = sp['k_nm_per_rad'], sp['q0_rad'], sp['mode']
        d = q - q0
        d = np.maximum(d, 0) if mode == 'uni+' else np.minimum(d, 0) if mode == 'uni-' else d
        return k*d*(stance if only_stance else 1.)
    s = mass/S['mass_kg']
    tk = (Lo['tau']['knee'] + spring('knee', Lo['ang']['knee']))*s
    ta = (Lo['tau']['ankle'] + spring('ankle', Lo['ang']['ankle'], only_stance=True))*s
    th = (Lo['tau']['hip'] + spring('hip', Lo['ang']['hip']))*s
    return dict(q_k=Lo['ang']['knee'], w_k=Lo['vel']['knee'], tau_kd=tk - ta, q_h=Lo['ang']['hip'], w_h=Lo['vel']['hip'], tau_h=th,
                stance=stance, dt=S40.DT)


def available(cls, w):
    c = CLASSES[cls]
    a = np.abs(w)
    env = np.where(a <= .5*c['w0'], c['tau'], c['tau']*np.clip((c['w0'] - a)/(.5*c['w0']), 0, None))
    return np.minimum(env, c['P']/np.maximum(a, 1e-3))


def motor_side(tau_out, w_out_rel, N, w_extra):
    """Motor torque and speed for an output torque through ratio N (motor rad per output rad) with efficiency."""
    w_m = N*w_out_rel + w_extra
    motoring = tau_out*w_out_rel >= 0
    tau_m = np.where(motoring, tau_out/(N*ETA), tau_out*ETA/N)
    return tau_m, w_m


def check(cls, tau_m, w_m):
    avail = available(cls, w_m)
    u = float(np.max(np.abs(tau_m)/np.maximum(avail, 1e-6)))
    rms = float(np.sqrt(np.mean(tau_m**2)))
    return dict(util=round(u, 2), rms_ratio=round(rms/(.4*CLASSES[cls]['tau']), 2), peak_nm=round(float(np.max(np.abs(tau_m))), 1),
                peak_w_rad_s=round(float(np.max(np.abs(w_m))), 1), fits=bool(u <= 1 and rms <= .4*CLASSES[cls]['tau']))


def fourbar_N(q, a, b, c, beta, branch):
    """Crank angle phi(q) (femur frame, crank at the hip axis) for knee angle q; returns phi, N, transmission angle."""
    K = np.array([0., -FEMUR])
    psi = q + beta
    Q = np.stack([K[0] + c*np.cos(psi), K[1] + c*np.sin(psi)], 1)
    d = np.linalg.norm(Q, axis=1)
    ok = (d < a + b) & (d > abs(a - b))
    cosA = np.clip((a*a + d*d - b*b)/(2*a*d), -1, 1)
    phi = np.arctan2(Q[:, 1], Q[:, 0]) + branch*np.arccos(cosA)
    phi = np.unwrap(phi)
    N = np.gradient(phi, q)
    P = np.stack([a*np.cos(phi), a*np.sin(phi)], 1)
    u1, u2 = (Q - P)/b, np.stack([np.cos(psi), np.sin(psi)], 1)
    trans = np.degrees(np.arccos(np.clip(np.abs(np.sum(u1*u2, 1)), -1, 1)))  # angle between coupler and lever
    return phi, N, trans, ok


def evaluate(L, knee_cls, hip_cls, N_of_q=None, N_const=None, mount='hip'):
    """mount 'hip': knee motor coaxial with the hip on the pelvis (its reaction loads the hip motor, motor speed adds
    the hip speed); 'thigh': knee motor stator on the femur next to the hip (no reaction on the hip motor, its mass rides
    on the femur close to the hip axis)."""
    q, w_k, tkd = L['q_k'], L['w_k'], L['tau_kd']
    N = np.full_like(q, N_const) if N_const is not None else N_of_q
    tau_m, w_m = motor_side(tkd, w_k, N, L['w_h'] if mount == 'hip' else 0.*L['w_h'])
    kr = check(knee_cls, tau_m, w_m)
    hr = check(hip_cls, L['tau_h'] - (tau_m*ETA if mount == 'hip' else 0.), L['w_h'])
    return kr, hr


def search(L, knee_cls='B', hip_cls='C', n=20000, seed=0, mount='hip'):
    rng = np.random.default_rng(seed)
    q = L['q_k']
    qq = np.linspace(q.min() - .35, q.max() + .35, 200)  # extra range for standing / folding postures
    best = None
    for _ in range(n):
        a, b, c = rng.uniform(.02, .12), rng.uniform(.12, .30), rng.uniform(.02, .12)
        beta, br = rng.uniform(-pi, pi), rng.choice([-1., 1.])
        phi, N, tr, ok = fourbar_N(qq, a, b, c, beta, br)
        if not ok.all() or tr.min() < 30. or np.any(np.abs(N) < .2) or np.any(np.sign(N) != np.sign(N[0])):
            continue
        Nq = np.interp(q, qq, N)
        kr, hr = evaluate(L, knee_cls, hip_cls, N_of_q=Nq, mount=mount)
        score = max(kr['util'], kr['rms_ratio'], hr['util'], hr['rms_ratio'])
        if best is None or score < best[0]:
            best = (score, dict(a=round(a, 3), b=round(b, 3), c=round(c, 3), beta_deg=round(float(np.degrees(beta))), branch=br,
                                N_range=[round(float(np.abs(N).min()), 2), round(float(np.abs(N).max()), 2)],
                                min_transmission_deg=round(float(tr.min()))), kr, hr)
    return best


def main():
    from dataclasses import replace
    legs = {'R-02 leg': D.Params(), 'distal-light leg': replace(D.Params(), tube_density=.35, foot_toes=.06)}
    out = {}
    for leg, p in legs.items():
        for mass in (11.36, 8.):
            L = loads(mass, p=p)
            rows = {}
            for mount in ('hip', 'thigh'):
                for hip_cls in ('B', 'C'):
                    best_const = min(((max(r[0]['util'], r[0]['rms_ratio']), N, r) for N, r in
                                      ((N, evaluate(L, 'B', hip_cls, N_const=N, mount=mount)) for N in np.linspace(.6, 3., 49))), key=lambda x: x[0])
                    fb = search(L, 'B', hip_cls, n=8000, mount=mount)
                    rows[f'knee motor on {mount}, hip {hip_cls}'] = dict(
                        best_constant=dict(N=round(float(best_const[1]), 2), knee=best_const[2][0], hip=best_const[2][1]),
                        four_bar=dict(geometry=fb[1], knee=fb[2], hip=fb[3]) if fb else None)
            out[f'{leg}, {mass} kg'] = rows
            print(f'== {leg}, {mass} kg (11.1 m/s design stride, springs (c), clutched ankle, 1:1 knee-ankle, knee class B)')
            for k, r in rows.items():
                bc, fbr = r['best_constant'], r['four_bar']
                print(f"   {k}: constant N={bc['N']} knee util {bc['knee']['util']} rms {bc['knee']['rms_ratio']} | hip util {bc['hip']['util']} rms {bc['hip']['rms_ratio']}"
                      + (f" || four-bar N {fbr['geometry']['N_range']} knee util {fbr['knee']['util']} rms {fbr['knee']['rms_ratio']} | hip util {fbr['hip']['util']} rms {fbr['hip']['rms_ratio']}" if fbr else ' || four-bar none'))
    Path(__file__).with_name('knee_linkage_study.json').write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
