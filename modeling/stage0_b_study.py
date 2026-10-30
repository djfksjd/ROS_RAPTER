"""Stage 0, option B (user decision 2026-10-31): can commercial actuators, with a free choice of external ratio, carry the
11.1 m/s design stride of R-02 once a hip yaw axis is added?  Calculation only; no policy, no running result.

Loads: spec_40kmh design stride (SLIP stance, springs (c), ankle spring clutched to stance, 1:1 knee-ankle coupling,
distal-light leg as in run_env real_mass) from knee_linkage_study.loads at 11.36 kg, scaled linearly with the total mass
(spec_40kmh model 5). The design stride is an idealised gait; learned policies have needed more (evidence 164-167), so
passing here is necessary, not sufficient.

Actuators: modules of docs/design/motor_candidates.json (RobStride verified against the vendor manuals there; the other
vendors are search summaries, marked). External stage: belt with a constant ratio G (motor rad per joint rad; G < 1 is a
speed-up belt) or, for the knee, a four-bar on the femur (ratio N(q), random search as knee_linkage_study). Efficiency
0.9 for any external stage, 1 for direct drive. Knee motor on the femur (no reaction on the hip motor).

Torque-speed envelope at the module output (no vendor curve is published):
  published   full peak torque up to 0.6 w0, then linear to zero at w0. Anchored on the RS04 reseller statement "stable torque
              up to 120 rpm" of 200 rpm no-load; the vendor rated point 40 N·m at 167 rpm lies under it. Used for the decision.
  assumed     full peak torque up to 0.5 w0 (run_env / knee_linkage_study assumption). Sensitivity.
  dc          min(i_peak, (V - kt w)/R) kt at 48 V, no battery sag (realism.py style). Sensitivity, optimistic.
Thermal: RMS module torque over the stride / vendor rated torque (rated on the vendor heat sink; mounted in a leg it can
be lower, evidence 167). Margin: every check is repeated with the loads x 1.2 (40 km/h spec section 6-2).

Hip roll: peak GRF x 0.10 m at 5 rad/s (spec_40kmh.full_budget). Hip yaw: ESTIMATE, no turning model solved (see
yaw_need). Mass: the chosen modules replace the class motors of the run_env real_mass model; total mass and loads are
iterated to a fixed point (structure, battery and electronics unchanged: an assumption).

Usage: .venv-sim/bin/python modeling/stage0_b_study.py [--v 11.1] [--json out.json]
"""
import argparse
import json
import sys
from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_r02 as D  # noqa: E402
import knee_linkage_study as K  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODS = json.loads((ROOT/'docs/design/motor_candidates.json').read_text())['modules']
ETA, MARGIN, V = .9, 1.2, 48.
REAL_MASS_TOTAL = 11.2912                      # RunEnv(arch8, real_mass, model_rev=2).mass, 2026-10-31
CLASS_MOTORS_KG = 2*(.5 + .96 + .96) + 2*.96   # run_env real_masses: hip roll A, hip pitch B, knee B per side, tail 2 x B
NON_ACTUATOR_KG = REAL_MASS_TOTAL - CLASS_MOTORS_KG   # torso payload, legs, ankle rods, tail structure (kept)
BELT_KG, FOURBAR_KG, YAW_BRACKET_KG = .10, .10, .05   # ASSUMED drive hardware per stage
ENVELOPES = ('manual', 'published', 'assumed', 'dc')
for _m in MODS.values():
    _m['w0_nominal'] = _m['w0']
USABLE = [m for m in MODS if MODS[m]['w0'] and MODS[m].get('tau_cont')]
VERIFIED = [m for m in USABLE if MODS[m].get('verified')]
# policy-calibrated loads: learned-policy / design-stride ratios at 5.72-6 m/s (docs/design/MOTOR_SELECTION.ko.md section 1,
# evidence 164 rev2_r3_ck7M; distal-light leg design stride at 6 m/s: hip RMS 7.3, knee drive peak 32.1 / RMS 15.8):
# hip RMS 18.2/7.3 = 2.5 (peak not measured: the same factor assumed), knee peak 63-78/32.1 -> 2.2, knee RMS (crank side)
# 16.3/15.8 -> 1.0. ASSUMED constant up to 11.1 m/s.
POLICY = dict(hip_peak=2.5, hip_rms=2.5, knee_peak=2.2, knee_rms=1.0)


def envelope(mod, w, kind):
    m = MODS[mod]
    tau, w0, a = m['tau_peak'], m['w0'], np.abs(w)
    if kind == 'manual':  # vendor 48 V T-N curve (motor_candidates.json tn_curve); its speed axis follows w0 (tolerance runs)
        c = m.get('tn_curve')
        if c is None:
            return None
        f = w0/m.get('w0_nominal', w0)
        return np.interp(a, np.asarray(c['rpm'])*np.pi/30*f, c['nm'], right=0.)
    if kind == 'dc':
        if None in (m.get('kt_out'), m.get('r_line'), m.get('i_peak')):
            return None
        return np.minimum(np.clip((V - m['kt_out']*a)/m['r_line'], 0., m['i_peak'])*m['kt_out'], tau)
    knee = .6 if kind == 'published' else .5
    return np.where(a <= knee*w0, tau, tau*np.clip((w0 - a)/((1 - knee)*w0), 0., None))


def through(tau_j, w_j, G, eta):
    """Module torque and speed for a joint torque and speed through ratio G; efficiency by the power direction."""
    motoring = tau_j*w_j >= 0
    return np.where(motoring, tau_j/(G*eta), tau_j*eta/G), G*w_j


def check(mod, tau_m, w_m, kind):
    env = envelope(mod, w_m, kind)
    if env is None:
        return None
    rms = float(np.sqrt(np.mean(tau_m**2)))
    return dict(util=round(float(np.max(np.abs(tau_m)/np.maximum(env, 1e-6))), 2), rms_over_rated=round(rms/MODS[mod]['tau_cont'], 2),
                peak_module_nm=round(float(np.max(np.abs(tau_m))), 1), peak_module_rad_s=round(float(np.max(np.abs(w_m))), 1))


def score(r):
    return max(r['util'], r['rms_over_rated']) if r else 9.


def best_constant(mod, tau, w, kind):
    best = None
    for G in np.round(np.arange(.4, 2.51, .05), 2):
        r = check(mod, *through(tau, w, G, 1. if G == 1. else ETA), kind)
        if r and (best is None or score(r) < score(best)):
            best = dict(drive=f'belt {G:g}' if G != 1. else 'direct', G=float(G), **r)
    return best


FOURBARS = None


def fourbar_table(q, n=6000, seed=0):
    """Candidate four-bars (crank at the hip axis, lever on the shin) valid over the stride range +-0.35 rad."""
    global FOURBARS
    if FOURBARS is None:
        rng, qq, FOURBARS = np.random.default_rng(seed), np.linspace(q.min() - .35, q.max() + .35, 200), []
        for _ in range(n):
            a, b, c = rng.uniform(.02, .12), rng.uniform(.12, .30), rng.uniform(.02, .12)
            beta, br = rng.uniform(-pi, pi), rng.choice([-1., 1.])
            _, N, tr, ok = K.fourbar_N(qq, a, b, c, beta, br)
            if not ok.all() or tr.min() < 30. or np.any(np.abs(N) < .2) or np.any(np.sign(N) != np.sign(N[0])):
                continue
            FOURBARS.append((dict(a=round(a, 3), b=round(b, 3), c=round(c, 3), beta_deg=round(float(np.degrees(beta))), branch=br,
                                  N_range=[round(float(np.abs(N).min()), 2), round(float(np.abs(N).max()), 2)]), np.abs(np.interp(q, qq, N))))
    return FOURBARS


def best_fourbar(mod, tau, w, q, kind):
    best = None
    for geo, Nq in fourbar_table(q):
        r = check(mod, *through(tau, w, Nq, ETA), kind)
        if r and (best is None or score(r) < score(best)):
            best = dict(drive='four-bar', geometry=geo, **r)
    return best


def yaw_need(M):
    """ESTIMATE. Body yaw inertia about the stance hip ~0.75 kg m^2 at 11.36 kg with the tail held (torso, sensor pod and the
    0.9 m tail with its 0.4 kg tip; the tail yaw axis can unload most of the tail). A 90 deg heading change in 0.5 s,
    bang-bang: 4*(pi/2)/0.5^2 = 25 rad/s^2 -> 19 N·m; with the foot friction couple and swing leg take 25 N·m at 8 rad/s,
    RMS 0.3 of peak. To be replaced by measured loads in the turning stage."""
    s = M/11.36
    return dict(tau=25.*s, w=8., rms=.3*25.*s)


def roll_need(M, grf_bw):
    F = grf_bw*M*9.81
    return dict(tau=F*.10, w=5., rms=.45*F*.10)


def fit_point(mod, need, kind):
    """Smallest-score constant ratio for a (peak torque, speed, RMS) requirement."""
    best = None
    for G in np.round(np.arange(.5, 2.51, .1), 2):
        eta = 1. if G == 1. else ETA
        env = envelope(mod, need['w']*G, kind)
        if env is None:
            return None
        u = need['tau']/(G*eta)/max(float(env), 1e-6)
        r = dict(drive=f'belt {G:g}' if G != 1. else 'direct', G=float(G), util=round(u, 2), rms_over_rated=round(need['rms']/(G*eta)/MODS[mod]['tau_cont'], 2))
        if best is None or score(r) < score(best):
            best = r
    return best


def lightest(cands, limit=1.):
    ok = [(MODS[m]['mass'], m, r) for m, r in cands.items() if r and score(r) <= limit]
    return min(ok, key=lambda x: x[0])[1:] if ok else (None, None)


def calibrate(tau, peak, rms):
    """Scale a load trace so that its peak grows by `peak` and its RMS by `rms` (rms <= peak): the part above the
    stride RMS level is stretched more. Identity for (1, 1)."""
    if peak == 1. and rms == 1.:
        return tau
    r0, p0 = float(np.sqrt(np.mean(tau**2))), float(np.max(np.abs(tau)))
    best = None
    for e in np.geomspace(.3, 60., 300):                  # tau' = c * sign(tau) |tau|^e, c from the peak
        t = np.sign(tau)*np.abs(tau/p0)**e*p0*peak
        err = abs(float(np.sqrt(np.mean(t**2)))/r0 - rms)
        if best is None or err < best[0]:
            best = (err, t)
    return best[1]


def select(L, grf_bw, M, kind, margin, mods, loads='design'):
    s = M/11.36*margin
    tau_h, tau_k = L['tau_h']*s, L['tau_kd']*s
    if loads == 'policy':
        tau_h = calibrate(tau_h, POLICY['hip_peak'], POLICY['hip_rms'])
        tau_k = calibrate(tau_k, POLICY['knee_peak'], POLICY['knee_rms'])
    pitch = [m for m in mods if MODS[m]['tau_peak'] >= 40]
    hip = {m: best_constant(m, tau_h, L['w_h'], kind) for m in pitch}
    knee = {}
    for m in pitch:
        c, f = best_constant(m, tau_k, L['w_k'], kind), best_fourbar(m, tau_k, L['w_k'], L['q_k'], kind)
        knee[m] = min((x for x in (c, f) if x), key=score, default=None)
    roll = {m: fit_point(m, roll_need(M, grf_bw) | {'tau': roll_need(M, grf_bw)['tau']*margin}, kind) for m in mods}
    yaw = {m: fit_point(m, yaw_need(M) | {'tau': yaw_need(M)['tau']*margin}, kind) for m in mods}
    return dict(hip_pitch=hip, knee=knee, hip_roll=roll, hip_yaw=yaw)


def drive_kg(axis, r):
    if r is None:
        return 0.
    if r['drive'] == 'four-bar':
        return FOURBAR_KG
    return (BELT_KG if r['drive'] != 'direct' else 0.) + (YAW_BRACKET_KG if axis == 'hip_yaw' else 0.)


def fixed_point(L, grf_bw, kind, margin, mods, loads, tail=('RS03', 'RS06')):
    M, picks = REAL_MASS_TOTAL, None
    for _ in range(12):
        sel = select(L, grf_bw, M, kind, margin, mods, loads)
        picks = {ax: lightest(c) for ax, c in sel.items()}
        if any(p[0] is None for p in picks.values()):
            return dict(mass_kg=round(M, 2), feasible=False, picks={ax: p[0] for ax, p in picks.items()},
                        best_scores={ax: {m: round(score(r), 2) for m, r in c.items() if r} for ax, c in sel.items()})
        act = 2*sum(MODS[m]['mass'] + drive_kg(ax, r) for ax, (m, r) in picks.items()) + sum(MODS[t]['mass'] for t in tail)
        M_new = NON_ACTUATOR_KG + act
        if abs(M_new - M) < .01:
            break
        M = M_new
    return dict(mass_kg=round(M, 2), actuator_kg=round(act, 2), actuator_fraction=round(act/M, 2), feasible=True,
                picks={ax: dict(module=m, **r) for ax, (m, r) in picks.items()}, tail=list(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v', type=float, default=11.1)
    ap.add_argument('--json')
    a = ap.parse_args()
    p = replace(D.Params(), tube_density=.35, foot_toes=.06)       # distal-light leg (run_env real_mass)
    L = K.loads(11.36, v=a.v, p=p)
    _, S, _ = K.S40.solve(p, K.S40.Gait(v=a.v))
    pk = lambda x: round(float(np.max(np.abs(x))), 1)
    out = dict(v_m_s=a.v, peak_grf_bw=S['peak_grf_bw'], non_actuator_kg=round(NON_ACTUATOR_KG, 2),
               loads_11p36kg=dict(hip_peak_nm=pk(L['tau_h']), hip_peak_rad_s=pk(L['w_h']), knee_drive_peak_nm=pk(L['tau_kd']),
                                  knee_peak_rad_s=pk(L['w_k'])), results={})
    cases = [(ld, 'verified', kind, mg) for ld in ('design', 'policy') for kind in ENVELOPES for mg in (1., MARGIN)]
    cases += [(ld, 'all', 'published', mg) for ld in ('design', 'policy') for mg in (1., MARGIN)]
    for ld, ms, kind, margin in cases:
            r = fixed_point(L, S['peak_grf_bw'], kind, margin, VERIFIED if ms == 'verified' else USABLE, ld)
            key = f'{ld} loads, {ms} modules, {kind} x{margin:g}'
            out['results'][key] = r
            print(f'{key}: feasible={r["feasible"]} M={r["mass_kg"]} act={r.get("actuator_fraction")}', json.dumps(r.get('picks')), flush=True)
            if not r['feasible']:
                print('   best scores', json.dumps(r['best_scores']), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()


# ------------------------------------------------------------------ the chosen option-B configuration
# hip pitch: 0.85 speed-up belt instead of direct drive, so the 11.1 m/s hip speed (20.6 rad/s) still fits when the module
# no-load speed is at the -10 % end of its tolerance (200 rpm +-10 %); hip roll 2.5 belt (1.5 exceeds the rated torque);
# hip yaw 1.5 belt (the yaw load is an estimate; margin kept for the turning stage). Knee four-bar: evidence 187 pick that
# keeps the rev 2 knee band (evidence 186 used a .115 b .143 c .113 beta 62, band ending at 2.08 rad)
CHOSEN = dict(hip_pitch=('RS04', .85), knee=('RS04', dict(a=.127, b=.176, c=.139, beta_deg=67.664, branch=1.)),
              hip_roll=('RS06', 2.5), hip_yaw=('RS06', 1.5))
CHOSEN_TAIL = ('RS03', 'RS06')


def chosen_mass():
    act = 0.
    for ax, (m, G) in CHOSEN.items():
        act += 2*(MODS[m]['mass'] + (FOURBAR_KG if isinstance(G, dict) else BELT_KG if G != 1. else 0.) + (YAW_BRACKET_KG if ax == 'hip_yaw' else 0.))
    act += sum(MODS[t]['mass'] for t in CHOSEN_TAIL)
    return NON_ACTUATOR_KG + act, act


def evaluate_chosen(L, grf_bw, w0_factor=1.):
    """Utilisation of CHOSEN at its own total mass for every load case, envelope and margin; w0_factor scales every
    module's no-load speed (vendor tolerance +-10 %)."""
    saved = {m: MODS[m]['w0'] for m in MODS}
    for m in MODS:
        if MODS[m]['w0']:
            MODS[m]['w0'] *= w0_factor
    try:
        return _evaluate_chosen(L, grf_bw)
    finally:
        for m, w in saved.items():
            MODS[m]['w0'] = w


def _evaluate_chosen(L, grf_bw):
    M, act = chosen_mass()
    q = L['q_k']
    out = dict(mass_kg=round(M, 2), actuator_kg=round(act, 2), actuator_fraction=round(act/M, 2), cases={})
    for ld in ('design', 'policy'):
        for kind in ENVELOPES:
            for margin in (1., MARGIN):
                s = M/11.36*margin
                th, tk = L['tau_h']*s, L['tau_kd']*s
                if ld == 'policy':
                    th, tk = calibrate(th, POLICY['hip_peak'], POLICY['hip_rms']), calibrate(tk, POLICY['knee_peak'], POLICY['knee_rms'])
                m, G = CHOSEN['hip_pitch']
                hip = check(m, *through(th, L['w_h'], G, 1. if G == 1. else ETA), kind)
                m, fb = CHOSEN['knee']
                qq = np.linspace(q.min() - .35, q.max() + .35, 200)
                _, N, _, _ = K.fourbar_N(qq, fb['a'], fb['b'], fb['c'], np.radians(fb['beta_deg']), fb['branch'])
                knee = check(m, *through(tk, L['w_k'], np.abs(np.interp(q, qq, N)), ETA), kind)
                pt = lambda ax, need: dict(zip(('util', 'rms_over_rated'), _point(CHOSEN[ax][0], need, CHOSEN[ax][1], kind)))
                rn, yn = roll_need(M, grf_bw), yaw_need(M)
                rn['tau'] *= margin; yn['tau'] *= margin
                out['cases'][f'{ld} loads, {kind} x{margin:g}'] = dict(hip_pitch=hip, knee=knee, hip_roll=pt('hip_roll', rn), hip_yaw=pt('hip_yaw', yn))
    return out


def _point(mod, need, G, kind):
    eta = 1. if G == 1. else ETA
    env = envelope(mod, need['w']*G, kind)
    if env is None:
        return (None, None)
    return (round(need['tau']/(G*eta)/max(float(env), 1e-6), 2), round(need['rms']/(G*eta)/MODS[mod]['tau_cont'], 2))
