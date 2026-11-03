"""Design-freeze plan F1 (docs/DESIGN_FREEZE_PLAN.ko.md): analytic screen of the R-03 leg shape. Calculation only; no
policy, no running result.

Candidates (fixed before the run): leg sum femur+tibia+meta 0.70 / 0.80 / 0.92 m x segment ratio current R-02 pick
1 : 1.73 : 1.45, ostrich 1 : 1.89 : 1.82, short-meta 1 : 1.5 : 1.0 x knee fold the four-bar band must reach 2.26 / 2.6 /
2.9 rad. Same rules as evidence 186/187/192: option-B modules (RS04 hip pitch on a 0.85 belt, RS04 knee four-bar on the
upper femur), vendor 48 V T-N curve ('manual'), efficiency 0.9, loads x1.2, module no-load speed x1 and x0.9, mass of the
chosen configuration (stage0_b_study.chosen_mass) plus the leg tube change.

Per candidate:
- design stride (spec_40kmh SLIP, springs (c)) at 8 and 11.1 m/s; None if the stride leaves the leg reach.
- hip pitch (RS04, best constant belt per candidate) utilisation two ways: the SLIP design stride (stance hip torque 0: the leg force passes through the hip) and
  with a stance hip term = leg force x its lever about the hip when the force line passes the whole-body CoM instead
  (zero pitching moment; CoM HIP_TO_COM from the hip, measured on R-03). The policy factor 2.5 of evidence 186 is not
  applied: this term is the physical part of it that a gait cannot avoid.
- knee: best of random four-bars (crank at the hip axis, femur = the candidate femur) whose band (both transmission
  angles >= 30 deg) covers the running knee range +-0.05 and the fold target. Score = worst of (peak utilisation,
  RMS / rated) over design x1.2 at w0 x1 and x0.9 (knee_fold_study.knee_score without the policy case, see above).
- prone belly clearance: lowest level-torso crouch with the toe COP under the CoM at the fold limit (pantograph leg),
  belly = hip height - BELLY_BELOW_HIP; the constant is fitted so the current shape reproduces evidence 191 at 2.26 rad
  (24.5 cm) and the other folds are reported next to evidence 191 as a check of this approximation.
Tail variants are NOT screened here (the SLIP model has no body pitch); they go to the dynamic check F2.
Usage: .venv-sim/bin/python modeling/shape_scan_r03.py [--n 20000] [--json docs/evidence/193-shape-scan/eval/scan.json]
"""
import argparse
import json
import sys
from dataclasses import replace
from math import pi
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'sim'/'rl'))
import design_r02 as D  # noqa: E402
import knee_linkage_study as K  # noqa: E402
import stage0_b_study as B  # noqa: E402
from run_env import fourbar_band  # noqa: E402

S40 = K.S40
RATIOS = {'current 1:1.73:1.45': (1., 1.73, 1.45), 'ostrich 1:1.89:1.82': (1., 1.89, 1.82), 'short-meta 1:1.5:1.0': (1., 1.5, 1.)}
SUMS = (.70, .80, .92)
FOLDS = (2.26, 2.6, 2.9)
SPEEDS = (8., 11.1)
HIP_TO_COM = np.array([-.057, -.05])   # R-03 at the stand pose: whole-body CoM 5.7 cm behind, 5 cm below the hip axis
EVIDENCE_191 = {2.26: .245, 2.6: .117, 2.9: .029}   # fold_geom.py on the MuJoCo R-03 model (no self-collision)
LEG0 = replace(D.Params(), tube_density=.35, foot_toes=.06)   # distal-light leg (run_env real_mass), as stage0_b_study


def shape(total, ratio):
    s = total/sum(ratio)
    return replace(LEG0, femur=round(s*ratio[0], 3), tibia=round(s*ratio[1], 3), meta=round(s*ratio[2], 3))


def loads(p, v):
    """knee_linkage_study.loads at 11.36 kg plus the stance leg force and centre of pressure (relative to the hip)."""
    g, S, Lo = S40.solve(p, S40.Gait(v=v))
    L = K.loads(11.36, v=v, p=p)
    J = Lo['J']
    n_st = J['n_st']
    F, cop = J['F'][:n_st], J['cop'][:n_st]
    # leg force along COP -> CoM (zero pitch moment); hip torque = moment of that force about the hip
    r_c = HIP_TO_COM - cop                       # COP -> CoM
    n = r_c/np.linalg.norm(r_c, axis=1, keepdims=True)
    f = n*F[:, None]
    r_h = -cop                                    # COP -> hip
    m_hip = r_h[:, 0]*f[:, 1] - r_h[:, 1]*f[:, 0] - (r_c[:, 0]*f[:, 1] - r_c[:, 1]*f[:, 0])
    th_st = np.zeros_like(L['tau_h'])
    th_st[:n_st] = m_hip                          # sign: hip motor holds it; magnitude is what matters for the check
    L['tau_h_stance'] = th_st
    L['stride'] = dict(stride_hz=S['stride_hz'], contact_s=S['contact_s'], peak_grf_bw=S['peak_grf_bw'],
                       hip_height_td_m=S['hip_height_td_m'], reach_frac=round(g.reach_frac, 2))
    return L


def hip_scores(L, M):
    """Best hip belt per candidate (0.4-2.5, as stage0_b_study.best_constant): worst over w0 x1, x0.9 of max(peak
    utilisation, RMS / rated), loads x1.2; >9 means the hip speed exceeds the module at every belt."""
    m = B.CHOSEN['hip_pitch'][0]
    out = {}
    for name, extra in (('design', 0.), ('design+stance', 1.)):
        th = (L['tau_h'] + extra*L['tau_h_stance'])*M/11.36*B.MARGIN
        best = (99., None)
        for G in np.round(np.arange(.4, 2.51, .05), 2):
            worst = 0.
            for w0 in (1., .9):
                saved = B.MODS[m]['w0']; B.MODS[m]['w0'] = saved*w0
                try:
                    worst = max(worst, B.score(B.check(m, *B.through(th, L['w_h'], G, 1. if G == 1. else B.ETA), 'manual')))
                finally:
                    B.MODS[m]['w0'] = saved
            best = min(best, (round(min(worst, 99.), 2), float(G)))
        out[name] = dict(score=best[0], belt=best[1])
    return out


def knee_score(fb, L, M):
    q = L['q_k']
    qq = np.linspace(q.min()-.35, q.max()+.35, 200)
    _, N, _, _ = K.fourbar_N(qq, fb['a'], fb['b'], fb['c'], np.radians(fb['beta_deg']), fb['branch'])
    Nq = np.abs(np.interp(q, qq, N))
    worst = 0.
    for w0 in (1., .9):
        saved = B.MODS['RS04']['w0']; B.MODS['RS04']['w0'] = saved*w0
        try:
            r = B.check('RS04', *B.through(L['tau_kd']*M/11.36*B.MARGIN, L['w_k'], Nq, B.ETA), 'manual')
            worst = max(worst, B.score(r))
        finally:
            B.MODS['RS04']['w0'] = saved
    return worst


def four_bars(femur, n, seed=0):
    K.FEMUR = femur                               # the four-bar spans the candidate femur (fourbar_N, fourbar_band)
    rng, out = np.random.default_rng(seed), []
    if abs(femur - LEG0.femur) < 1e-9:              # the R-03 four-bar (evidence 187) competes for the current femur
        ref = dict(B.CHOSEN['knee'][1])
        out.append((ref, fourbar_band(ref, hi=3.2)))
    for _ in range(n):
        fb = dict(a=round(rng.uniform(.03, .20), 3), b=round(rng.uniform(.08, .40), 3), c=round(rng.uniform(.03, .20), 3),
                  beta_deg=round(float(np.degrees(rng.uniform(-pi, pi))), 1), branch=float(rng.choice([-1., 1.])))
        try:
            out.append((fb, fourbar_band(fb, hi=3.2)))
        except ValueError:
            pass
    return out


def belly_hip_height(p, fold):
    """Lowest hip height with the toe COP under the CoM at knee = fold (pantograph: hip->MTP = (femur+meta, tibia))."""
    L1, L2 = p.femur + p.meta, p.tibia
    d = np.sqrt(L1*L1 + L2*L2 - 2*L1*L2*np.cos(pi - fold))
    dx = HIP_TO_COM[0] - p.cop_frac*p.toe3        # MTP sits a COP offset behind the CoM
    return float(np.sqrt(max(d*d - dx*dx, 0.)))


def run(task):
    total, rname, n = task
    p = shape(total, RATIOS[rname])
    M = B.chosen_mass()[0] + 2*LEG0.tube_density*(D.leg_len(p) - D.leg_len(LEG0))
    row = dict(sum_m=total, ratio=rname, femur=p.femur, tibia=p.tibia, meta=p.meta, mass_kg=round(M, 2), speeds={})
    Ls = {}
    for v in SPEEDS:
        try:
            Ls[v] = loads(p, v)
        except RuntimeError:
            row['speeds'][str(v)] = None
            continue
        L = Ls[v]
        row['speeds'][str(v)] = dict(**L['stride'], hip=hip_scores(L, M),
                                     knee_run_range=[round(float(L['q_k'].min()), 2), round(float(L['q_k'].max()), 2)])
    row['hip_height_fold'] = {str(f): round(belly_hip_height(p, f), 3) for f in FOLDS}
    if 11.1 not in Ls:
        return row
    L = Ls[11.1]
    lo_need = min(float(L['q_k'].min()) - .05, .72)
    bars = four_bars(p.femur, n)
    row['knee'] = {}
    for f in FOLDS:
        hi_need = max(f, float(L['q_k'].max()) + .05)
        ok = [(fb, band) for fb, band in bars if band[0] <= lo_need and band[1] >= hi_need]
        best = min(((knee_score(fb, L, M), fb, band) for fb, band in ok), key=lambda x: x[0], default=None)
        row['knee'][str(f)] = None if best is None else dict(score=round(best[0], 2), geometry=best[1],
                                                             band=[round(best[2][0], 3), round(best[2][1], 3)], n=len(ok))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--json')
    a = ap.parse_args()
    tasks = [(s, r, a.n) for s in SUMS for r in RATIOS]
    with Pool(min(9, len(tasks))) as pool:
        rows = pool.map(run, tasks)
    cur = next(r for r in rows if r['sum_m'] == .92 and r['ratio'].startswith('current'))
    belly0 = cur['hip_height_fold']['2.26'] - EVIDENCE_191[2.26]
    for r in rows:
        r['belly_clear_m'] = {f: round(h - belly0, 3) for f, h in r['hip_height_fold'].items()}
    out = dict(belly_below_hip_fitted_m=round(belly0, 3), evidence_191_check={str(k): v for k, v in EVIDENCE_191.items()},
               hip_to_com=HIP_TO_COM.tolist(), rows=rows)
    for r in rows:
        s = r['speeds']
        hs = lambda v: s[v] and {k: (x['score'], x['belt']) for k, x in s[v]['hip'].items()}
        print(f"{r['sum_m']:.2f} {r['ratio']:<22} M {r['mass_kg']}  8: {hs('8.0')}  11.1: {hs('11.1')}"
              f"  knee {json.dumps({k: v and v['score'] for k, v in (r.get('knee') or {}).items()})}  belly {r['belly_clear_m']}", flush=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
