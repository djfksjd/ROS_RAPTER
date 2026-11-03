"""Stage 0 rework (user decision 2026-11-03, evidence 191): a knee four-bar that also folds deep enough to tuck the legs
under the prone body. Evidence 191 (kinematics): the lowest crouch with the CoM over the feet leaves the belly 24.5 cm
up at the R-03 knee limit 2.26 rad, 11.7 cm at 2.6, 2.9 cm at 2.9.

Same rules as the evidence 187 pick (stage0_b_study): RS04 on the upper femur, four-bar (crank at the hip axis, lever on
the shin), efficiency 0.9, 11.1 m/s design stride at the chosen option-B mass, vendor 48 V T-N curve ('manual'), loads
x1 and x1.2, policy-calibrated loads, module no-load speed x1 and x0.9. The knee band is the run_env rule (both
transmission angles >= 30 deg, longest run, fourbar_band) and must cover [0.72, FOLD]. Score per candidate = worst of
(peak utilisation, RMS / rated) over design loads x1.2 and policy loads x1 at both speeds. Also reported: the ratio N at
the deepest fold (joint torque available there = N x 0.9 x module peak).
Usage: .venv-sim/bin/python modeling/knee_fold_study.py [--folds 2.6 2.7 2.8 2.9] [--n 40000] [--json out.json]
"""
import argparse
import json
import sys
from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'sim'/'rl'))
import design_r02 as D  # noqa: E402
import knee_linkage_study as K  # noqa: E402
import stage0_b_study as B  # noqa: E402
from run_env import fourbar_band  # noqa: E402

LO = .72  # rev 2 / R-03 band start (0.7159)


def knee_score(fb, L, M):
    q = L['q_k']
    qq = np.linspace(q.min()-.35, q.max()+.35, 200)
    _, N, _, _ = K.fourbar_N(qq, fb['a'], fb['b'], fb['c'], np.radians(fb['beta_deg']), fb['branch'])
    Nq = np.abs(np.interp(q, qq, N))
    worst, cases = 0., {}
    for w0 in (1., .9):
        saved = B.MODS['RS04']['w0']; B.MODS['RS04']['w0'] = saved*w0
        try:
            for ld, margin in (('design', B.MARGIN), ('policy', 1.)):
                tk = L['tau_kd']*M/11.36*margin
                if ld == 'policy':
                    tk = B.calibrate(tk, B.POLICY['knee_peak'], B.POLICY['knee_rms'])
                r = B.check('RS04', *B.through(tk, L['w_k'], Nq, B.ETA), 'manual')
                cases[f'{ld} x{margin:g} w0x{w0:g}'] = r
                worst = max(worst, B.score(r))
        finally:
            B.MODS['RS04']['w0'] = saved
    return worst, cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=float, nargs='+', default=[2.6, 2.7, 2.8, 2.9])
    ap.add_argument('--n', type=int, default=40000)
    ap.add_argument('--json')
    a = ap.parse_args()
    p = replace(D.Params(), tube_density=.35, foot_toes=.06)
    L = K.loads(11.36, v=11.1, p=p)
    M, _ = B.chosen_mass()
    ref = dict(B.CHOSEN['knee'][1])
    out = dict(mass_kg=round(M, 2), reference=dict(geometry=ref, band=[round(x, 3) for x in fourbar_band(ref, hi=3.2)],
                                                   score=round(knee_score(ref, L, M)[0], 2)), folds={})
    rng = np.random.default_rng(0)
    cands = []
    for _ in range(a.n):
        fb = dict(a=round(rng.uniform(.03, .20), 3), b=round(rng.uniform(.08, .35), 3), c=round(rng.uniform(.03, .20), 3),
                  beta_deg=round(float(np.degrees(rng.uniform(-pi, pi))), 1), branch=float(rng.choice([-1., 1.])))
        try:
            band = fourbar_band(fb, hi=3.2)
        except ValueError:
            continue
        if band[0] <= LO and band[1] >= min(a.folds):
            cands.append((fb, band))
    print(f'{len(cands)} four-bars cover [{LO}, {min(a.folds)}]', flush=True)
    scored = []
    for fb, band in cands:
        s, cases = knee_score(fb, L, M)
        q = np.array([band[1]])
        _, Nd, _, _ = K.fourbar_N(np.linspace(band[1]-.05, band[1], 5), fb['a'], fb['b'], fb['c'], np.radians(fb['beta_deg']), fb['branch'])
        scored.append(dict(geometry=fb, band=[round(band[0], 3), round(band[1], 3)], score=round(s, 2),
                           N_at_fold=round(float(abs(Nd[-1])), 2), cases=cases))
    for f in a.folds:
        ok = sorted((c for c in scored if c['band'][1] >= f), key=lambda c: c['score'])
        out['folds'][str(f)] = dict(n=len(ok), best=ok[:3])
        b = ok[0] if ok else None
        print(f'fold {f}: {len(ok)} candidates; best score {b and b["score"]} band {b and b["band"]} N_at_fold {b and b["N_at_fold"]} {b and json.dumps(b["geometry"])}', flush=True)
    print('reference (evidence 187)', json.dumps(out['reference']))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
