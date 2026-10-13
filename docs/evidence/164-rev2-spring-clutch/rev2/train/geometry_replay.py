"""Fixed-motion replay for the geometry check (no training): knee heat of the recorded gait under three engagement rules.
  stance(q0)     spring active in every stance (upper bound, engagement ignored)
  latch(q0)      the implemented rule: active in a stance only if, at a 20 ms control step of the preceding swing, the knee was
                 at or below q0 after having been above it
  touchdown      clutch that locks the spring's rest angle at the knee angle of the first loaded control step
Usage (project root): .venv-sim/bin/python geometry_replay.py <out.json> [run=rev2_n10]"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
out_path = Path(sys.argv[1]); run = sys.argv[2] if len(sys.argv) > 2 else 'rev2_n10'; sys.argv[1] = str(HERE)
import spring_id as S
S.RUN = Path.cwd()/'sim/rl/runs'/run
from run_env import LINK_ETA
K170 = 170*11.2912/11.36


def heat(sub, ctl, kr, k, rule, q0=None, skip=250):
    res = []; eng = []
    for j in range(2):
        q, req = sub[:, j], sub[:, 4+j]+sub[:, 6+j]; qc = ctl[:, j]; ld = ctl[:, 2+j].astype(bool); n = len(ld)
        on = np.zeros(n, bool); rest = np.full(n, 0. if q0 is None else q0); armed = latched = cur_on = False; cur = 0.; hit = tot = 0
        for i in range(1, n):
            if ld[i]:
                if not ld[i-1]:  # first loaded control step
                    cur = qc[i]; cur_on = rule in ('stance', 'touchdown') or latched
                    tot += i >= skip; hit += bool(cur_on) and i >= skip
                on[i] = cur_on
                if rule == 'touchdown':
                    rest[i] = cur
                armed = latched = False
            else:
                armed = armed or qc[i] > rest[i]
                latched = latched or (armed and qc[i] <= rest[i])
        onr, restr = np.repeat(on, 20), np.repeat(rest, 20)
        tm = req+np.where(onr, k*np.clip(q-restr, 0., None), 0.); N = np.interp(q, *kr); s = slice(skip*20, None)
        res.append(float(np.mean((tm[s]/(N[s]*LINK_ETA)/S.CONT)**2))); eng.append(hit/max(tot, 1))
    return dict(heat=max(res), engaged_frac=float(np.mean(eng)))


res = {}
for cmd in (2., 4., 7.):
    sub, ctl, kr = S.record(cmd, 96001); row = dict(none=heat(sub, ctl, kr, 0., 'stance', 1.)['heat'])
    row['stance'] = {f'{q0:g}': heat(sub, ctl, kr, K170, 'stance', q0)['heat'] for q0 in (.80, .90, .956, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30)}
    row['latch'] = {f'{q0:g}': heat(sub, ctl, kr, K170, 'latch', q0) for q0 in (.956, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30)}
    row['touchdown'] = {f'k{k}': heat(sub, ctl, kr, K170*k/170, 'touchdown') for k in (170, 340, 680)}
    # knee angle at the last unloaded and first loaded control step, extension rate there, stance flexion after touchdown
    ev = []
    for j in range(2):
        qc = ctl[:, j]; ld = ctl[:, 2+j].astype(bool)
        for i in np.flatnonzero(ld[1:] & ~ld[:-1])+1:
            if i < 250: continue
            e = i
            while e < len(ld) and ld[e]: e += 1
            ev.append((qc[i-1], qc[i], (qc[i]-qc[i-1])/.02, qc[i:e].max()-qc[i], qc[e-1]))
    ev = np.array(ev); row['touchdown_events'] = dict(n=len(ev), knee_last_swing=float(ev[:, 0].mean()), knee_first_stance=float(ev[:, 1].mean()), knee_rate=float(ev[:, 2].mean()),
                                                      flexion_after_touchdown=float(ev[:, 3].mean()), knee_liftoff=float(ev[:, 4].mean()))
    res[f'{cmd:g}'] = row
    print(cmd, 'none %.2f' % row['none']); print('  stance', {k: round(v, 2) for k, v in row['stance'].items()})
    print('  latch', {k: (round(v['heat'], 2), round(v['engaged_frac'], 2)) for k, v in row['latch'].items()})
    print('  touchdown', {k: (round(v['heat'], 2), round(v['engaged_frac'], 2)) for k, v in row['touchdown'].items()}); print('  ', {k: round(v, 3) for k, v in row['touchdown_events'].items()})
out_path.write_text(json.dumps(res, indent=1))
