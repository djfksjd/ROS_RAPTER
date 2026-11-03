"""Evidence 193 chart: the F1 shape screen as one table (rows = leg sum x segment ratio; columns = hip pitch score at
8 / 11.1 m/s with the stance term, knee four-bar score per fold target, prone belly clearance per fold).
Usage: .venv-sim/bin/python docs/evidence/193-shape-scan/plot.py -> media/shape_table.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
d = json.loads((OUT/'eval/scan.json').read_text())
FOLDS = ('2.26', '2.6', '2.9')
cols = ['hip 8 m/s', 'hip 11.1 m/s'] + [f'knee fold {f}' for f in FOLDS] + [f'belly cm, fold {f}' for f in FOLDS]
rows, vals = [], []
for r in d['rows']:
    rows.append(f"{r['sum_m']:.2f} m  {r['ratio'].split()[0]}  ({r['femur']:.2f}:{r['tibia']:.2f}:{r['meta']:.2f})")
    s = r['speeds']
    v = [s[k]['hip']['design+stance']['score'] if s[k] else np.nan for k in ('8.0', '11.1')]
    v += [(r.get('knee') or {}).get(f, {}) and r['knee'][f]['score'] for f in FOLDS]
    v += [100*r['belly_clear_m'][f] for f in FOLDS]
    vals.append([np.nan if x is None else float(x) for x in v])
V = np.array(vals)
fig, ax = plt.subplots(figsize=(13, 5.2))
C = np.zeros(V.shape+(3,))
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        x = V[i, j]
        if j < 5:   # motor score: <= 1 green, <= 1.1 amber, else red
            C[i, j] = (.80, .92, .80) if x <= 1. else (.98, .88, .66) if x <= 1.1 else (.97, .75, .75)
        else:       # belly clearance: smaller is better for getting the legs under the prone body
            C[i, j] = (.80, .92, .80) if x <= 5 else (.98, .88, .66) if x <= 12 else (.97, .75, .75)
ax.imshow(C, aspect='auto')
for i in range(V.shape[0]):
    for j in range(V.shape[1]):
        ax.text(j, i, f'{V[i, j]:.2f}' if j < 5 else f'{V[i, j]:.0f}', ha='center', va='center', fontsize=8.5)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8.5, rotation=20, ha='right')
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8.5)
ax.axvline(1.5, color='k', lw=.8); ax.axvline(4.5, color='k', lw=.8)
ax.set_title('F1 shape screen, R-03 option B (RS04 hip pitch best belt, RS04 knee four-bar), design stride x1.2, '
             'module speed x1 and x0.9, vendor T-N curve\nmotor score = max(peak / available torque, RMS / rated), '
             '<= 1 fits; belly = lowest level crouch with the feet under the CoM (pantograph approx.). Calculation, not running',
             fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT/'media/shape_table.png', dpi=130)
