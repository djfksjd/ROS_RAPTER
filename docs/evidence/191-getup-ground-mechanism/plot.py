"""Evidence 191 chart: lowest statically balanced crouch vs the knee fold limit (eval/fold_geom.jsonl).
Usage: .venv-sim/bin/python docs/evidence/191-getup-ground-mechanism/plot.py -> media/fold_clearance.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
r = [json.loads(l) for l in (OUT/'eval/fold_geom.jsonl').read_text().splitlines()]
k = [x['knee_max'] for x in r]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k, [100*x['belly_clear'] for x in r], 'o-', color='#123f5c', label='belly above ground (cm)')
ax.plot(k, [100*x['base_h'] for x in r], 's--', color='#9aa5b1', label='base height (cm)')
ax.axvline(2.265, color='#c2410c', lw=1); ax.text(2.275, 44, 'R-03 knee limit 2.26 rad\n(four-bar band)', color='#c2410c', fontsize=8)
ax.axhspan(0, 2, color='#fde68a', alpha=.6); ax.text(2.85, 3, 'lying flat', fontsize=8, ha='right')
ax.set_xlabel('knee fold limit (rad)'); ax.set_ylabel('cm')
ax.set_title('Lowest crouch with the whole-body CoM over the feet (torso level, ankle on the knee coupling)\n'
             'kinematics only: no self-collision, torque or four-bar check (MuJoCo R-03, not hardware)', fontsize=8.5)
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/'media/fold_clearance.png', dpi=130)
