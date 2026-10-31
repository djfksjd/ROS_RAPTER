"""Evidence 188 figure from eval/paths.json and eval/odom_sensitivity.jsonl (stand_probe.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).resolve().parent
paths = json.loads((D/'eval/paths.json').read_text())
odom = [json.loads(l) for l in (D/'eval/odom_sensitivity.jsonl').read_text().splitlines()]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for run, c, lab in (('r3_s1lp2', '#c0504d', 'r3_s1lp2 (no position input)'), ('r3_s1ho', '#2f6db5', 'r3_s1ho (odometry input)')):
    for i, p in enumerate(paths[run]):
        p = np.array(p)
        ax[0].plot(p[:, 0], p[:, 1], color=c, lw=.8, alpha=.8, label=lab if i == 0 else None)
        ax[1].plot(np.arange(len(p))*.02, np.linalg.norm(p, axis=1), color=c, lw=.8, alpha=.8, label=lab if i == 0 else None)
ax[0].add_patch(plt.Circle((0, 0), .5, fill=False, ls='--', color='k'))
ax[0].set_aspect('equal'); ax[0].set_xlabel('x (m)'); ax[0].set_ylabel('y (m)')
ax[0].set_title('Base path, 30 s zero command, 8 gate seeds'); ax[0].legend(fontsize=8)
ax[1].axhline(.5, ls='--', color='k', lw=1); ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('distance from start (m)')
ax[1].set_title('Gate: max distance <= 0.5 m'); ax[1].legend(fontsize=8)
names = [r['cond'] for r in odom]
ax[2].barh(names, [r['drift_max_m'] for r in odom], color='#2f6db5')
ax[2].axvline(.5, ls='--', color='k', lw=1); ax[2].invert_yaxis()
ax[2].set_xlabel('max drift over 8 seeds (m)'); ax[2].set_title('r3_s1ho, odometry error at test\n(noise m, walk m/sqrt s)')
fig.tight_layout()
fig.savefig(D/'media/stand_drift.png', dpi=110)
