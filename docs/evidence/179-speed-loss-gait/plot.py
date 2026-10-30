"""Evidence 179 figure: gait metrics vs command on flat stiff ground (nominal, 8 x 20 s) for rev2_ct5 (flat-trained)
and the terrain-trained rev2_tr1 / tr2 / tr3. Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
E = 'docs/evidence/179-speed-loss-gait'
R = [json.loads(l) for l in open(f'{E}/eval/gait.jsonl')]
C = {'ct5': '#d62728', 'tr1': '#2ca02c', 'tr2': '#1f77b4', 'tr3': '#9467bd'}
K = [('vx', 'speed (m/s)'), ('stride', 'stride per touchdown (m)'), ('step_hz', 'touchdowns per second'),
     ('flight', 'flight fraction'), ('clear', 'swing foot clearance (m)'), ('power', 'mechanical power (W)')]
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for a, (k, lab) in zip(ax.flat, K):
    for p, c in C.items():
        rs = sorted((r for r in R if r['run'] == f'rev2_{p}_st_r01'), key=lambda r: r['cmd'])
        a.plot([r['cmd'] for r in rs], [r[k] for r in rs], 'o-', color=c, label=f'rev2_{p}')
    a.set_xlabel('command (m/s)'); a.set_ylabel(lab); a.grid(alpha=.3)
ax[0, 0].plot([2, 7], [2, 7], 'k:', lw=1); ax[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{E}/media/gait_compare.png', dpi=110)
