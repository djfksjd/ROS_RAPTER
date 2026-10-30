"""Evidence 182 figure. edge_falls.png: falls (of 8) of the blind policies rev2_tr2 and rev2_tr4 walking off a drop or into
a raised ledge at 1/2/4 m/s, and the median peak ground force within 1 s after the edge (rev2_tr2). Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/182-drop-ledge-baseline'
R = [json.loads(l) for l in open(f'{E}/eval/edge.jsonl') if l.strip()]
H = {'drop': [.1, .2, .3, .4, .5, .6, .8, 1.], 'ledge': [.05, .1, .15, .2, .25, .3, .4]}
F = {(r['run'].split('_')[1], r['kind'], r['height'], r['cmd']): r for r in R}
fig, ax = plt.subplots(1, 5, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.4]))
for a, (p, k) in zip(ax, [('tr2', 'drop'), ('tr4', 'drop'), ('tr2', 'ledge'), ('tr4', 'ledge')]):
    M = np.array([[F[(p, k, h, c)]['falls'] for c in (1., 2., 4.)] for h in H[k]])
    a.imshow(M/8, cmap='Reds', vmin=0, vmax=1, aspect='auto')
    for i in range(len(H[k])):
        for j in range(3):
            a.text(j, i, M[i, j], ha='center', va='center', color='white' if M[i, j] > 5 else 'black', fontsize=9)
    a.set_xticks(range(3), ['1 m/s', '2 m/s', '4 m/s'])
    a.set_yticks(range(len(H[k])), [f'{"drop down" if k == "drop" else "ledge up"} {h*100:g} cm' for h in H[k]])
    a.set_title(f'rev2_{p}: falls of 8', fontsize=10)
for c, col in ((1., '#2ca02c'), (2., '#1f77b4'), (4., '#d62728')):
    g = [np.median([e['grf_max_bw'] for e in F[('tr2', 'drop', h, c)]['episodes'] if e['t_fall'] is None] or [np.nan]) for h in H['drop']]
    ax[4].plot([h*100 for h in H['drop']], g, 'o-', color=col, label=f'{c:g} m/s')
ax[4].set_xlabel('drop height (cm)'); ax[4].set_ylabel('peak ground force (body weights)')
ax[4].set_title('rev2_tr2 drop: peak ground force\n(median of no-fall runs, 1 s after edge, 50 Hz)', fontsize=9)
ax[4].legend(fontsize=8); ax[4].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{E}/media/edge_falls.png', dpi=100)
