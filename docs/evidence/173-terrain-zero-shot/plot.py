"""Evidence 173 figure: zero-shot falls (of 8) per ground for rev2_hr05 and rev2_hd40 at 1/2/4 m/s commands. Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/173-terrain-zero-shot'
T = {(r['run'][5:9], r['kind'], r['value'], r['cmd']): r['falls'] for r in map(json.loads, open(f'{E}/eval/terrain.jsonl'))}
G = [('flat', 0., 'flat'), *[('rough', a, f'rough +-{a*100:g} cm') for a in (.01, .02, .03, .04, .05, .06, .08)],
     *[('slope_up', g, f'uphill {g}°') for g in (5., 10., 15., 20., 25.)], *[('slope_down', g, f'downhill {g}°') for g in (5., 10., 15., 20., 25.)],
     *[('side_slope', g, f'side slope {g}°') for g in (5., 10., 15., 20.)], *[('stairs', r, f'stairs {r*100:g} cm') for r in (.02, .04, .06, .08, .1, .12)]]
cmds = (1., 2., 4.)
fig, ax = plt.subplots(1, 2, figsize=(9, 10), sharey=True)
for a, p in zip(ax, ('hr05', 'hd40')):
    M = np.array([[T[(p, k, v, c)] for c in cmds] for k, v, _ in G])
    a.imshow(M/8, cmap='Reds', vmin=0, vmax=1, aspect='auto')
    for i in range(len(G)):
        for j in range(3):
            a.text(j, i, M[i, j], ha='center', va='center', color='white' if M[i, j] > 5 else 'black', fontsize=8)
    a.set_xticks(range(3), [f'{c:g} m/s' for c in cmds]); a.set_title(f'rev2_{p}: falls of 8 (no retraining)')
ax[0].set_yticks(range(len(G)), [g[2] for g in G])
fig.tight_layout(); fig.savefig(f'{E}/media/terrain_falls.png', dpi=100)
