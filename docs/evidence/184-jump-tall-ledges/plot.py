"""Evidence 184 figure. edge_falls.png: falls (of 8) walking off a drop or into a raised ledge for rev2_tr4 (blind,
evidence 182-183), rev2_jp1 (scan, evidence 183) and rev2_jp2 (scan, more tall ledges), the median peak ground force of
no-fall drops at 2 m/s, and the hottest winding over 10 minutes on flat ground. Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/184-jump-tall-ledges'
D = {'tr4': 'docs/evidence/181-terrain-slope-stairs', 'jp1': 'docs/evidence/183-jump-scan', 'jp2': E}
J = lambda f: [json.loads(l) for l in open(f) if l.strip()]
R = J('docs/evidence/182-drop-ledge-baseline/eval/edge.jsonl')+J(f'{D["jp1"]}/eval/edge.jsonl')+J(f'{E}/eval/edge.jsonl')
H = {'drop': [.1, .2, .3, .4, .5, .6, .8, 1.], 'ledge': [.05, .1, .15, .2, .25, .3, .35, .4, .45]}
F = {(r['run'].split('_')[1], r['kind'], r['height'], r['cmd']): r for r in R}
C = {'tr4': '#ff7f0e', 'jp1': '#2ca02c', 'jp2': '#9467bd'}
fig, ax = plt.subplots(2, 4, figsize=(15, 9), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
for i, k in enumerate(H):
    for a, p in zip(ax[i], C):
        M = np.array([[F[(p, k, h, c)]['falls'] for c in (1., 2., 4.)] for h in H[k]])
        a.imshow(M/8, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        for y in range(len(H[k])):
            for x in range(3):
                a.text(x, y, M[y, x], ha='center', va='center', color='white' if M[y, x] > 5 else 'black', fontsize=9)
        a.set_xticks(range(3), ['1 m/s', '2 m/s', '4 m/s'])
        a.set_yticks(range(len(H[k])), [f'{"drop down" if k == "drop" else "ledge up"} {h*100:g} cm' for h in H[k]])
        a.set_title(f'rev2_{p}: falls of 8 ({M.sum()})', fontsize=10)
for p, col in C.items():
    g = [np.median([e['grf_max_bw'] for e in F[(p, 'drop', h, 2.)]['episodes'] if e['t_fall'] is None] or [np.nan]) for h in H['drop']]
    ax[0, 3].plot([h*100 for h in H['drop']], g, 'o-', color=col, label=f'rev2_{p}')
ax[0, 3].set_xlabel('drop height (cm)'); ax[0, 3].set_ylabel('peak ground force (body weights)')
ax[0, 3].set_title('drop at 2 m/s: peak ground force\n(median of no-fall runs, 1 s after edge, 50 Hz)', fontsize=9)
ax[0, 3].legend(fontsize=8); ax[0, 3].grid(alpha=.3)
L = {p: {(r['cmd'], r['mount']): np.nanmax([t['t_winding'] for t in r['trace']]) for r in J(f'{d}/eval/long_run.jsonl')}
     for p, d in D.items()}
for r in J(f'{E}/eval/long_run_2mps.jsonl'):  # 2 m/s, worst mount, measured here for all three
    L[r['run'].split('_')[1]][(r['cmd'], r['mount'])] = np.nanmax([t['t_winding'] for t in r['trace']])
cases = [(2., 'worst'), (4., 'plate'), (4., 'worst'), (7., 'worst')]
for j, (p, col) in enumerate(C.items()):
    ax[1, 3].bar(np.arange(4)+(j-1)*.27, [L[p].get(c, np.nan) for c in cases], .27, color=col, label=f'rev2_{p}')
ax[1, 3].set_xticks(range(4), [f'{v:g} m/s\n{m}' for v, m in cases]); ax[1, 3].set_ylabel('hottest winding (°C)')
ax[1, 3].set_title('flat, 10 min: hottest winding (knee)', fontsize=9); ax[1, 3].legend(fontsize=8); ax[1, 3].grid(alpha=.3, axis='y')
fig.tight_layout(); fig.savefig(f'{E}/media/edge_falls.png', dpi=100)
