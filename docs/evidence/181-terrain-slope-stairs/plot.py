"""Evidence 181 figures. terrain_falls.png: terrain falls (of 8) on stiff ground for rev2_tr2 (evidence 176) and rev2_tr4
(slopes and stairs weighted x2), and their difference. flat_compare.png: randomized flat-ground falls per speed and
nominal speed for rev2_ct5, rev2_tr2 and rev2_tr4. Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D = {'ct5': 'docs/evidence/174-contact-stiffness', 'tr2': 'docs/evidence/176-terrain-stage2',
     'tr4': 'docs/evidence/181-terrain-slope-stairs'}
E = D['tr4']
J = lambda f: [json.loads(l) for l in open(f) if l.strip()]
cmds = [2., 4., 5.5, 7.]
C = {'ct5': '#d62728', 'tr2': '#1f77b4', 'tr4': '#ff7f0e'}
T = {(r['run'].split('_')[1], r['kind'], r['value'], r['cmd']): r['falls'] for d in D.values() for r in J(f'{d}/eval/terrain.jsonl')}
G = [('flat', 0., 'flat'), *[('rough', a, f'rough +-{a*100:g} cm') for a in (.01, .02, .03, .04, .05, .06, .08)],
     *[('slope_up', g, f'uphill {g:g}°') for g in (5., 10., 15., 20., 25.)], *[('slope_down', g, f'downhill {g:g}°') for g in (5., 10., 15., 20., 25.)],
     *[('side_slope', g, f'side slope {g:g}°') for g in (5., 10., 15., 20.)], *[('stairs', r, f'stairs {r*100:g} cm') for r in (.02, .04, .06, .08, .1, .12)]]
M = {p: np.array([[T[(p, k, v, c)] for c in (1., 2., 4.)] for k, v, _ in G]) for p in ('tr2', 'tr4')}
fig, ax = plt.subplots(1, 3, figsize=(12, 10), sharey=True)
for a, (title, X, cmap, lo) in zip(ax, [(f'rev2_tr2: falls of 8 ({M["tr2"].sum()}/672)', M['tr2'], 'Reds', 0),
                                        (f'rev2_tr4: falls of 8 ({M["tr4"].sum()}/672)', M['tr4'], 'Reds', 0),
                                        ('tr4 - tr2 (blue = fewer falls)', M['tr4']-M['tr2'], 'coolwarm', -8)]):
    a.imshow(X/8, cmap=cmap, vmin=lo/8, vmax=1, aspect='auto')
    for i in range(len(G)):
        for j in range(3):
            a.text(j, i, f'{X[i, j]:+d}' if lo else X[i, j], ha='center', va='center',
                   color='white' if abs(X[i, j]) > 5 else 'black', fontsize=8)
    a.set_xticks(range(3), ['1 m/s', '2 m/s', '4 m/s']); a.set_title(title, fontsize=10)
ax[0].set_yticks(range(len(G)), [g[2] for g in G])
fig.tight_layout(); fig.savefig(f'{E}/media/terrain_falls.png', dpi=100)
A = {(r['run'], r['cmd']): r['falls'] for d in D.values() for r in J(f'{d}/eval/ablate.jsonl')}
S = {p: {r['cmd']: r['vx'] for r in J(f'{d}/eval/' + ('probe_nominal_ct5.jsonl' if p == 'ct5' else 'probe_nominal.jsonl'))}
     for p, d in D.items()}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, p in enumerate(C):
    for a, c, t in ((ax[0], 'r01', 'roll 0.1/s'), (ax[1], 'r02', 'roll 0.2/s')):
        x = [A[(f'rev2_{p}_st_{c}', v)] for v in cmds]
        a.bar(np.arange(4)+(i-1)*.25, [n/24 for n in x], .25, color=C[p], label=f'rev2_{p} ({sum(x)}/96)')
        a.set_title(f'flat, randomized, {t}, 24 x 60 s'); a.set_ylim(0, 1); a.set_ylabel('fall rate')
    ax[2].bar(np.arange(4)+(i-1)*.25, [S[p][v] for v in cmds], .25, color=C[p], label=f'rev2_{p}')
ax[2].set_ylabel('speed (m/s)'); ax[2].set_title('flat, nominal speed, 8 x 20 s')
for a in ax:
    a.set_xticks(range(4), [f'{v:g} m/s cmd' for v in cmds]); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{E}/media/flat_compare.png', dpi=110)
