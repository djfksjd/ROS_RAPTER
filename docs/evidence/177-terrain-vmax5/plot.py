"""Evidence 177 figures. terrain_falls.png: terrain falls (of 8) on stiff ground for rev2_ct5 (flat-trained, evidence 174),
rev2_tr1 (stage 1, evidence 175) rev2_tr2 (stage 2, evidence 176) and rev2_tr3 (terrain command up to 5 m/s). flat_compare.png: randomized flat-ground falls per speed and
nominal speed for the same three. Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D = {'ct5': 'docs/evidence/174-contact-stiffness', 'tr1': 'docs/evidence/175-terrain-training',
     'tr2': 'docs/evidence/176-terrain-stage2', 'tr3': 'docs/evidence/177-terrain-vmax5'}
E = D['tr3']
J = lambda f: [json.loads(l) for l in open(f)]
cmds = [2., 4., 5.5, 7.]
C = {'ct5': '#d62728', 'tr1': '#2ca02c', 'tr2': '#1f77b4', 'tr3': '#9467bd'}
T = {(r['run'].split('_')[1], r['kind'], r['value'], r['cmd']): r['falls'] for d in D.values() for r in J(f'{d}/eval/terrain.jsonl')}
G = [('flat', 0., 'flat'), *[('rough', a, f'rough +-{a*100:g} cm') for a in (.01, .02, .03, .04, .05, .06, .08)],
     *[('slope_up', g, f'uphill {g:g}°') for g in (5., 10., 15., 20., 25.)], *[('slope_down', g, f'downhill {g:g}°') for g in (5., 10., 15., 20., 25.)],
     *[('side_slope', g, f'side slope {g:g}°') for g in (5., 10., 15., 20.)], *[('stairs', r, f'stairs {r*100:g} cm') for r in (.02, .04, .06, .08, .1, .12)]]
fig, ax = plt.subplots(1, 4, figsize=(15, 10), sharey=True)
for a, p in zip(ax, C):
    M = np.array([[T[(p, k, v, c)] for c in (1., 2., 4.)] for k, v, _ in G])
    a.imshow(M/8, cmap='Reds', vmin=0, vmax=1, aspect='auto')
    for i in range(len(G)):
        for j in range(3):
            a.text(j, i, M[i, j], ha='center', va='center', color='white' if M[i, j] > 5 else 'black', fontsize=8)
    a.set_xticks(range(3), ['1 m/s', '2 m/s', '4 m/s']); a.set_title(f'rev2_{p}: falls of 8 ({M.sum()}/{M.size*8})')
ax[0].set_yticks(range(len(G)), [g[2] for g in G])
fig.tight_layout(); fig.savefig(f'{E}/media/terrain_falls.png', dpi=100)
A = {(r['run'], r['cmd']): r['falls'] for d in D.values() for r in J(f'{d}/eval/ablate.jsonl')}
S = {p: {r['cmd']: r['vx'] for r in J(f'{d}/eval/' + ('probe_nominal_ct5.jsonl' if p == 'ct5' else 'probe_nominal.jsonl'))}
     for p, d in D.items()}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, p in enumerate(C):
    for a, c, t in ((ax[0], 'r01', 'roll 0.1/s'), (ax[1], 'r02', 'roll 0.2/s')):
        x = [A[(f'rev2_{p}_st_{c}', v)] for v in cmds]
        a.bar(np.arange(4)+(i-1.5)*.2, [n/24 for n in x], .2, color=C[p], label=f'rev2_{p} ({sum(x)}/96)')
        a.set_title(f'flat, randomized, {t}, 24 x 60 s'); a.set_ylim(0, 1); a.set_ylabel('fall rate')
    ax[2].bar(np.arange(4)+(i-1.5)*.2, [S[p][v] for v in cmds], .2, color=C[p], label=f'rev2_{p}')
ax[2].set_ylabel('speed (m/s)'); ax[2].set_title('flat, nominal speed, 8 x 20 s')
for a in ax:
    a.set_xticks(range(4), [f'{v:g} m/s cmd' for v in cmds]); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{E}/media/flat_compare.png', dpi=110)
