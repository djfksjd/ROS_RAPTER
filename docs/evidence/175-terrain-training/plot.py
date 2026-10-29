"""Evidence 175 figures. terrain_falls.png: terrain falls (of 8) on stiff ground, rev2_ct5 (flat-trained, evidence 174) vs
rev2_tr1 (terrain stage 1). flat_compare.png: randomized flat-ground falls per speed and nominal speed, ct5 vs tr1.
Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E, O = 'docs/evidence/175-terrain-training', 'docs/evidence/174-contact-stiffness'
J = lambda f: [json.loads(l) for l in open(f)]
cmds = [2., 4., 5.5, 7.]
C = {'ct5': '#d62728', 'tr1': '#2ca02c'}
T = {(r['run'].split('_')[1], r['kind'], r['value'], r['cmd']): r['falls']
     for f in (f'{O}/eval/terrain.jsonl', f'{E}/eval/terrain.jsonl') for r in J(f)}
G = [('flat', 0., 'flat'), *[('rough', a, f'rough +-{a*100:g} cm') for a in (.01, .02, .03, .04, .05, .06, .08)],
     *[('slope_up', g, f'uphill {g:g}°') for g in (5., 10., 15., 20., 25.)], *[('slope_down', g, f'downhill {g:g}°') for g in (5., 10., 15., 20., 25.)],
     *[('side_slope', g, f'side slope {g:g}°') for g in (5., 10., 15., 20.)], *[('stairs', r, f'stairs {r*100:g} cm') for r in (.02, .04, .06, .08, .1, .12)]]
fig, ax = plt.subplots(1, 2, figsize=(9, 10), sharey=True)
for a, p in zip(ax, C):
    M = np.array([[T[(p, k, v, c)] for c in (1., 2., 4.)] for k, v, _ in G])
    a.imshow(M/8, cmap='Reds', vmin=0, vmax=1, aspect='auto')
    for i in range(len(G)):
        for j in range(3):
            a.text(j, i, M[i, j], ha='center', va='center', color='white' if M[i, j] > 5 else 'black', fontsize=8)
    a.set_xticks(range(3), ['1 m/s', '2 m/s', '4 m/s']); a.set_title(f'rev2_{p}, stiff ground: falls of 8 ({M.sum()}/{M.size*8})')
ax[0].set_yticks(range(len(G)), [g[2] for g in G])
fig.tight_layout(); fig.savefig(f'{E}/media/terrain_falls.png', dpi=100)
A = {(r['run'], r['cmd']): r['falls'] for f in (f'{O}/eval/ablate.jsonl', f'{E}/eval/ablate.jsonl') for r in J(f)}
S = {'ct5': {r['cmd']: r['vx'] for r in J(f'{O}/eval/probe_nominal_ct5.jsonl')},
     'tr1': {r['cmd']: r['vx'] for r in J(f'{E}/eval/probe_nominal.jsonl')}}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, p in enumerate(C):
    for a, c, t in ((ax[0], 'r01', 'roll 0.1/s'), (ax[1], 'r02', 'roll 0.2/s')):
        x = [A[(f'rev2_{p}_st_{c}', v)] for v in cmds]
        a.bar(np.arange(4)+(i-.5)*.38, [n/24 for n in x], .38, color=C[p], label=f'rev2_{p} ({sum(x)}/96)')
        a.set_title(f'flat, randomized, {t}, 24 x 60 s'); a.set_ylim(0, 1); a.set_ylabel('fall rate')
    ax[2].bar(np.arange(4)+(i-.5)*.38, [S[p][v] for v in cmds], .38, color=C[p], label=f'rev2_{p}')
ax[2].set_ylabel('speed (m/s)'); ax[2].set_title('flat, nominal speed, 8 x 20 s')
for a in ax:
    a.set_xticks(range(4), [f'{v:g} m/s cmd' for v in cmds]); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{E}/media/flat_compare.png', dpi=110)
