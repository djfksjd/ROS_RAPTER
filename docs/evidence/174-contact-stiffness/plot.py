"""Evidence 174 figures. contact_falls.png: (1) ground penetration of rev2_hr05 on flat ground per contact time constant;
(2) randomized falls per speed on stiff ground (0.005), rev2_hr05 vs rev2_ct5; (3) nominal speed on stiff ground.
terrain_falls.png: zero-shot terrain falls (of 8) on stiff ground. Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/174-contact-stiffness'
J = lambda f: [json.loads(l) for l in open(f'{E}/eval/{f}')]
cmds = [2., 4., 5.5, 7.]
C = {'hr05': '#1f77b4', 'ct5': '#d62728'}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
P = J('contact_hr05.jsonl')
for i, tc in enumerate((.02, .01, .005)):
    r = {x['cmd']: x for x in P if x['tc'] == tc}
    v = [2., 4., 7.]
    ax[0].bar(np.arange(3)+(i-1)*.27, [r[c]['pen_max']*100 for c in v], .27, alpha=.35, color=f'C{i}')
    ax[0].bar(np.arange(3)+(i-1)*.27, [r[c]['pen_med']*100 for c in v], .27, color=f'C{i}', label=f'tc {tc} (dark median, light max)')
ax[0].set_xticks(range(3), ['2 m/s', '4 m/s', '7 m/s']); ax[0].set_ylabel('foot penetration (cm)'); ax[0].legend(fontsize=8)
ax[0].set_title('rev2_hr05 on flat ground, 8 x 20 s')
A = {(r['run'], r['cmd']): r['falls'] for r in J('ablate.jsonl')}
for i, p in enumerate(C):
    x = [A[(f'rev2_{p}_st_r01', v)] for v in cmds]
    ax[1].bar(np.arange(4)+(i-.5)*.38, [a/24 for a in x], .38, color=C[p], label=f'rev2_{p} ({sum(x)}/96)')
    s = {r['cmd']: r['vx'] for r in J(f'probe_nominal_{p}.jsonl')}
    ax[2].bar(np.arange(4)+(i-.5)*.38, [s[v] for v in cmds], .38, color=C[p], label=f'rev2_{p}')
ax[1].set_ylim(0, 1); ax[1].set_ylabel('fall rate'); ax[1].set_title('stiff ground 0.005: randomized, roll 0.1/s, 24 x 60 s')
ax[2].set_ylabel('speed (m/s)'); ax[2].set_title('stiff ground 0.005: nominal speed, 8 x 20 s')
for a in ax[1:]:
    a.set_xticks(range(4), [f'{v:g} m/s cmd' for v in cmds]); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{E}/media/contact_falls.png', dpi=110)
T = {(r['run'].split('_')[1], r['kind'], r['value'], r['cmd']): r['falls'] for r in J('terrain.jsonl')}
G = [('flat', 0., 'flat'), *[('rough', a, f'rough +-{a*100:g} cm') for a in (.01, .02, .03, .04, .05, .06, .08)],
     *[('slope_up', g, f'uphill {g:g}°') for g in (5., 10., 15., 20., 25.)], *[('slope_down', g, f'downhill {g:g}°') for g in (5., 10., 15., 20., 25.)],
     *[('side_slope', g, f'side slope {g:g}°') for g in (5., 10., 15., 20.)], *[('stairs', r, f'stairs {r*100:g} cm') for r in (.02, .04, .06, .08, .1, .12)]]
fig, ax = plt.subplots(1, 2, figsize=(9, 10), sharey=True)
for a, p in zip(ax, ('hr05', 'ct5')):
    M = np.array([[T[(p, k, v, c)] for c in (1., 2., 4.)] for k, v, _ in G])
    a.imshow(M/8, cmap='Reds', vmin=0, vmax=1, aspect='auto')
    for i in range(len(G)):
        for j in range(3):
            a.text(j, i, M[i, j], ha='center', va='center', color='white' if M[i, j] > 5 else 'black', fontsize=8)
    a.set_xticks(range(3), ['1 m/s', '2 m/s', '4 m/s']); a.set_title(f'rev2_{p}, stiff ground: falls of 8 ({M.sum()}/{M.size*8})')
ax[0].set_yticks(range(len(G)), [g[2] for g in G])
fig.tight_layout(); fig.savefig(f'{E}/media/terrain_falls.png', dpi=100)
