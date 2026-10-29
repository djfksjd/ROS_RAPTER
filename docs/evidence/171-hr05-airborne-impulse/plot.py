"""Evidence 171 figure: (1) roll-impulse falls by airborne share of the 0.2 s after the impulse (rev2_hr05, 4-7 m/s);
(2) airborne share per command speed, rev2_hr05 vs rev2_hd40; (3) randomized falls per speed (24 x 60 s, roll 0.1/s).
Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/171-hr05-airborne-impulse'
D = {n: [json.loads(l) for l in open(f'{E}/eval/roll_diag_air_{n}.jsonl')] for n in ('hr05', 'hd40')}
cmds = [2., 4., 5.5, 7.]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
H = [d for d in D['hr05'] if d['mag'] >= 1 and d['cmd'] >= 4]
bins = [(0, .34, '< 1/3'), (.34, .67, '1/3 - 2/3'), (.67, 1.01, '> 2/3')]
rate = [(sum(d['fell'] for d in s), len(s)) for s in ([d for d in H if lo <= d['air_02'] < hi] for lo, hi, _ in bins)]
ax[0].bar(range(3), [f/n for f, n in rate], color='#1f77b4')
for i, (f, n) in enumerate(rate):
    ax[0].text(i, f/n+.02, f'{f}/{n}', ha='center')
ax[0].set_xticks(range(3), [b[2] for b in bins]); ax[0].set_ylim(0, 1.1); ax[0].set_xlabel('airborne share of 0.2 s after impulse')
ax[0].set_ylabel('fell'); ax[0].set_title('rev2_hr05, roll impulse >= 1 N m s, 4-7 m/s')
for i, (n, c) in enumerate([('hr05', '#1f77b4'), ('hd40', '#2ca02c')]):
    ax[1].bar(np.arange(4)+(i-.5)*.38, [np.mean([d['air_frac'] for d in D[n] if d['cmd'] == v]) for v in cmds], .38, color=c, label=f'rev2_{n}')
ax[1].set_xticks(range(4), [f'{v:g} m/s' for v in cmds]); ax[1].set_ylim(0, .7); ax[1].set_ylabel('airborne share (0.3 s windows)')
ax[1].set_title('time with no foot loaded'); ax[1].legend()
T = {}
for f in ['docs/evidence/169-hip-roll-band/eval/ablate.jsonl', 'docs/evidence/169-hip-roll-band/eval/ablate_more_seeds.jsonl', f'{E}/eval/ablate.jsonl']:
    for l in open(f):
        r = json.loads(l)
        if r['group'] == 'all' and r['run'] in ('rev2_hr05_ev_r01', 'rev2_hd40_ev_r01'):
            k = (r['run'], r['cmd']); T.setdefault(k, [0, 0]); T[k][0] += r['falls']; T[k][1] += r['n']
for i, (n, c) in enumerate([('hr05', '#1f77b4'), ('hd40', '#2ca02c')]):
    x = [T[(f'rev2_{n}_ev_r01', v)] for v in cmds]
    ax[2].bar(np.arange(4)+(i-.5)*.38, [a/b for a, b in x], .38, color=c, label=f'rev2_{n} ({sum(a for a, _ in x)}/{sum(b for _, b in x)})')
ax[2].set_xticks(range(4), [f'{v:g} m/s' for v in cmds]); ax[2].set_ylim(0, 1); ax[2].set_ylabel('fall rate')
ax[2].set_title('randomized, roll 0.1/s, 24 x 60 s'); ax[2].legend()
fig.tight_layout(); fig.savefig(f'{E}/media/airborne_falls.png', dpi=110)
