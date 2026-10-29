"""Evidence 172 figure: rev2_hr05 (duty 0.28) vs rev2_hd34 (0.34) vs rev2_hd40 (0.40):
(1) randomized falls per speed (24 x 60 s, roll 0.1/s); (2) airborne share per speed; (3) nominal speed. Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/172-gait-duty-034'
A = 'docs/evidence/171-hr05-airborne-impulse'
P = {'hr05': ('#1f77b4', 'docs/evidence/169-hip-roll-band'), 'hd34': ('#ff7f0e', E), 'hd40': ('#2ca02c', A)}
cmds = [2., 4., 5.5, 7.]
T = {}
for f in ['docs/evidence/169-hip-roll-band/eval/ablate.jsonl', 'docs/evidence/169-hip-roll-band/eval/ablate_more_seeds.jsonl',
          f'{A}/eval/ablate.jsonl', f'{E}/eval/ablate.jsonl']:
    for l in open(f):
        r = json.loads(l)
        if r['group'] == 'all' and r['run'] in [f'rev2_{n}_ev_r01' for n in P]:
            k = (r['run'], r['cmd']); T.setdefault(k, [0, 0]); T[k][0] += r['falls']; T[k][1] += r['n']
air = {n: [json.loads(l) for l in open(f'{A if n != "hd34" else E}/eval/roll_diag_air_{n}.jsonl')] for n in P}
vx = {n: {r['cmd']: r['vx'] for r in map(json.loads, open(f'{d}/eval/probe_nominal.jsonl'))} for n, (_, d) in P.items()}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
w = .27
for i, (n, (c, _)) in enumerate(P.items()):
    x = np.arange(4)+(i-1)*w
    t = [T[(f'rev2_{n}_ev_r01', v)] for v in cmds]
    ax[0].bar(x, [a/b for a, b in t], w, color=c, label=f'rev2_{n} ({sum(a for a, _ in t)}/{sum(b for _, b in t)})')
    ax[1].bar(x, [np.mean([d['air_frac'] for d in air[n] if d['cmd'] == v]) for v in cmds], w, color=c, label=f'rev2_{n}')
    ax[2].bar(x, [vx[n][v] for v in cmds], w, color=c, label=f'rev2_{n}')
for a, yl, t in [(ax[0], 'fall rate', 'randomized, roll 0.1/s, 24 x 60 s'), (ax[1], 'airborne share (0.3 s windows)', 'time with no foot loaded'),
                 (ax[2], 'speed (m/s)', 'nominal speed, 8 x 20 s')]:
    a.set_xticks(range(4), [f'{v:g} m/s cmd' for v in cmds]); a.set_ylabel(yl); a.set_title(t); a.legend(fontsize=8)
ax[0].set_ylim(0, 1); ax[1].set_ylim(0, .7)
fig.tight_layout(); fig.savefig(f'{E}/media/duty_compare.png', dpi=110)
