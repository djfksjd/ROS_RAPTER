"""Evidence 169 figure: falls per command speed, rev2_rz1 vs rev2_hr05 (24 seeds, v2 realism, randomized), and roll-impulse
diagnosis (falls vs survived) for both. Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/169-hip-roll-band'
T = {}
def add(path, ren={}):
    for l in open(path):
        r = json.loads(l)
        if r['group'] != 'all': continue
        k = (ren.get(r['run'], r['run']), r['cmd']); T.setdefault(k, [0, 0]); T[k][0] += r['falls']; T[k][1] += r['n']
add('docs/evidence/167-robstride-public-data-fit/eval/ablate_v2.jsonl', {'rev2_rz1_v2eval': 'rev2_rz1_ev_r01'})
add('docs/evidence/168-impulse-robust-training/eval/ablate.jsonl')
add(f'{E}/eval/ablate.jsonl'); add(f'{E}/eval/ablate_more_seeds.jsonl')
cmds = [2., 4., 5.5, 7.]
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, rate, title in [(ax[0], 'r01', 'roll impulse 0.1/s'), (ax[1], 'r02', 'roll impulse 0.2/s')]:
    for i, (run, c) in enumerate([('rev2_rz1', '#888'), ('rev2_hr05', '#1f77b4')]):
        f = [T[(f'{run}_ev_{rate}', v)] for v in cmds]
        a.bar(np.arange(4) + (i - .5) * .38, [x[0] / x[1] for x in f], .38, color=c,
              label=f'{run} ({sum(x[0] for x in f)}/{sum(x[1] for x in f)})')
    a.set_xticks(range(4), [f'{v:g} m/s' for v in cmds]); a.set_ylim(0, 1); a.set_ylabel('fall rate (24 x 60 s)')
    a.set_title(title); a.legend()
D = {'rev2_rz1': [json.loads(l) for l in open('docs/evidence/168-impulse-robust-training/eval/roll_diag_rz1.jsonl')],
     'rev2_hr05': [json.loads(l) for l in open(f'{E}/eval/roll_diag_hr05.jsonl')]}
keys = [('act_sat', 'action at band end'), ('q_frac', 'angle / joint limit')]
for i, (run, c) in enumerate([('rev2_rz1', '#888'), ('rev2_hr05', '#1f77b4')]):
    s = [d for d in D[run] if abs(d['mag']) >= 1 and d['fell']]
    ax[2].bar(np.arange(2) + (i - .5) * .38, [np.mean([d[k] for d in s]) for k, _ in keys], .38, color=c, label=f'{run} falls (n={len(s)})')
ax[2].set_xticks(range(2), [n for _, n in keys]); ax[2].set_ylim(0, 1)
ax[2].set_title('hip roll, 1.5 s after roll impulse >= 1 N m s'); ax[2].legend()
fig.tight_layout(); fig.savefig(f'{E}/media/falls_rz1_vs_hr05.png', dpi=110)
