"""Evidence 170 figure: roll-impulse fall rate by impulse size at 2 m/s vs 4-7 m/s (rev2_hr05, rev2_hr065), and signed
hip-roll use in falls (largest outward / inward angle over the joint stop, action at band end). Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/170-hr05-highspeed-falls'
D = {n: [json.loads(l) for l in open(f'{E}/eval/roll_diag_signed_{n}.jsonl')] for n in ('hr05', 'hr065')}
bins = [0, .6, 1., 1.4, 1.8, 2.1]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n, c in [('hr05', '#1f77b4'), ('hr065', '#ff7f0e')]:
    for cmds, ls, lab in [((2.,), '--', '2 m/s'), ((4., 5.5, 7.), '-', '4-7 m/s')]:
        y = []
        for lo, hi in zip(bins, bins[1:]):
            s = [d for d in D[n] if lo <= d['mag'] < hi and d['cmd'] in cmds]
            y.append(sum(d['fell'] for d in s)/max(len(s), 1))
        ax[0].plot([(a+b)/2 for a, b in zip(bins, bins[1:])], y, ls, color=c, marker='o', label=f'rev2_{n} {lab}')
ax[0].set_xlabel('roll impulse (N m s)'); ax[0].set_ylabel('fell within 2.5 s'); ax[0].set_ylim(0, 1); ax[0].legend()
ax[0].set_title('roll impulse size vs falls (24 x 65 s per speed)')
keys = [('act_sat', 'action at band end'), ('out_frac', 'outward / stop'), ('in_frac', 'inward / stop')]
for i, (n, c) in enumerate([('hr05', '#1f77b4'), ('hr065', '#ff7f0e')]):
    s = [d for d in D[n] if d['mag'] >= 1 and d['fell']]
    ax[1].bar(np.arange(3)+(i-.5)*.38, [np.mean([d[k] for d in s]) for k, _ in keys], .38, color=c, label=f'rev2_{n} falls (n={len(s)})')
ax[1].set_xticks(range(3), [l for _, l in keys]); ax[1].set_ylim(0, 1); ax[1].legend()
ax[1].set_title('hip roll in falls (stop 0.5 vs 0.65 rad)')
fig.tight_layout(); fig.savefig(f'{E}/media/impulse_size_falls.png', dpi=110)
