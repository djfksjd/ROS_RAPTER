"""Evidence 178 figure: policy switching on flat stiff ground. Left: mean speed per phase (nominal, 24 seeds, v = 3 m/s)
for the switch profile and the two single-policy controls. Right: randomized falls (24 seeds) per profile and speed.
Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/178-policy-switch'
R = [json.loads(l) for l in open(f'{E}/eval/switch.jsonl')]
C = {'switch': '#9467bd', 'tr2_only': '#1f77b4', 'ct5_only': '#d62728'}
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
dur = [6, 6, 6, 4, 6]; edges = np.concatenate([[0], np.cumsum(dur)])
for r in R:
    if not r['dr'] and r['v'] == 3.:
        vx = np.nanmean([[x if x is not None else np.nan for x in e['vx']] for e in r['episodes']], 0)
        for i in range(5):
            ax[0].plot(edges[i:i+2], [vx[i]]*2, color=C[r['profile']], lw={'switch': 6, 'tr2_only': 2.5, 'ct5_only': 2.5}[r['profile']], alpha={'switch': .45, 'tr2_only': 1, 'ct5_only': 1}[r['profile']], ls={'switch': '-', 'tr2_only': '--', 'ct5_only': ':'}[r['profile']], label=r['profile'] if i == 0 else None)
            if r['profile'] == 'switch':
                ax[0].text((edges[i]+edges[i+1])/2, 5.15, r['phases'][i], ha='center', fontsize=8)
ax[0].set_ylim(2, 5.4); ax[0].set_xlabel('time (s)'); ax[0].set_ylabel('mean speed per phase (m/s)')
ax[0].set_title('nominal, 24 seeds, command 3 -> 5.5 -> 3 m/s'); ax[0].legend(fontsize=8, loc='center left')
for j, p in enumerate(C):
    f = [next(r['falls'] for r in R if r['dr'] and r['v'] == v and r['profile'] == p) for v in (2., 3., 4.)]
    ax[1].bar(np.arange(3)+(j-1)*.27, f, .27, color=C[p], label=f'{p} ({sum(f)}/72)')
ax[1].set_xticks(range(3), ['v = 2 m/s', 'v = 3 m/s', 'v = 4 m/s']); ax[1].set_ylabel('falls of 24 (28 s run)')
ax[1].set_title('randomized + roll impulses 0.1/s'); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f'{E}/media/switch.png', dpi=110)
