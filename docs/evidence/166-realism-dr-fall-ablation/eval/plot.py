"""Charts of evidence 166 from ablation_results.jsonl and impulse_log.jsonl."""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
E = 'docs/evidence/166-realism-dr-fall-ablation'
R = [json.loads(l) for l in open(f'{E}/eval/ablation_results.jsonl')]
G = [('all', 'all on'), ('impulse', 'impulses off'), ('plant', 'mass/friction off'), ('init', 'start noise off'),
     ('delay', 'delay nominal'), ('sensor', 'sensor bias off'), ('motor', 'motor/thermal nominal'),
     ('only_roll', 'only roll impulse'), ('only_yaw', 'only yaw impulse'), ('only_push', 'only push')]
f = [sum(r['falls'] for r in R if r['group'] == g) for g, _ in G]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].barh([l for _, l in G][::-1], f[::-1], color=['#c0392b' if g in ('all', 'only_roll', 'only_yaw') else '#7f8c8d' for g, _ in G][::-1])
ax[0].set_xlim(0, 32); ax[0].set_xlabel('falls / 32 (60 s, cmd 2-7 m/s, 8 seeds each)')
ax[0].set_title('rev2_rz1, realism on, randomize on: one group changed')
for i, v in enumerate(f[::-1]): ax[0].text(v+.3, i, str(v), va='center')
L = [json.loads(l) for l in open(f'{E}/eval/impulse_log.jsonl')]
bins = [0, .6, 1., 1.4, 1.7, 2., 2.5]
for axis, g, c in (('r', 'only_roll', '#c0392b'), ('z', 'only_yaw', '#2980b9')):
    ev = np.array([(abs(m), e['t_fall'] is not None and 0 <= e['t_fall']-t < 2.5) for r in L if r['group'] == g
                   for e in r['episodes'] for t, a, m in e['impulses'] if a == axis])
    rate = [ev[(ev[:, 0] >= lo) & (ev[:, 0] < hi), 1].mean()*100 if ((ev[:, 0] >= lo) & (ev[:, 0] < hi)).any() else np.nan
            for lo, hi in zip(bins, bins[1:])]
    ax[1].plot([(lo+hi)/2 for lo, hi in zip(bins, bins[1:])], rate, 'o-', color=c, label=f'{"roll" if axis == "r" else "yaw"} impulse')
ax[1].set_xlabel('impulse magnitude (N·m·s, 0.04 s)'); ax[1].set_ylabel('fall within 2.5 s (%)')
ax[1].set_title('fall rate per impulse'); ax[1].legend(); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{E}/media/ablation_falls.png', dpi=110)
