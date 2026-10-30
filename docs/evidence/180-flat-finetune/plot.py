"""Evidence 180 figure: rev2_tr2f (rev2_tr2 + 5M flat-only steps) vs rev2_tr2 and rev2_ct5. Left: nominal speed per
command. Middle: randomized falls per command (roll 0.1/s). Right: training episode length (steps of 1500) over the run.
Run from repo root."""
import csv, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D = {'ct5': 'docs/evidence/174-contact-stiffness', 'tr2': 'docs/evidence/176-terrain-stage2', 'tr2f': 'docs/evidence/180-flat-finetune'}
E = D['tr2f']
J = lambda f: [json.loads(l) for l in open(f)]
C = {'ct5': '#d62728', 'tr2': '#1f77b4', 'tr2f': '#ff7f0e'}
cmds = [2., 4., 5.5, 7.]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for p, d in D.items():
    s = {r['cmd']: r['vx'] for r in J(f'{d}/eval/' + ('probe_nominal_ct5.jsonl' if p == 'ct5' else 'probe_nominal.jsonl'))}
    ax[0].plot(cmds, [s[v] for v in cmds], 'o-', color=C[p], label=f'rev2_{p}')
    a = {r['cmd']: r['falls'] for r in J(f'{d}/eval/ablate.jsonl') if r['run'] == f'rev2_{p}_st_r01'}
    ax[1].plot(cmds, [a[v] for v in cmds], 'o-', color=C[p], label=f'rev2_{p} ({sum(a.values())}/96)')
    rows = [x for x in csv.DictReader(open(f'{d}/policy/progress.csv')) if x['rollout/ep_len_mean']]
    st = np.array([float(x['time/total_timesteps']) for x in rows]); st -= st[0]
    ax[2].plot(st/1e6, [float(x['rollout/ep_len_mean']) for x in rows], color=C[p], label=f'rev2_{p}')
ax[0].plot([2, 7], [2, 7], 'k:', lw=1); ax[0].set_xlabel('command (m/s)'); ax[0].set_ylabel('speed (m/s)')
ax[0].set_title('flat, nominal, 8 x 20 s')
ax[1].set_xlabel('command (m/s)'); ax[1].set_ylabel('falls of 24'); ax[1].set_title('flat, randomized, roll 0.1/s, 24 x 60 s')
ax[2].set_xlabel('steps in this run (M)'); ax[2].set_ylabel('training episode length (steps, max 1500)'); ax[2].set_title('training')
for a in ax: a.legend(fontsize=8); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f'{E}/media/flat_finetune.png', dpi=110)
