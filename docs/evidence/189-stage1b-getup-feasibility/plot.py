"""Evidence 189 chart: get-up success of the fall curriculum by start tilt, and the 1b gate per run.
Usage: .venv-sim/bin/python docs/evidence/189-stage1b-getup-feasibility/plot.py -> media/getup_progress.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
rates = json.loads((OUT/'eval/fall_rates.json').read_text())
BINS = [(0, .3, 'standing\n< 0.3'), (.3, .8, 'tipping\n0.3-0.8'), (.8, 1.3, 'falling\n0.8-1.3'), (1.3, 3.2, 'on the ground\n> 1.3')]
RUNS = ['r3_gfa', 'r3_gfb', 'r3_gfc', 'r3_gfd']
COL = ['#9cc3d5', '#5b9bbd', '#2f6f95', '#123f5c']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [1.5, 1]})
w = .2
for i, run in enumerate(RUNS):
    rows = [r for k, v in rates.items() if k.startswith(run+'@') for r in v]
    ok, n = [], []
    for lo, hi, _ in BINS:
        sel = [r for r in rows if lo <= r['start_tilt'] < hi]
        n.append(len(sel)); ok.append(sum(r['t_up'] is not None for r in sel))
    frac = [o/max(m, 1) for o, m in zip(ok, n)]
    x = np.arange(len(BINS))+(i-1.5)*w
    ax1.bar(x, frac, w, color=COL[i], label=run)
    for xi, o, m in zip(x, ok, n):
        ax1.text(xi, o/max(m, 1)+.02, f'{o}/{m}', ha='center', fontsize=6.5, rotation=90)
ax1.set_xticks(range(len(BINS))); ax1.set_xticklabels([b[2] for b in BINS], fontsize=8)
ax1.set_xlabel('torso tilt (rad) when the policy takes over'); ax1.set_ylabel('stood and held 1 s within 10 s')
ax1.set_ylim(0, 1.15); ax1.legend(fontsize=8, ncol=4, loc='upper right')
ax1.set_title('Fall curriculum: kicked from standing, joints held, policy takes over after U(0, fall_t) s\n'
              '(each run on 20 starts at its own fall_t and 20 at 2.5 s; randomize off; strict 1 s hold)', fontsize=9)

gates = {}
for run in ('r3_g3', 'r3_g4', 'r3_gfd'):
    p = OUT/f'eval/gate_{run}.jsonl'
    if p.exists():
        gates[run] = {json.loads(l)['mode']: json.loads(l)['success'] for l in p.read_text().splitlines()}
modes = ['sit', 'side', 'back']
for i, (run, g) in enumerate(gates.items()):
    x = np.arange(3)+(i-1)*.25
    ax2.bar(x, [g[m] for m in modes], .25, label=run, color=['#bbbbbb', '#888888', '#123f5c'][i])
    for xi, m in zip(x, modes):
        ax2.text(xi, g[m]+.3, str(g[m]), ha='center', fontsize=8)
ax2.axhline(18, color='#b91c1c', ls='--', lw=1); ax2.text(2.45, 18.4, 'pass 18/20', color='#b91c1c', fontsize=8, ha='right')
ax2.set_xticks(range(3)); ax2.set_xticklabels(['sit (legs folded)', 'side', 'back'])
ax2.set_ylim(0, 21); ax2.set_ylabel('successes / 20'); ax2.legend(fontsize=8, loc='upper left')
ax2.set_title('Stage 1b gate (fallen starts, then 10 s on the 1a stand policy)', fontsize=9)
fig.suptitle('R-03 get-up, MuJoCo with the RobStride realism layer (not hardware)', fontsize=10)
fig.tight_layout()
fig.savefig(OUT/'media/getup_progress.png', dpi=130)
