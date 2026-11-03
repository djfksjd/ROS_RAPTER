"""Evidence 192 chart: best deep-fold four-bar per fold target (running motor score, >1 exceeds RS04) and the strict
A/B get-up probe at 3.2M of 8M steps.
Usage: .venv-sim/bin/python docs/evidence/192-knee-fold-fourbar/plot.py -> media/fold_ab.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
fs = json.loads((OUT/'eval/fold_search.json').read_text())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
folds = [(float(k), v['best'][0]) for k, v in fs['folds'].items() if v['best']]
x = [2.26]+[f for f, _ in folds]; y = [fs['reference']['score']]+[b['score'] for _, b in folds]
ax1.plot(x, y, 'o-', color='#2f6f95')
for xi, yi, b in zip(x, y, [None]+[b for _, b in folds]):
    ax1.annotate(f'{yi:.2f}' + (f'\nN={b["N_at_fold"]:.2f}' if b else '\ncurrent'), (xi, yi), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=8)
ax1.axhline(1., color='#b91c1c', ls='--', lw=1); ax1.text(2.27, 1.005, 'RS04 limit', color='#b91c1c', fontsize=8)
ax1.set_xlabel('knee fold the four-bar band must reach (rad)'); ax1.set_ylabel('worst running motor score (11.1 m/s design)')
ax1.set_ylim(.95, 1.3); ax1.set_title('Deep-fold four-bar search (best of random four-bars per target)', fontsize=9)
runs = [('A current knee', 'strict_r3_gxa_back.json'), ('B deep-fold knee', 'strict_r3_gxb_back.json')]
cats = ['fixed start\n(1)', 'fixed start\nrandomized (10)', 'unseen back\nstarts (10)']
for i, (lab, f) in enumerate(runs):
    d = json.loads((OUT/'eval'/f).read_text())
    vals = [d['fixed_nominal'], d['fixed_dr']/10, d['unseen_back']/10]
    hs = max(r['h_max'] for r in d['fixed']+d['unseen'])
    ax2.bar([j+(i-.5)*.35 for j in range(3)], vals, .35, color=['#9cc3d5', '#123f5c'][i], label=f'{lab} (max base height {hs:.2f} m)')
ax2.set_xticks(range(3)); ax2.set_xticklabels(cats, fontsize=8); ax2.set_ylim(0, 1.05)
ax2.set_ylabel('stood and held 1 s within 10 s'); ax2.legend(fontsize=8, loc='upper right')
ax2.text(1, .5, 'all 0 / 21\n(standing base height 0.80 m)', ha='center', fontsize=10, color='#555')
ax2.set_title('Strict get-up probe at 3.2M of 8M steps (leak off, deterministic)', fontsize=9)
fig.suptitle('R-03 deep-fold knee: running cost vs get-up result, MuJoCo with the RobStride realism layer (not hardware)', fontsize=10)
fig.tight_layout(); fig.savefig(OUT/'media/fold_ab.png', dpi=130)
