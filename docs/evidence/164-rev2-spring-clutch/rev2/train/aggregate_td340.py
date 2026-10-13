"""Aggregate the touchdown-clutch run (rev2_td340), its fixed-seed acceptance and the three-run comparison chart.
Run from the project root after rev2_accept.py, td340_latch_probe.py and media/make_media.py."""
import json, runpy, sys, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
T = Path(__file__).resolve().parent; A = T.parents[2]/'assets'
sys.argv = ['x']; G = runpy.run_path(str(T/'aggregate_train.py'), run_name='agg'); agg, run, rows = G['agg'], G['run'], G['rows']
p = json.loads((T/'td340_latch_probe.json').read_text()); acc = rows('td340_accept.jsonl'); pre = rows('n10_clutch_zero_shot.jsonl')
res = dict(train=run('rev2_td340'), accept=agg(acc), dr=agg(rows('td340_dr.jsonl')), ablate=agg(rows('td340_ablate.jsonl')), zero_shot_on_n10=agg(pre),
           gait={f'{float(c):g}': {k: [v['left'][k], v['right'][k]] for k in ('touchdown_mean', 'liftoff_mean', 'stance_max', 'duty')} for c, v in p.items()},
           hottest={f'{c:g}': dict(joint=sorted({r['heat_argmax'] for r in acc if r['cmd'] == c}), heat_max=float(np.mean([r['heat_max'] for r in acc if r['cmd'] == c]))) for c in (2., 4., 5.5, 7.)},
           media={m.stem: json.loads(m.read_text()) for m in sorted((T/'media').glob('*_4mps.json'))},
           rollouts=len(acc)+len(rows('td340_dr.jsonl'))+len(rows('td340_ablate.jsonl'))+len(pre))
(T/'td340_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
# ---- comparison chart
FONT = '/Library/Fonts/Arial Unicode.ttf'; font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(), 'axes.unicode_minus': False, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
R = json.loads((T/'train_results.json').read_text()); N = json.loads((T/'n10_results.json').read_text())
runs = [('rev2_a8_h2\n걸쇠, 명목 무릎 1.5', R['accept'], R['dr'], '#81939F'), ('rev2_n10\n걸쇠, 명목 무릎 1.0', N['accept'], N['dr'], '#BC6333'), ('rev2_td340\n접지각 클러치 k 340', res['accept'], res['dr'], '#147F86')]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.4), dpi=150); x = np.arange(4); cs = ('2', '4', '5.5', '7')
for i, (lab, a, d, col) in enumerate(runs):
    h = [a[f'accept@{c}']['heat_knee'] for c in cs]; b = ax[0].bar(x+(i-1)*.27, h, .26, color=col, label=lab)
    for bb, v in zip(b, h): ax[0].text(bb.get_x()+bb.get_width()/2, v+.02, f'{v:.2f}', ha='center', fontsize=7)
    ax[1].bar(x+(i-1)*.27, [a[f'accept@{c}']['speed'] for c in cs], .26, color=col)
    f = [d[f'dr@{c}']['falls'] for c in ('2', '4', '7')]; b = ax[2].bar(np.arange(3)+(i-1)*.27, f, .26, color=col)
    for bb, v in zip(b, f): ax[2].text(bb.get_x()+bb.get_width()/2, v+.6, f'{v}', ha='center', fontsize=7)
ax[0].axhline(1., color='#B22', ls='--', lw=1); ax[0].text(-.5, 1.03, '연속 정격', color='#B22', fontsize=8, ha='left')
ax[0].set(xticks=x, xticklabels=[f'{c} m/s' for c in cs], ylabel='무릎 부하제곱 (정격 = 1)', title='무릎 발열 (299회 평균)', ylim=(0, 2.45)); ax[0].legend(fontsize=7, loc='upper left', ncol=3)
ax[1].plot(x, [2, 4, 5.5, 7], 'k_', ms=22); ax[1].set(xticks=x, xticklabels=[f'{c} m/s' for c in cs], ylabel='실제 속도 (m/s)', title='속도 (가로선 = 명령)')
ax[2].set(xticks=np.arange(3), xticklabels=['2 m/s', '4 m/s', '7 m/s'], ylabel='60초 안 전도 / 59회', title='학습 무작위화 조건의 전도', ylim=(0, 59))
for a_ in ax: a_.grid(axis='y', alpha=.2)
fig.text(.01, .01, '고정 시드 96001–96299, 결정적 60초 rollout. MuJoCo 시뮬레이션이며 실물 성능이 아님. 학습 시드는 실행마다 하나.', fontsize=8); fig.tight_layout(rect=[0, .04, 1, 1])
fig.savefig(T/'media'/'rev2_compare.png'); plt.close(fig)
for src, dst in [('rev2_compare.png', 'rev2-compare.png'), ('rev2_n10_4mps_knee.png', 'rev2-n10-knee.png'), ('rev2_td340_4mps_knee.png', 'rev2-td340-knee.png'), ('rev2_n10_4mps_touchdown.png', 'rev2-n10-touchdown.png'), ('rev2_td340_4mps_touchdown.png', 'rev2-td340-touchdown.png')]:
    shutil.copyfile(T/'media'/src, A/dst)
print({k: (v['falls'], v['n'], round(v['speed'], 2), round(v['heat_knee'], 2), round(v['heat_max'], 2)) for k, v in {**res['accept'], **res['dr']}.items()})
print(res['train']['steps'], res['train']['first_pass_step'], res['train']['final_cmd_max'], res['hottest'], res['rollouts'])
