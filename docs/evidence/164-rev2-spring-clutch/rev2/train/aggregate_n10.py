"""Aggregate the nominal-knee-1.0 run (rev2_n10) and its fixed-seed acceptance into n10_results.json.
Run from the project root after rev2_accept.py, n10_latch_probe.py and n10_spring_id.py."""
import json, runpy, sys
from pathlib import Path
T = Path(__file__).resolve().parent
sys.argv = ['x']; A = runpy.run_path(str(T/'aggregate_train.py'), run_name='agg')
agg, run, rows = A['agg'], A['run'], A['rows']
p = json.loads((T/'n10_latch_probe.json').read_text()); sid = json.loads((T/'n10_spring_id.json').read_text())
acc = rows('n10_accept.jsonl'); a7 = [r for r in acc if r['cmd'] == 7.]
res = dict(train=run('rev2_n10'), accept=agg(acc), dr=agg(rows('n10_dr.jsonl')), ablate=agg(rows('n10_ablate.jsonl')), peek_3M=agg(rows('n10_3M_peek.jsonl')),
           gait={f'{float(c):g}': {k: [v['left'][k], v['right'][k]] for k in ('touchdown_mean', 'liftoff_mean', 'swing_min', 'swing_max', 'stance_min', 'stance_max', 'duty')} for c, v in p.items()},
           replay={c: dict(none=sid[c]['none'], stance_k170_q0956=sid[c]['stance']['k170_q0956'], stance_best=sid[c]['stance']['best'], req_peak=sid[c]['req_peak']) for c in ('2', '4', '7')},
           seeds_with_engagement_7=sum(r['engagements'] > 0 for r in a7), max_engagements_7=max(r['engagements'] for r in a7),
           binomial_1_of_299_upper95=None,
           rollouts=len(acc)+len(rows('n10_dr.jsonl'))+len(rows('n10_ablate.jsonl'))+len(rows('n10_3M_peek.jsonl')))
# one-sided 95 % Clopper-Pearson upper bound for 1 fall in 299: solve (1-p)^299 + 299 p (1-p)^298 = 0.05
lo, hi = 0., 1.
for _ in range(200):
    m = (lo+hi)/2; lo, hi = (m, hi) if (1-m)**299+299*m*(1-m)**298 > .05 else (lo, m)
res['binomial_1_of_299_upper95'] = lo
(T/'n10_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
print({k: (v['falls'], v['n'], round(v['speed'], 3), round(v['heat_knee'], 3), v['engagements'], v['fall_t']) for k, v in {**res['accept'], **res['dr'], **res['ablate'], **res['peek_3M']}.items()})
print(res['train']['steps'], res['train']['first_pass_step'], res['train']['final_cmd_max'], res['train']['last_eval_speeds'], res['binomial_1_of_299_upper95'], res['rollouts'])
print({k: [round(x, 3) for x in v['touchdown_mean']] for k, v in res['gait'].items()}, {c: (round(v['none'][0], 2), round(v['stance_k170_q0956'][0], 2), round(v['req_peak'], 1)) for c, v in res['replay'].items()})
