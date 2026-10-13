"""Aggregate the model_rev 2 training run (rev2_a8_h2), its fixed-seed acceptance and the follow-up fine-tune into train_results.json.
Run from the project root after rev2_accept.py, latch_probe.py and spring_id_rev2.py."""
import csv, json, statistics as st
from pathlib import Path
T = Path(__file__).resolve().parent; RUNS = Path.cwd()/'sim/rl/runs'
rows = lambda n: [json.loads(l) for l in (T/n).read_text().splitlines()]


def agg(R):
    out = {}
    for lab in sorted({r['label'] for r in R}):
        for c in sorted({r['cmd'] for r in R}):
            g = [r for r in R if r['cmd'] == c and r['label'] == lab]; ok = [r for r in g if not r['fell']]
            m = lambda k, G: float(st.mean(r[k] for r in G)) if G else None
            out[f'{lab}@{c:g}'] = dict(n=len(g), falls=sum(r['fell'] for r in g), speed=m('speed_tail', ok), speed_min=min((r['speed_tail'] for r in ok), default=None),
                                       heat_knee=m('heat_knee', ok), heat_knee_max=max((r['heat_knee'] for r in ok), default=None), heat_max=m('heat_max', ok),
                                       hottest=sorted({r['heat_argmax'].split('_', 1)[1] for r in ok}), engagements=sum(r['engagements'] for r in g),
                                       latched_stance_frac=m('latched_stance_frac', g), energy_injected_J=sum(r['energy_injected_J'] for r in g),
                                       self_contact_step_frac=m('self_contact_step_frac', g), knee_min=min(r['knee_min'] for r in g), fall_t=sorted(r['t'] for r in g if r['fell']))
    return out


def run(name):
    cur = [json.loads(l) for l in (RUNS/name/'curriculum.jsonl').read_text().splitlines()]; prog = list(csv.DictReader(open(RUNS/name/'progress.csv')))
    first = next((c for c in cur if c['eval_pass']), None)
    return dict(steps=int(float(prog[-1]['time/total_timesteps'])), first_pass_step=first and first['steps'], final_cmd_max=cur[-1]['cmd_max'][0],
                last_eval_speeds=[c['eval_speed'] for c in cur[-4:]], last_eval_falls=[c['eval_falls'] for c in cur[-4:]], args=json.loads((RUNS/name/'args.json').read_text()))


def probe(n):
    p = json.loads((T/n).read_text())
    return {f'{float(c):g}': {k: [v['left'][k], v['right'][k]] for k in ('touchdown_mean', 'liftoff_mean', 'swing_min', 'swing_max', 'stance_min', 'stance_max', 'duty')} for c, v in p.items()}


sid = json.loads((T/'spring_id_rev2.json').read_text())
res = dict(seeds='96001..96299 (fixed before the first result)', binomial=dict(n=299, falls=0, upper95_one_sided=1-.05**(1/299)),
           train=run('rev2_a8_h2'), baseline_train=run('scr_a8_h2'), finetune=run('rev2_ft_extend2'),
           accept=agg(rows('accept.jsonl')), dr=agg(rows('dr.jsonl')), dr_baseline_rev1=agg(rows('dr_baseline_rev1.jsonl')), on_rev1=agg(rows('rev1.jsonl')),
           ft_accept59=agg(rows('ft_extend2_accept59.jsonl')), ft_ablate=agg(rows('ft_extend2_ablate.jsonl')),
           gait=probe('latch_probe.json'), ft_gait=probe('latch_probe_ft_extend2.json'),
           replay={c: dict(none=sid[c]['none'], stance_k170_q0956=sid[c]['stance']['k170_q0956'], stance_best=sid[c]['stance']['best'], req_peak=sid[c]['req_peak'], req_stance_rms=sid[c]['req_stance_rms']) for c in ('2', '4', '7')},
           rollouts=dict(accept=len(rows('accept.jsonl')), dr=len(rows('dr.jsonl')), dr_baseline=len(rows('dr_baseline_rev1.jsonl')), on_rev1=len(rows('rev1.jsonl')),
                         ft=len(rows('ft_extend2_accept59.jsonl'))+len(rows('ft_extend2_ablate.jsonl'))))
res['training_steps'] = res['train']['steps']+res['finetune']['steps']-res['train']['steps'] if res['finetune']['steps'] > res['train']['steps'] else res['train']['steps']
(T/'train_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
a = res['accept']; print({k: (v['falls'], round(v['speed'], 3), round(v['heat_knee'], 3)) for k, v in a.items()}); print(res['train']['steps'], res['finetune']['steps'], res['training_steps'], res['binomial'])
print(res['train']['first_pass_step'], res['train']['final_cmd_max'], res['baseline_train']['final_cmd_max'], res['finetune']['final_cmd_max'])
print({k: [round(x, 3) for x in v['touchdown_mean']] for k, v in res['gait'].items()}, {c: (round(v['none'][0], 2), round(v['stance_k170_q0956'][0], 2)) for c, v in res['replay'].items()})
