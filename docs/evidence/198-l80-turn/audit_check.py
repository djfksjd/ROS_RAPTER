"""Recompute the stopped run's check decisions from saved episode data; no simulation."""
import csv
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from check_training import EVIDENCE, ROOT, metrics, stop_reason


def audit(name='l80_t1'):
    run = ROOT/'sim/rl/runs'/name
    recorded = json.loads((run/'turn_checks.json').read_text())
    history, results = [], []
    for old in recorded:
        block = old['block']
        rows = [json.loads(line) for line in (EVIDENCE/f'eval/{name}/turn_{block}M.jsonl').read_text().splitlines()]
        conditions = sorted([r['cmd'], r['yaw_cmd'], e['seed']]
                            for r in rows if r['cmd'] >= 2 for e in r['episodes'])
        snap = run.with_name(f'{name}_{block}M')
        with (snap/'progress.csv').open() as f:
            lengths = [float(r['rollout/ep_len_mean']) for r in csv.DictReader(f) if r.get('rollout/ep_len_mean')]
        entry = dict(block=block, added_steps=old['added_steps'], **metrics(rows))
        entry['decision'] = stop_reason(entry, history, lengths)
        assert entry == old, (entry, old)
        prior = next((r for r in results if r['block'] == block-2), None)
        results.append(dict(**entry, conditions=conditions, compare_block=prior['block'] if prior else None,
                            same_conditions=conditions == prior['conditions'] if prior else None,
                            first10_length=sum(lengths[:10])/10, last10_length=sum(lengths[-10:])/10,
                            progress_rows=len(lengths)))
        history.append(entry)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    args = json.loads((run/'args.json').read_text())
    initial = json.loads((run/'initial-policy.json').read_text())['steps']
    log = [json.loads(s) for s in (ROOT/f'sim/rl/runs_{name}.log').read_text().splitlines() if s.startswith('{')]
    logged_steps = [r['steps'] for r in log if 'steps' in r]
    assert [r for r in log if 'block' in r] == recorded
    assert [n-initial for n in logged_steps] == [r['added_steps'] for r in recorded]
    assert [r['added_steps'] for r in recorded] == [999424*r['block'] for r in recorded]
    assert results[-1]['added_steps'] <= args['steps']
    with zipfile.ZipFile(run/'model.zip') as z:
        saved = json.loads(z.read('data'))
    with (run/'progress.csv').open() as f:
        progress = list(csv.DictReader(f))
    assert saved['num_timesteps'] == int(progress[-1]['time/total_timesteps']) == logged_steps[-1]
    rollout = saved['n_steps']*saved['n_envs']
    usable_budget = int(args['steps'])//rollout*rollout
    assert all(r['added_steps'] % rollout == 0 for r in recorded)
    return dict(run=name, blocks=results, initial_steps=initial, final_steps=logged_steps[-1],
                declared_budget=args['steps'], consumed=results[-1]['added_steps'],
                rollout_steps=rollout, usable_budget=usable_budget,
                unused_usable_budget=usable_budget-results[-1]['added_steps'],
                budget_ok=True, log_matches=True, model_progress_steps_match=True,
                stop_valid=results[-1]['same_conditions'] and results[-1]['decision'].startswith('stop:'),
                improvement_1M_to_3M=results[0]['yaw_mae']-results[2]['yaw_mae'],
                current_equals_3M={n: digest(run/n) == digest(run.with_name(f'{name}_3M')/n)
                                   for n in ('model.zip', 'vecnorm.pkl')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', choices=['l80_t1', 'l80_t2'], default='l80_t1')
    print(json.dumps(audit(parser.parse_args().run), indent=2, allow_nan=False))
