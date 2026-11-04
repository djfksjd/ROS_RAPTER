"""Recompute the stopped run's check decisions from saved episode data; no simulation."""
import csv
import hashlib
import json
from pathlib import Path

from check_training import EVIDENCE, ROOT, metrics, stop_reason


def audit():
    run = ROOT/'sim/rl/runs/l80_t1'
    recorded = json.loads((run/'turn_checks.json').read_text())
    history, results = [], []
    for old in recorded:
        block = old['block']
        rows = [json.loads(line) for line in (EVIDENCE/f'eval/l80_t1/turn_{block}M.jsonl').read_text().splitlines()]
        conditions = sorted([r['cmd'], r['yaw_cmd'], e['seed']]
                            for r in rows if r['cmd'] >= 2 for e in r['episodes'])
        snap = run.with_name(f'l80_t1_{block}M')
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
    return dict(blocks=results, stop_valid=results[-1]['same_conditions'] and results[-1]['decision'].startswith('stop:'),
                improvement_1M_to_3M=results[0]['yaw_mae']-results[2]['yaw_mae'],
                current_equals_3M={n: digest(run/n) == digest(run.with_name('l80_t1_3M')/n)
                                   for n in ('model.zip', 'vecnorm.pkl')})


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
