"""Verify complete decision-B evidence, run the official judge and save a summary."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
from gate_judge import judge, sustained_speed, turn_episode_ok, turn_suite_status

E = Path(__file__).resolve().parent
OUT = E/'eval/decision_b'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row):
    return tuple(row.get(k) for k in ('gate', 'cond', 'cmd', 'yaw_cmd', 'dir', 'dv', 'mount'))


def main():
    inputs = json.loads((OUT/'inputs.json').read_text())
    assert inputs['unchanged'] and inputs['before'] == inputs['after']
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in inputs['after'].items())
    old = {key(r): r for r in read(E/'eval/t3_full_turn.jsonl')}
    results = {}
    for run in ('l80_t3', 'l80_r1_6M'):
        full, extra = (read(OUT/f'{run}_{part}.jsonl') for part in ('full', 'extra'))
        assert len(full) == 36 and sum(r['n'] for r in full) == 240
        assert len(extra) == 9 and sum(r['n'] for r in extra) == 26
        assert len({key(r) for r in full+extra}) == 45
        assert {key(r) for r in full} == set(old)
        assert {(r['cmd'], r['mount']) for r in extra if r['cond'] == 'heat'} == {
            (v, m) for v in (4., 5., 6.) for m in ('plate', 'worst')}
        assert {r['dir'] for r in extra if r['cond'] == 'lateral'} == {'left', 'right'}
        assert sum(r['cond'] == 'speed' and r['cmd'] == 8. and r['n'] == 4 for r in extra) == 1
        for row in full+extra:
            if 'episodes' in row:
                assert len(row['episodes']) == row['n']
                assert row['falls'] == sum(e['t_fall'] is not None for e in row['episodes'])
            if row['cond'] == 'moving_turn':
                assert row['quick'] is False and {e['seed'] for e in row['episodes']} == set(range(7300, 7304))
                assert all(e['load_samples'] == 10000 and len(e['joints']) == 2 for e in row['episodes'])
            if run == 'l80_t3' and row['gate'] != 'switch' and key(row) in old:
                assert row['episodes'] == old[key(row)]['episodes'], key(row)
        judged_turns = [r for r in full if r['cond'] == 'moving_turn' and r['cmd'] != 6.]
        records = []
        for row in full+extra:
            record = {k: row[k] for k in ('gate', 'cond', 'cmd', 'yaw_cmd', 'dir', 'dv', 'mount', 'n', 'falls') if k in row}
            record.update(passed=judge(row)[0], reason=judge(row)[1])
            if row['cond'] == 'moving_turn':
                record['passing_episodes'] = sum(turn_episode_ok(row, e) for e in row['episodes'])
                for k in ('v_mean', 'yaw_rate_mean', 'radius_m', 't_90_s'):
                    values = [e[k] for e in row['episodes'] if e[k] is not None]
                    record[k] = [min(values), max(values)] if values else None
            records.append(record)
        exit_codes = {}
        for part in ('full', 'extra'):
            path = OUT/f'{run}_{part}.jsonl'
            with path.with_suffix('.judge.txt').open('w') as out:
                proc = subprocess.run([sys.executable, str(ROOT/'sim/rl/gate_judge.py'), str(path)], stdout=out, check=False)
            exit_codes[part] = proc.returncode
        results[run] = dict(full_suite=dict(zip(('passed', 'reason'), turn_suite_status(full))),
            judged_turn_passed=sum(turn_episode_ok(r, e) for r in judged_turns for e in r['episodes']),
            judged_turn_total=24, judged_turn_falls=sum(r['falls'] for r in judged_turns),
            sustained_command=sustained_speed(extra), judgments=records, judge_exit_codes=exit_codes)
    for a, b in zip(sorted((r for r in read(OUT/'l80_t3_full.jsonl') if r['gate'] in ('stand', 'walk')), key=key),
                    sorted((r for r in read(OUT/'l80_r1_6M_full.jsonl') if r['gate'] in ('stand', 'walk')), key=key)):
        assert a == b
    routed = read(OUT/'routed_turn.jsonl')
    assert len(routed) == 4 and sum(r['n'] for r in routed) == 16
    assert all(e['policy_switches'] == 1 for r in routed for e in r['episodes'])
    with (OUT/'routed_turn.judge.txt').open('w') as out:
        subprocess.run([sys.executable, str(ROOT/'sim/rl/gate_judge.py'), str(OUT/'routed_turn.jsonl')], stdout=out, check=False)
    results['routed'] = dict(passed=sum(turn_episode_ok(r, e) for r in routed for e in r['episodes']),
                             n=16, falls=sum(r['falls'] for r in routed), decision=json.loads((OUT/'routed_decision.json').read_text()))
    compatibility = json.loads((OUT/'runtime_compatibility.json').read_text())
    assert len(compatibility) == 20 and all(r['exact'] for r in compatibility)
    results['verification'] = dict(candidate_rollouts=352, routed_rollouts=16, compatibility_rollouts=40,
                                   inputs_unchanged=True, default_t3_full_non_switch_episodes_identical=True,
                                   equal_scale_switch_episodes_identical=16, default_run_episodes_identical=4)
    (OUT/'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'judgments'} for k, v in results.items()}, indent=2))


if __name__ == '__main__':
    main()
