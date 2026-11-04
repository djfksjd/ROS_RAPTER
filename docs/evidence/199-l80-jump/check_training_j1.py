"""j1 <=8M contract: fixed quick jump seeds every block; no automatic adoption/extension.

Called by train_run only. Exit 10 stops; any error aborts training (fail closed).
"""
import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import stage_gates as sg
from gate_judge import judge, jump_episode_ok, jump_suite_status

E = Path(__file__).resolve().parent
BLOCK = 999_424
MAX_STEPS = BLOCK*8


def metrics(rows):
    episodes = [e for r in rows if r['cond'] in ('in_place', 'in_place_dr') for e in r['episodes']]
    if len(episodes) != 6 or {e['seed'] for e in episodes} != {9300,9301,9310,9311,9312,9313}:
        raise ValueError('incomplete or changed quick seed set')
    if any(not math.isfinite(e['progress']) for e in episodes):
        raise ValueError('nonfinite progress')
    return dict(passed=sum(jump_episode_ok(e) for e in episodes), total=6,
                progress=sum(e['progress'] for e in episodes)/6, falls=sum(e['t_fall'] is not None for e in episodes))


def stop_reason(block, lengths, history, current, eligible=False):
    if not lengths or any(not math.isfinite(v) for v in lengths):
        return 'stop: missing/nonfinite episode lengths'
    if len(lengths) >= 50 and sum(lengths[-10:]) < .5*sum(lengths[:10]):
        return 'stop: episode length below half the first ten updates'
    if eligible:
        return 'stop: full 5a + policy regressions passed; supervisor adoption only'
    if block >= 4 and current['passed'] == 0:
        return 'stop: zero quick successes at/after 4M'
    if block >= 3:
        previous = history[-2]
        if current['passed'] <= previous['passed'] and current['progress']+1e-12 < previous['progress']+.05:
            return 'stop: no success-count gain or >=0.05 progress gain over two blocks'
    if block >= 8:
        return 'stop: 8M budget exhausted; no extension'
    return 'continue'


def write_rows(path, rows):
    result = []
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush(); result.append(row)
    return result


def judge_text(path, rows, full=False):
    lines = [f'{judge(r)[0]} {r["gate"]} {r["cond"]}: {judge(r)[1]}' for r in rows]
    if full:
        ok, why = jump_suite_status(rows); lines.append(f'5a suite {ok}: {why}')
    else:
        lines.append('QUICK DEVELOPMENT ONLY; never adoption')
    path.write_text('\n'.join(lines)+'\n')


def main():
    run, steps = Path(sys.argv[1]), int(sys.argv[2])
    from launch_j1 import verify_inputs, plan
    verify_inputs()
    actual = json.loads((run/'args.json').read_text())
    for key, value in plan().items():
        if actual.get(key) != value:
            raise ValueError('predeclared run option changed: '+key)
    history_path = run/'jump_checks.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    block = len(history)+1
    if not 1 <= block <= 8 or steps != block*BLOCK:
        raise ValueError('unexpected interval/budget; no automatic extension')
    with (run/'progress.csv').open() as f:
        log = list(csv.DictReader(f))
    lengths = [float(r['rollout/ep_len_mean']) for r in log if r.get('rollout/ep_len_mean')]
    for row in log:
        for key, value in row.items():
            if value and (key.startswith('train/') or key == 'rollout/ep_rew_mean') and not math.isfinite(float(value)):
                raise ValueError('nonfinite training statistic: '+key)
    # Do not spend evaluation time on a collapsed or invalid run.
    early = stop_reason(1, lengths, [], dict(passed=0, progress=0.))
    if early != 'continue':
        print(early, flush=True); return 10
    snap = run.with_name(f'{run.name}_{block}M'); snap.mkdir()
    for name in ('model.zip', 'vecnorm.pkl', 'args.json', 'progress.csv', 'initial-policy.json'):
        shutil.copy2(run/name, snap/name)
    out = E/'eval'/run.name; out.mkdir(parents=True, exist_ok=True)
    traces = out/f'traces_{block}M'
    walk, runner = ROOT/'sim/rl/runs/l80_w2', ROOT/'sim/rl/runs/l80_r1_6M'
    quick = write_rows(out/f'quick_{block}M.jsonl',
        ({'gate':'jump', 'run':snap.name, **r} for r in sg.jump(snap, True, walk, trace_dir=traces/'quick')))
    judge_text(out/f'quick_{block}M_judge.txt', quick)
    entry = dict(block=block, added_steps=steps, **metrics(quick))
    decision = stop_reason(block, lengths, history, entry)
    if entry['passed'] == 6 or (block == 8 and decision == 'stop: 8M budget exhausted; no extension'):
        full = write_rows(out/f'full_{block}M.jsonl',
            ({'gate':'jump', 'run':snap.name, **r} for r in sg.jump(snap, False, walk, runner, traces/'full')))
        judge_text(out/f'full_{block}M_judge.txt', full, True)
        eligible, why = jump_suite_status(full)
        (out/f'adoption_{block}M.json').write_text(json.dumps(dict(eligible_5a=eligible, reason=why, stage5_passed=False), indent=2)+'\n')
        decision = stop_reason(block, lengths, history, entry, eligible)
    entry['decision'] = decision; history.append(entry)
    history_path.write_text(json.dumps(history, indent=2)+'\n'); print(json.dumps(entry), flush=True)
    return 0 if decision == 'continue' else 10


if __name__ == '__main__': raise SystemExit(main())
