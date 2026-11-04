"""Decision-B short remedy: two <=1M blocks; never launches training or adopts.

Called only by train_run --checkpoint-check. Exit 10 stops, all other errors
fail closed. The original check_training.py contract remains separate.
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
from gate_judge import judge, turn_episode_ok, turn_suite_status

EVIDENCE = Path(__file__).resolve().parent
MAX_STEPS = 1_998_848


def metrics(rows):
    turns = [(r, e) for r in rows if r['cond'] == 'moving_turn' and r['cmd'] in (2., 4.) for e in r['episodes']]
    straight = [(r, e) for r in rows if r['cond'] == 'run_regression' for e in r['episodes']]
    return dict(turn_passed=sum(turn_episode_ok(r, e) for r, e in turns), turn_total=len(turns),
                straight_passed=sum(judge({**r, 'n': 1, 'falls': int(e['t_fall'] is not None), 'episodes': [e]})[0]
                                    for r, e in straight), straight_total=len(straight))


def stop_reason(steps, lengths, eligible=False):
    if any(not math.isfinite(x) for x in lengths):
        return 'stop: nonfinite episode length'
    if len(lengths) >= 50 and sum(lengths[-10:]) < .5*sum(lengths[:10]):
        return 'stop: episode length below half the initial ten-update mean'
    if eligible:
        return 'stop: adjusted full turn + regressions passed; candidate only'
    if steps >= MAX_STEPS:
        return 'stop: 2M budget exhausted; no extension'
    return 'continue'


def write_rows(path, rows):
    measured = []
    with path.open('x') as out:
        for row in rows:
            out.write(json.dumps(row, allow_nan=False)+'\n'); out.flush()
            measured.append(row)
    return measured


def development(run, walk):
    for row in sg.turn(run, quick=True, walk_dir=walk):
        yield {'gate': 'turn', 'run': run.name, **row}
    for cmd in (2., 4., 6.):
        episodes = []
        for seed in (7200, 7201):
            fell, v, der, hot = sg.run_rollout(run, seed, cmd, 20.)
            episodes.append(dict(seed=seed, t_fall=fell, v_mean=float(v[500:].mean()) if fell is None else None,
                                 min_derate=der, t_winding_max_c=hot))
        yield dict(gate='turn', run=run.name, cond='run_regression', quick=True, cmd=cmd, n=2,
                   falls=sum(e['t_fall'] is not None for e in episodes), episodes=episodes)


def main():
    run, steps = Path(sys.argv[1]), int(sys.argv[2])
    history_path = run/'turn_checks_b.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    block = len(history)+1
    if block not in (1, 2) or steps != block*999_424:
        raise ValueError('unexpected check interval or budget')
    snap = run.with_name(f'{run.name}_{block}M')
    snap.mkdir()  # refuse overwrite
    for name in ('model.zip', 'vecnorm.pkl', 'args.json', 'progress.csv', 'initial-policy.json'):
        shutil.copy2(run/name, snap/name)
    out = EVIDENCE/'eval'/run.name
    out.mkdir(parents=True, exist_ok=True)
    with (run/'progress.csv').open() as f:
        lengths = [float(r['rollout/ep_len_mean']) for r in csv.DictReader(f) if r.get('rollout/ep_len_mean')]
    decision = stop_reason(steps, lengths)
    entry = dict(block=block, added_steps=steps, decision=decision)
    # Collapse/invalid-log stop takes precedence over expensive evaluation.
    if decision in ('continue', 'stop: 2M budget exhausted; no extension'):
        walk = ROOT/'sim/rl/runs/l80_w2'
        rows = write_rows(out/f'development_{block}M.jsonl', development(snap, walk))
        entry.update(metrics(rows))
        judged = [r for r in rows if not (r['cond'] == 'moving_turn' and r['cmd'] == 6.)]
        if block == 2 or all(judge(r)[0] is True for r in judged):
            full = write_rows(out/f'full_{block}M.jsonl',
                ({'gate': 'turn', 'run': snap.name, **r} for r in sg.turn(snap, walk_dir=walk)))
            ok, why = turn_suite_status(full)
            ok = ok and all(judge(r)[0] is not False for r in full)
            (out/f'adoption_{block}M.json').write_text(json.dumps(dict(eligible=ok, reason=why), indent=2)+'\n')
            entry['decision'] = stop_reason(steps, lengths, eligible=ok)
    history.append(entry)
    history_path.write_text(json.dumps(history, indent=2)+'\n')
    print(json.dumps(entry), flush=True)
    return 0 if entry['decision'] == 'continue' else 10


if __name__ == '__main__':
    raise SystemExit(main())
