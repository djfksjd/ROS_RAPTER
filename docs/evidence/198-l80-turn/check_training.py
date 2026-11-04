"""Called by train_run --checkpoint-check. Exit 10 stops; never adopts a policy automatically."""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import stage_gates as sg
from gate_judge import turn_suite_status, turn_episode_ok
from turn_eval import turn_condition

EVIDENCE = Path(__file__).resolve().parent


def write_rows(path, rows):
    out = []
    with path.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n'); f.flush()
            out.append(row)
    return out


def metrics(rows):
    episodes = [(r, e) for r in rows if r['gate'] == 'turn' and r['cond'] == 'moving_turn' and r['cmd'] >= 2
                for e in r['episodes']]
    # Preserve the original t1/t2/t3 training contract, including 6 m/s, after decision B.
    passed = sum(turn_episode_ok(r, e) for r, e in episodes)
    error = sum(abs(e['yaw_rate_mean']-r['yaw_cmd']) if e['yaw_rate_mean'] is not None else 10.
                for r, e in episodes)/len(episodes)
    return dict(passed=passed, total=len(episodes), yaw_mae=error)


def stop_reason(entry, history, lengths):
    if len(lengths) >= 50 and sum(lengths[-10:])/10 < .5*sum(lengths[:10])/10:
        return 'stop: episode length below half the initial ten-update mean'
    if entry['block'] >= 4 and entry['passed'] == 0:
        return 'stop: zero passing running-turn episodes at/after 4M'
    # Odd checks have 4 running episodes, even checks have 12. Compare the same
    # parity (two blocks back), never adjacent blocks or their raw pass counts.
    prior = next((r for r in history if r['block'] == entry['block']-2), None)
    if prior and entry['passed'] <= prior['passed'] and entry['yaw_mae'] > prior['yaw_mae']-.02:
        return 'stop: no extra passing episode or 0.02 rad/s yaw error improvement over 2M'
    return 'continue'


def main():
    run, steps = Path(sys.argv[1]), int(sys.argv[2])
    state_path = run/'turn_checks.json'
    history = json.loads(state_path.read_text()) if state_path.exists() else []
    block = len(history)+1
    snap = run.with_name(f'{run.name}_{block}M')
    if snap.exists():
        raise RuntimeError(f'refusing to overwrite {snap}')
    snap.mkdir()
    for name in ('model.zip', 'vecnorm.pkl', 'args.json', 'progress.csv', 'initial-policy.json'):
        shutil.copy2(run/name, snap/name)
    eval_dir = EVIDENCE/'eval'/run.name
    eval_dir.mkdir(parents=True, exist_ok=True)
    walk = ROOT/'sim/rl/runs/l80_w2'
    # Every ~1M: fixed development seeds. Every ~2M: the complete quick turn gate.
    if block % 2 == 0:
        rows = ({'gate': 'turn', 'run': snap.name, **r} for r in sg.turn(snap, quick=True, walk_dir=walk))
    else:
        rows = ({'gate': 'turn', 'run': snap.name, 'quick': True,
                 **turn_condition(snap, v, w, [7200])} for v in (4., 6.) for w in (.5, -.5))
    measured = write_rows(eval_dir/f'turn_{block}M.jsonl', rows)
    entry = dict(block=block, added_steps=steps, **metrics(measured), decision='continue')
    with (run/'progress.csv').open() as f:
        lengths = [float(r['rollout/ep_len_mean']) for r in csv.DictReader(f) if r.get('rollout/ep_len_mean')]
    entry['decision'] = stop_reason(entry, history, lengths)
    if block % 2 == 0 and all(r['falls'] == 0 and all(turn_episode_ok(r, e) for e in r['episodes']) for r in measured):
        full = write_rows(eval_dir/f'full_{block}M.jsonl',
                          ({'gate': 'turn', 'run': snap.name, **r} for r in sg.turn(snap, walk_dir=walk)))
        ok, why = turn_suite_status(full)
        # This legacy checker still requires the original 6 m/s held-out turns.
        original_turns_ok = all(turn_episode_ok(r, e) for r in full if r['cond'] == 'moving_turn'
                                for e in r['episodes'])
        if ok and not original_turns_ok:
            ok, why = False, 'original t1/t2/t3 turn criteria (including 6 m/s) failed'
        (eval_dir/f'adoption_{block}M.json').write_text(json.dumps(dict(eligible=ok, reason=why), indent=2)+'\n')
        if ok:
            entry['decision'] = 'stop: full turn + regression passed; eligible snapshot, retained policies unchanged'
    if block >= 8 and entry['decision'] == 'continue':
        entry['decision'] = 'stop: 8M budget exhausted'
    history.append(entry)
    state_path.write_text(json.dumps(history, indent=2)+'\n')
    print(json.dumps(entry), flush=True)
    return 0 if entry['decision'] == 'continue' else 10


if __name__ == '__main__':
    raise SystemExit(main())
