"""Validate t4fallback (not started, fallback) without executing the shell or calling learn()."""
import contextlib
import io
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
from evaluate import load
from run_env import RunEnv
from train_run import budget_chunk, env_kwargs, widen_command_inputs

EVIDENCE = Path(__file__).resolve().parent


def main():
    old = (EVIDENCE/'train_l80_t1.sh').read_text()
    new = (EVIDENCE/'train_l80_t4fallback.sh').read_text()
    assert new.replace('# NOT STARTED, FALLBACK (command-distribution-only plan).\n', '') == old.replace('NAME=${NAME:-l80_t1}', 'NAME=${NAME:-l80_t4fallback}').replace(
        'commands.json', 'commands_t4fallback.json').replace('--steps 8000000', '--steps 4000000')
    argv = shlex.split(new[new.index('exec "$PY"'):].replace('\\\n', ' '))
    weights = json.loads(shlex.split(next(line[2:] for line in new.splitlines() if line.startswith('W=')))[0])
    original_args = json.loads((ROOT/'sim/rl/runs/l80_t1/args.json').read_text())
    assert weights == json.loads(original_args['weights'])
    mix = json.loads((EVIDENCE/'commands_t4fallback.json').read_text())
    args = {**original_args, 'cmd_mix': json.dumps(mix)}
    env = RunEnv(**env_kwargs(args), randomize=False, seed=0)
    env.reset(seed=0)
    assert len(env.policy_idx) == 10
    assert float(env.model.opt.gravity[2]) == -9.81
    source = ROOT/argv[argv.index('--init')+1]
    model, vn = load(str(source), env)
    with contextlib.redirect_stdout(io.StringIO()):
        widen_command_inputs(vn, model, original_args['cmd_final'])
    std8 = float((vn.obs_rms.var[8]+vn.epsilon)**.5)
    assert std8 == .375
    budgets, done = [], 0
    limit = int(argv[argv.index('--steps')+1])
    while n := budget_chunk(limit-done, 8192):
        budgets.append(n); done += n
    assert done == 3_997_696 and len(budgets) == 4
    weight = sum(r[3] for r in mix)
    groups = dict(gate=sum(r[3] for r in mix if (r[0], abs(r[2])) in ((2, 1), (4, .5), (6, .5))),
                  gentle=sum(r[3] for r in mix if r[0] > 0 and abs(r[2]) == .25),
                  straight=sum(r[3] for r in mix if r[0] > 0 and r[2] == 0),
                  stop=sum(r[3] for r in mix if r[0] == r[2] == 0),
                  in_place=sum(r[3] for r in mix if r[0] == 0 and r[2] != 0))
    assert groups == dict(gate=60, gentle=20, straight=15, stop=2.5, in_place=2.5)
    result = dict(script_changes_only=['name', 'command mixture', 'step budget'], source=str(source.relative_to(ROOT)),
                  all_reward_weights=env.weights, weights_equal_t1=True, policy_dof=len(env.policy_idx),
                  std8_after_widen=std8, command_shares={k: v/weight for k, v in groups.items()},
                  blocks=budgets, actual_budget=done, training_started=False)
    vn.close()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
