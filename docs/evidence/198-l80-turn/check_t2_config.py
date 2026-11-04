"""Validate the supervisor's yaw-scale recipe without executing the shell or learn()."""
import contextlib
import io
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'sim/rl'))
import numpy as np
import torch
from evaluate import load
from run_env import RunEnv
from train_run import budget_chunk, env_kwargs, widen_command_inputs, rescale_action_outputs

EVIDENCE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    old = (EVIDENCE/'train_l80_t1.sh').read_text()
    new = (EVIDENCE/'train_l80_t2.sh').read_text()
    expected = old.replace('NAME=${NAME:-l80_t1}', 'NAME=${NAME:-l80_t2}').replace(
        '--steps 8000000', '--steps 4000000').replace(
        '--init sim/rl/runs/l80_r1_6M/model.zip', '--init sim/rl/runs/l80_t1/model.zip').replace(
        "--scale-override '{\"hip_roll\": 0.5}'", "--scale-override '{\"hip_roll\": 0.5, \"hip_yaw\": 0.5}'")
    assert new == expected
    argv = shlex.split(new[new.index('exec "$PY"'):].replace('\\\n', ' '))
    source = ROOT/argv[argv.index('--init')+1]
    args = json.loads(source.with_name('args.json').read_text())
    weights = json.loads(shlex.split(next(line[2:] for line in new.splitlines() if line.startswith('W=')))[0])
    assert weights == json.loads(args['weights'])
    assert json.loads((EVIDENCE/'commands.json').read_text()) == json.loads(args['cmd_mix'])
    scale_override = argv[argv.index('--scale-override')+1]
    env = RunEnv(**env_kwargs({**args, 'scale_override': scale_override}), randomize=False, seed=0)
    model, vn = load(str(source), env)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            widened = widen_command_inputs(vn, model, args['cmd_final'])
            factors = rescale_action_outputs(model, env.active, env.policy_idx, args, env.scale, .15)
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(.15)))
        yaw = [j for j, i in enumerate(env.policy_idx) if 'hip_yaw' in env.active[i]]
        assert yaw == [1, 5] and len(env.policy_idx) == 10 and not widened
        np.testing.assert_allclose(factors[yaw], .6)
        np.testing.assert_allclose(model.policy.log_std.detach().exp().numpy(), .15)
        env.reset(seed=0)
        np.testing.assert_allclose(env.lo[env.policy_idx[yaw]], -.5)
        np.testing.assert_allclose(env.hi[env.policy_idx[yaw]], .5)
        budgets, done = [], 0
        limit = int(argv[argv.index('--steps')+1])
        while n := budget_chunk(limit-done, 8192):
            budgets.append(n); done += n
        assert done == 3_997_696 and len(budgets) == 4
        result = dict(script_changes_only=['name', 'init', 'hip_yaw scale', 'step budget'],
            source=str(source.relative_to(ROOT)), scale_override=json.loads(scale_override),
            commands_equal_t1=True, weights_equal_t1=True, policy_dof=len(env.policy_idx),
            yaw_action_indices=yaw, factors=factors.tolist(), rescale_log=captured.getvalue().splitlines(),
            resume_action_std=.15, std8_after_widen=float(np.sqrt(vn.obs_rms.var[8]+vn.epsilon)),
            blocks=budgets, actual_budget=done, training_started=False)
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        vn.close()


if __name__ == '__main__':
    main()
