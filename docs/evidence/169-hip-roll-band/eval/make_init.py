"""Warm start for a wider hip-roll action band: rev2_rz1 re-expressed for scale_override {'hip_roll': NEW} so that the
initial policy commands exactly the same joint targets. Hip-roll action outputs (mean rows/bias and log_std) are scaled by
OLD/NEW, and the frozen normalisation of the matching last_action observation entries (VecNormalize and the SymPolicy
buffers) by the same factor, so the network sees identical inputs. argv: src_dir dst_dir new_scale."""
import json, sys
from pathlib import Path
import numpy as np
import torch as th
sys.path.insert(0, 'sim/rl')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from run_env import RunEnv, SCALE_RUN
from train_run import env_kwargs
src, dst, new = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
f = SCALE_RUN['hip_roll']/new
dst.mkdir(parents=True, exist_ok=True)
a = json.loads((src/'args.json').read_text())
env = RunEnv(**env_kwargs(a), randomize=False, seed=0); env.reset(seed=0)
hr = [i for i, j in enumerate(np.array(env.active)[env.policy_idx]) if 'hip_roll' in j]  # action indices
# observation entries that carry last_action: perturb it and diff the observation
env.last_action = np.zeros(len(env.policy_idx)); o0 = env._obs().copy()
obs_hr = []
for k in hr:
    la = np.zeros(len(env.policy_idx)); la[k] = 1.; env.last_action = la
    d = np.flatnonzero(np.abs(env._obs()-o0) > 1e-9); assert len(d) == 1, d; obs_hr.append(int(d[0]))
print('hip roll actions', hr, 'last_action obs', obs_hr, 'factor', round(f, 4))
model = PPO.load(str(src/'model.zip'), device='cpu')
pol = model.policy
with th.no_grad():
    pol.action_net.weight[hr] *= f; pol.action_net.bias[hr] *= f
    pol.log_std[hr] += float(np.log(f))
    if hasattr(pol, 'sym_mean'):
        for k in obs_hr:
            pol.sym_mean[k] *= f; pol.sym_std[k] *= f
            pol.sym_kwargs['mean'][k] *= f; pol.sym_kwargs['std'][k] *= f
model.save(str(dst/'model.zip'))
vn = VecNormalize.load(str(src/'vecnorm.pkl'), DummyVecEnv([lambda: env]))
vn.obs_rms.mean[obs_hr] *= f; vn.obs_rms.var[obs_hr] *= f*f
vn.save(str(dst/'vecnorm.pkl'))
a['scale_override'] = json.dumps({'hip_roll': new}); a['name'] = dst.name
(dst/'args.json').write_text(json.dumps(a, indent=1))
