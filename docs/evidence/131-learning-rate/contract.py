from pathlib import Path
import sys,json,subprocess,tempfile
import numpy as np
import torch
sys.path.insert(0,str(Path('sim/rl').resolve()))
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from symmetric_policy import SymmetricPolicy
base=['.venv-sim/bin/python','sim/rl/train_symmetric.py','sim/rl/runs/sym_init_20261009']
for value in ['0','-1','nan','inf']:
 r=subprocess.run(base+['--name','lr_invalid_never_created','--learning-rate',value],capture_output=True,text=True);assert r.returncode==2 and 'learning rate must be finite and positive' in r.stderr;assert not Path('sim/rl/runs/lr_invalid_never_created').exists();print('invalid rejected',value,flush=True)
models=[]
for name,extra,want in [('lr_default_contract_20261009',[],.0003),('lr_low_contract_20261009',['--learning-rate','3e-5'],.00003)]:
 subprocess.run(base+['--name',name,'--initialize-only','--target-kl','.005']+extra,check=True)
 p=Path('sim/rl/runs')/name;m=PPO.load(p/'model.zip',device='cpu');assert json.loads((p/'args.json').read_text())['learning_rate']==want;assert m.lr_schedule(1.)==want
 with tempfile.TemporaryDirectory() as d:m.set_logger(configure(d,[]));m._update_learning_rate(m.policy.optimizer)
 assert all(g['lr']==want for g in m.policy.optimizer.param_groups);models.append(m)
x,y=[m.policy.state_dict() for m in models]
for k in x:assert torch.equal(x[k],y[k]),k
obs=np.random.default_rng(11).normal(size=(100,46)).astype(np.float32);a,_=models[0].predict(obs,deterministic=True);b,_=models[1].predict(obs,deterministic=True);np.testing.assert_array_equal(a,b)
print('24 tensors and100 predictions equal; saved rate, lr_schedule and actual optimizer LR update verified',flush=True)
