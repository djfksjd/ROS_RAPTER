from pathlib import Path
import sys,subprocess,json
import numpy as np
import torch
sys.path.insert(0,str(Path('sim/rl').resolve()))
from stable_baselines3 import PPO
from symmetric_policy import SymmetricPolicy
base=['.venv-sim/bin/python','sim/rl/train_symmetric.py','sim/rl/runs/sym_init_20261009','--initialize-only','--target-kl','.005']
models=[]
for name,extra in [('optimizer_transfer_contract_20261009',[]),('optimizer_fresh_contract_20261009',['--reset-optimizer'])]:
 subprocess.run(base+['--name',name]+extra,check=True)
 p=Path('sim/rl/runs')/name;m=PPO.load(p/'model.zip',device='cpu');models.append(m)
 assert json.loads((p/'args.json').read_text())['reset_optimizer']==bool(extra)
a,b=models;x=a.policy.state_dict();y=b.policy.state_dict();assert x.keys()==y.keys()
for k in x:assert torch.equal(x[k],y[k]),k
assert len(a.policy.optimizer.state)>0 and len(b.policy.optimizer.state)==0
obs=np.random.default_rng(51).normal(size=(100,46)).astype(np.float32)
aa,_=a.predict(obs,deterministic=True);bb,_=b.predict(obs,deterministic=True);np.testing.assert_array_equal(aa,bb)
print('24 policy tensors equal;100-state deterministic actions exactly equal; default optimizer populated, fresh optimizer empty; saved flags verified')
