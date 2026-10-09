import subprocess,json,sys
from pathlib import Path
sys.path.insert(0,str(Path('sim/rl').resolve()))
from stable_baselines3 import PPO
from symmetric_policy import SymmetricPolicy
import torch
base=['.venv-sim/bin/python','sim/rl/train_symmetric.py','sim/rl/runs/sym_init_20261009']
for value in ['0','-1','nan','inf']:
 r=subprocess.run(base+['--name','kl_invalid_never_created','--target-kl',value],capture_output=True,text=True)
 assert r.returncode==2 and 'target KL must be finite and positive' in r.stderr
 assert not Path('sim/rl/runs/kl_invalid_never_created').exists()
 print('invalid value rejected:',value,flush=True)
models=[]
for name,extra,want in [('kl_default_contract_20261009',[],None),('kl_guard_contract_20261009',['--target-kl','.005'],.005)]:
 subprocess.run(base+['--name',name,'--initialize-only']+extra,check=True)
 p=Path('sim/rl/runs')/name
 m=PPO.load(p/'model.zip',device='cpu');assert m.target_kl==want
 assert json.loads((p/'args.json').read_text())['target_kl']==want
 models.append(m)
a=models[0].policy.state_dict();b=models[1].policy.state_dict();assert a.keys()==b.keys()
for k in a:assert torch.equal(a[k],b[k]),k
assert models[0].policy.optimizer.state_dict()['param_groups']==models[1].policy.optimizer.state_dict()['param_groups']
print('default None / .005 serialization verified; initial policy tensors equal:',len(a),flush=True)
