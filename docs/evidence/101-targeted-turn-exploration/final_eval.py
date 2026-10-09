import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'sim/rl'))
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
from evaluate_run import episode,turn_pass
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,nargs='+',default=list(range(37001,37006)));a=p.parse_args()
kw=env_kwargs(json.loads((Path(a.run)/'args.json').read_text()))
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('x') as f:
 for speed,yaw in [(4.,None),(7.,None),(4.,-1.),(4.,-.5),(4.,.5),(4.,1.)]:
  for seed in a.seeds:
   env=RunEnv(**{**kw,'level':0.,'episode_s':15.},randomize=False,seed=7);env.init_seed=seed
   model,norm=load(str(Path(a.run)/'model.zip'),env)
   try:
    r=episode(model,norm,env,[speed,0.,0.],10.,turn=(yaw,4.,2.) if yaw is not None else None)
    record=dict(run=a.run,initial_seed=seed,command=[speed,0.,0.],turn=[yaw,4.,2.] if yaw is not None else None,result=r,turn_pass_20pct=turn_pass(r,2*yaw) if yaw is not None else None)
    f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
   finally:norm.close()
