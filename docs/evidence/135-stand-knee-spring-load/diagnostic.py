from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('sim/rl').resolve()))
from pose_hold import probe
source=Path('sim/rl/runs/kl005_lr3e5_update64k_20261009')
args=json.loads((source/'args.json').read_text())
out=Path('sim/rl/runs/screen_logs/stand_knee_spring_ablation.diagnostic.jsonl')
with out.open('x') as f:
 for stiffness in [170.,0.]:
  temp=Path('/tmp/raptor_stand_knee_'+str(int(stiffness)));temp.mkdir(exist_ok=True)
  cfg=dict(args);cfg['springs_override']=json.dumps({'knee_pitch':[stiffness,.956,'bi','latch']});(temp/'args.json').write_text(json.dumps(cfg))
  for seed in [22001,22002]:
   r=probe(temp,0.,0.,seed,10.);r['source']=str(source);r['knee_stiffness']=stiffness
   f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
   print({k:v for k,v in r.items() if k not in ['rows','final_qpos']},flush=True)
