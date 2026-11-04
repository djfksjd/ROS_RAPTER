"""Read-only comparison against merged PR #47; run from the repository root.

Sixteen equal-scale switches and four default runs, each before/after (40 rollouts).
The revision is pinned so merging this change cannot turn the test into self-comparison.
"""
import ast, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0,str(Path.cwd()/'sim/rl'))
import stage_gates as sg
old=subprocess.check_output(['git','show','e0ac1059306b7e7c336017004b6e5aa7cf3da364:sim/rl/stage_gates.py'],text=True)
namespace=dict(vars(sg))
for name in ('switch_rollout','run_rollout'):
 node=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<HEAD '+name+'>','exec'),namespace)
root=Path('sim/rl/runs'); results=[]
for randomize,accel,n in ((False,None,4),(False,2.,4),(True,2.,8)):
 for i in range(n):
  args=(root/'l80_w2',root/'l80_r1_6M',7100+i,randomize)
  before=namespace['switch_rollout'](*args,accel=accel)
  after=sg.switch_rollout(*args,accel=accel)
  assert before==after,(args,before,after)
  results.append(dict(kind='equal_scale_switch',seed=7100+i,randomize=randomize,accel=accel,exact=True,result=after))
  print('verified',len(results),flush=True)
for run in ('l80_r1_6M','l80_t3'):
 for seed,speed in ((6700,2.),(6701,4.)):
  args=(root/run,seed,speed,20.)
  before=namespace['run_rollout'](*args);after=sg.run_rollout(*args)
  assert before[0]==after[0] and before[2:]==after[2:]
  np.testing.assert_array_equal(before[1],after[1])
  results.append(dict(kind='default_run',run=run,seed=seed,cmd=speed,exact=True))
Path('docs/evidence/198-l80-turn/eval/decision_b/runtime_compatibility.json').write_text(json.dumps(results,indent=2)+'\n')
