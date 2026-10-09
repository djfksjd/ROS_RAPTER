import sys,subprocess,json
from pathlib import Path
kind=sys.argv[1];run=f'sim/rl/runs/dense_speed_{kind}_20261009';base=f'sim/rl/runs/screen_logs/dense_speed_final.{kind}'
assert json.loads((Path(run)/'completion.json').read_text())['final_steps']==11841536
commands=[
 [sys.executable,'/tmp/raptor_dense_final_eval.py',run,'--out',base+'.eval.jsonl'],
 [sys.executable,'sim/rl/heat_test.py',run,'--cmds','4','7','--seconds','60','--init-seeds','38001','38002','38003'],
 [sys.executable,'sim/rl/motion_transition.py',run,'--out',base+'.motion.jsonl','--seeds','39001','39002']]
for index,cmd in enumerate(commands):
 log=Path(base+['.eval.log','.heat.jsonl','.motion.log'][index])
 with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
print(json.dumps(dict(kind=kind,status='all_validation_commands_completed')))
