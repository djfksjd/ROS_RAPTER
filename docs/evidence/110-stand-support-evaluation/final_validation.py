import sys,subprocess,json
from pathlib import Path
kind=sys.argv[1]
run=f'sim/rl/runs/stand_support_{kind}_20261009'
base=f'sim/rl/runs/screen_logs/stand_support_final.{kind}'
assert json.loads((Path(run)/'completion.json').read_text())['final_steps']==11841536
commands=[
 [sys.executable,'/tmp/raptor_dense_final_eval.py',run,'--out',base+'.eval.jsonl','--seeds','54001','54002','54003','54004','54005'],
 [sys.executable,'sim/rl/heat_test.py',run,'--cmds','4','7','--seconds','60','--init-seeds','55001','55002','55003'],
 [sys.executable,'sim/rl/motion_transition.py',run,'--out',base+'.motion.jsonl','--seeds','56001','56002']]
for index,cmd in enumerate(commands):
 log=Path(base+['.eval.log','.heat.jsonl','.motion.log'][index])
 with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
print(json.dumps(dict(kind=kind,status='all_validation_commands_completed')))
