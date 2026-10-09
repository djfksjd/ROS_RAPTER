import sys,subprocess,json
from pathlib import Path
kind=sys.argv[1]
run={'control':'sim/rl/runs/turn_duration2_control_20261009','candidate':'sim/rl/runs/turn_duration6_candidate_20261009'}[kind]
base=f'sim/rl/runs/screen_logs/turn_duration_final.{kind}'
assert json.loads((Path(run)/'completion.json').read_text())['final_steps']==11841536
commands=[
 [sys.executable,'/tmp/raptor_dense_final_eval.py',run,'--out',base+'.eval.jsonl','--seeds','61001','61002','61003','61004','61005'],
 [sys.executable,'sim/rl/heat_test.py',run,'--cmds','4','7','--seconds','60','--init-seeds','62001','62002','62003'],
 [sys.executable,'sim/rl/motion_transition.py',run,'--out',base+'.motion.jsonl','--seeds','63001','63002']]
for index,cmd in enumerate(commands):
 log=Path(base+['.eval.log','.heat.jsonl','.motion.log'][index])
 with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
print(json.dumps(dict(kind=kind,status='all_validation_commands_completed')))
