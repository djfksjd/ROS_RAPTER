"""Evaluate stand-only skill with the existing catalogue gate, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate import load
from stand_env import StandEnv
from train_run import env_kwargs
from motion_transition import stand_gate


def trial(run, seed, seconds):
    run=Path(run)
    config=json.loads((run/'args.json').read_text())
    if not config.get('stand_task'):
        raise ValueError('a stand-task run is required')
    env=StandEnv(**{**env_kwargs(config), 'episode_s':seconds+5., 'level':0.},
                 randomize=False, seed=7)
    env.init_seed=seed
    policy,norm=load(str(run/'model.zip'),env)
    rows=[]
    try:
        obs=norm.reset()
        for i in range(round(seconds/.02)):
            before=env.data.xpos[env.base][:2].copy()
            action,_=policy.predict(obs,deterministic=True)
            raw,_,fell,_,info=env.step(action[0])
            rows.append(dict(t=(i+1)*.02,position_xy=env.data.xpos[env.base][:2].tolist(),
                             before_position_xy=before.tolist(),tilt=float(info['tilt']),
                             flight=bool(info['flight']),speed_xy=float(np.linalg.norm(info['v_body'][:2])),
                             height=float(env.data.xpos[env.base][2]),heat=env.heat_inst.tolist()))
            obs=norm.normalize_obs(raw[None])
            if fell:
                break
        steady=np.mean([r['heat'] for r in rows[-2000:]],axis=0)
        heat=float(np.max(steady[env.motor_cont<1e8]))
        gate=stand_gate(rows)
        complete=len(rows)==round(seconds/.02)
        passed=complete and not fell and gate['pass_gate'] and heat<=1.
        return dict(run=str(run),seed=seed,requested_seconds=seconds,seconds=rows[-1]['t'],
                    fell=bool(fell),stand_gate=gate,max_steady_heat=heat,
                    screen_pass=bool(passed),catalogue_pass=bool(passed and seconds>=60.),
                    physical_temperature=False,rows=rows,
                    hashes={n:hashlib.sha256((run/n).read_bytes()).hexdigest()
                            for n in ['model.zip','vecnorm.pkl','args.json']})
    finally:
        norm.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<.02:
        parser.error('finite duration of at least one control step required')
    with Path(args.out).open('x') as stream:
        for seed in args.seeds:
            result=trial(args.run,seed,args.seconds)
            stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
            print({k:v for k,v in result.items() if k not in ['rows','hashes']},flush=True)


if __name__=='__main__':
    main()
