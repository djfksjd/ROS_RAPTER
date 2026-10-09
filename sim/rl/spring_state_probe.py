"""Synthetic hidden-spring-state counterfactual, on two isolated simulation copies.

Tests whether an omitted field matters, not whether altered states are reachable
in normal rollouts or whether it causes the task failures.
"""
import argparse,json
from pathlib import Path
import numpy as np
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load


def probe(run,yaw,seed):
    kw=env_kwargs(json.loads((Path(run)/'args.json').read_text()))
    envs=[RunEnv(**{**kw,'episode_s':15.,'level':0.},randomize=False,seed=7) for _ in range(2)]
    a,b=envs
    for e in envs:e.init_seed=seed
    model,norm=load(str(Path(run)/'model.zip'),a)
    try:
        norm.reset();b.reset()
        for e in envs:e.command=np.array([4.,0.,0.]);e.resample_steps=0
        for _ in range(250):
            t=round(float(a.data.time),12)
            for e in envs:e.command[2]=yaw if t>=4. else 0.
            action,_=model.predict(norm.normalize_obs(a._obs()[None]),deterministic=True)
            results=[e.step(action[0]) for e in envs]
            if any(r[2] for r in results):raise RuntimeError('warmup fell')
        np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
        np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
        np.testing.assert_array_equal(a.model.tendon_stiffness,b.model.tendon_stiffness)
        np.testing.assert_array_equal(a.model.tendon_lengthspring,b.model.tendon_lengthspring)
        np.testing.assert_array_equal(a._obs(),b._obs())
        before=dict(loaded={k:bool(v) for k,v in a.loaded.items()},qpos=a.data.qpos.tolist(),qvel=a.data.qvel.tolist(),
                    stiffness=a.model.tendon_stiffness.tolist(),rest_lengths=a.model.tendon_lengthspring.tolist(),obs=a._obs().tolist())
        b.loaded['left']=not b.loaded['left']
        obs_equal=bool(np.array_equal(a._obs(),b._obs()));assert obs_equal
        actions=[model.predict(norm.normalize_obs(e._obs()[None]),deterministic=True)[0][0] for e in envs]
        np.testing.assert_array_equal(*actions)
        info=[e.step(action)[4] for e,action in zip(envs,actions)]
        return dict(run=run,yaw=yaw,seed=seed,warmup_seconds=5.,changed_field="loaded['left']",synthetic_counterfactual=True,
                    reachable_state_proven=False,pre_observation_identical=obs_equal,pre_action_identical=True,before=before,
                    altered_loaded_before_step=not before['loaded']['left'],
                    max_next_qpos_difference=float(np.abs(a.data.qpos-b.data.qpos).max()),
                    max_next_qvel_difference=float(np.abs(a.data.qvel-b.data.qvel).max()),
                    after_stiffness=[e.model.tendon_stiffness.tolist() for e in envs],
                    after_rest_lengths=[e.model.tendon_lengthspring.tolist() for e in envs],
                    after_tilt=[float(r['tilt']) for r in info],after_qpos=[e.data.qpos.tolist() for e in envs],
                    after_qvel=[e.data.qvel.tolist() for e in envs])
    finally:norm.close();b.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=46001);a=p.parse_args();out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True)
    with out.open('x') as f:
        for yaw in [-1.,0.,1.]:
            r=probe(a.run,yaw,a.seed);f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
            print(json.dumps({k:v for k,v in r.items() if k not in ['before','after_qpos','after_qvel','after_rest_lengths']}),flush=True)

if __name__=='__main__':main()
