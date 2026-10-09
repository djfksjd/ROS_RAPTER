"""Bounded simulation screen of inference averaging; saved policies are untouched."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate import load
from evaluate_run import episode, turn_pass
from run_env import RunEnv
from train_run import env_kwargs
from mirror_probe import ObservationReflection, symmetric_action
from yaw_feedback import YawCommandFeedback, heading_rate

class RawPolicyAdapter:
    """Existing evaluator calls predict(normalized_obs); read its current raw state.

    Deliberately avoid unnormalizing clipped observations, which loses information.
    This adapter is confined to this standalone simulation experiment.
    """
    def __init__(self, model, norm, env, yaw_scale=1., feedback_gain=0., trace=False, correction_limit=None, heading_feedback=False):
        self.model, self.norm, self.env = model, norm, env
        if not np.isfinite(yaw_scale) or yaw_scale <= 0:
            raise ValueError("yaw scale must be finite and positive")
        self.yaw_scale = yaw_scale
        self.feedback = YawCommandFeedback(scale=yaw_scale,gain=feedback_gain,correction_limit=correction_limit)
        self.trace = []
        self.keep_trace = trace
        self.heading_feedback=heading_feedback
        self.reflection = ObservationReflection(env.active, env.policy_idx, env.q0, env.obs_vel,getattr(env,'obs_contact',False))

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError('diagnostic adapter requires deterministic evaluation')
        raw=self.env._obs().copy()
        desired=float(raw[8])*2.
        angular,linear=self.env.body_velocity(self.env.base) if self.feedback.gain or self.keep_trace else (np.zeros(3),np.zeros(3))
        heading=heading_rate(self.env.data.xmat[self.env.base],angular) if self.feedback.gain or self.keep_trace else 0.
        measured=heading if self.heading_feedback else float(angular[2])
        internal=self.feedback.update(desired,measured)
        if self.feedback.gain:
            raw[8] = internal*.5
        else:
            raw[8] *= self.yaw_scale
        if self.feedback.gain or self.keep_trace:
            self.trace.append(dict(t=float(self.env.data.time),desired_yaw=desired,
                                   measured_yaw=measured,body_yaw=float(angular[2]),heading_rate=heading,filtered_yaw=self.feedback.filtered_rate,
                                   internal_yaw=internal,vx=float(linear[0]),
                                   tilt=float(np.arccos(np.clip(-raw[5],-1,1)))))
        return symmetric_action(self.model,self.norm,self.reflection,raw)[None],None


def run(run_dir, mode, speed, yaw, seed, yaw_scale=1., feedback_gain=0., correction_limit=None, heading_feedback=False):
    run_dir=Path(run_dir)
    args=json.loads((run_dir/'args.json').read_text())
    env=RunEnv(**{**env_kwargs(args),'level':0.,'episode_s':15.},randomize=False,seed=7)
    env.init_seed=seed
    model,norm=load(str(run_dir/'model.zip'),env)
    try:
        policy=RawPolicyAdapter(model,norm,env,yaw_scale,feedback_gain,trace=True,correction_limit=correction_limit,heading_feedback=heading_feedback) if mode=='averaged' else model
        turn=(yaw,4.,2.) if yaw is not None else None
        result=episode(policy,norm,env,[speed,0.,0.],10.,turn=turn)
        return dict(mode=mode,run=str(run_dir),speed_cmd=speed,yaw_cmd=yaw,initial_seed=seed,
                    result=result,turn_pass_20pct=turn_pass(result,2*yaw) if turn else None,
                    saved_policy_modified=False, internal_yaw_scale=yaw_scale if mode=="averaged" else 1.,
                    feedback_gain=feedback_gain if mode=="averaged" else 0., correction_limit=correction_limit, heading_feedback=heading_feedback,
                    trace=policy.trace if mode=="averaged" else None)
    finally:norm.close()


def heat_run(run_dir,mode,speed,seed,yaw_scale,seconds,feedback_gain=0., correction_limit=None, heading_feedback=False):
    """Same 1 ms motor-load measure as heat_test; retain unrounded verdict inputs."""
    run_dir=Path(run_dir);args=json.loads((run_dir/'args.json').read_text())
    env=RunEnv(**{**env_kwargs(args),'level':0.,'episode_s':seconds+5},randomize=False,seed=7)
    env.init_seed=seed;model,norm=load(str(run_dir/'model.zip'),env)
    try:
        norm.reset();env.command=np.array([speed,0.,0.]);env.resample_steps=0
        policy=RawPolicyAdapter(model,norm,env,yaw_scale,feedback_gain,correction_limit=correction_limit,heading_feedback=heading_feedback) if mode=='averaged' else model
        velocities=[];heat=[];fell=False
        for _ in range(int(seconds/.02)):
            raw=env._obs();action,_=policy.predict(norm.normalize_obs(raw[None]),deterministic=True)
            _,_,fell,_,_=env.step(action[0])
            velocities.append(float(env.body_velocity(env.base)[1][0]));heat.append(env.heat_inst.copy())
            if fell:break
        tail=slice(max(0,len(heat)-2000),len(heat))
        steady=np.mean(np.asarray(heat)[tail],axis=0)
        max_heat=float(np.max(steady[env.motor_cont<1e8]))
        velocity=float(np.mean(velocities[tail]))
        completed=len(velocities)==int(seconds/.02)
        return dict(mode=mode,run=str(run_dir),initial_seed=seed,command_vx=speed,
                    internal_yaw_scale=yaw_scale if mode=='averaged' else 1.,seconds=len(heat)*.02,
                    fell=bool(fell),completed=completed,steady_heat_each_motor=steady.tolist(),
                    max_heat=max_heat,speed_last40_mps=velocity,
                    sustainable=completed and not fell and max_heat<=1.,
                    tracked_10pct=completed and not fell and abs(velocity-speed)<=.1*speed,
                    physical_temperature=False,feedback_gain=feedback_gain if mode=="averaged" else 0.,correction_limit=correction_limit,heading_feedback=heading_feedback)
    finally:norm.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir');p.add_argument('--out',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[15001,15002])
    p.add_argument('--yaw-scale',type=float,default=1.)
    p.add_argument('--heat-seconds',type=float)
    p.add_argument('--feedback-gain',type=float,default=0.)
    p.add_argument('--correction-limit',type=float)
    p.add_argument('--heading-feedback',action='store_true')
    p.add_argument('--modes',choices=['original','averaged'],nargs='+',default=['original','averaged'])
    a=p.parse_args()
    if not np.isfinite(a.yaw_scale) or a.yaw_scale<=0:p.error('yaw scale must be finite and positive')
    if not np.isfinite(a.feedback_gain) or a.feedback_gain<0:p.error('feedback gain must be finite and nonnegative')
    if a.heat_seconds is not None and (not np.isfinite(a.heat_seconds) or a.heat_seconds<60):
        p.error('heat measurement requires at least 60 seconds')
    if a.correction_limit is not None and (not np.isfinite(a.correction_limit) or a.correction_limit<=0):
        p.error("correction limit must be finite and positive")
    path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        for mode in a.modes:
            for speed,yaw in ([(4.,None),(7.,None)] if a.heat_seconds else [(4.,None),(7.,None),(4.,-1.),(4.,-.5),(4.,.5),(4.,1.)]):
                for seed in a.seeds:
                    result=(heat_run(a.run_dir,mode,speed,seed,a.yaw_scale,a.heat_seconds,a.feedback_gain,a.correction_limit,a.heading_feedback) if a.heat_seconds
                            else run(a.run_dir,mode,speed,yaw,seed,a.yaw_scale,a.feedback_gain,a.correction_limit,a.heading_feedback))
                    f.write(json.dumps(result)+'\n');f.flush();print(json.dumps(result),flush=True)

if __name__=='__main__':main()
