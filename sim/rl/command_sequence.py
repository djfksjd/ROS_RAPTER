"""Catalogue-shaped training commands; unchanged RunEnv physics and action space.

The inherited sampler already zeros all components when vx is zero. This
curriculum changes their timing/duration: full holds, yaw steps and run-to-stop.
"""
import numpy as np
from run_env import RunEnv

KINDS=('turn','straight','stop','stand')
PROBABILITIES=(.4,.2,.2,.2)


def sequence_command(kind,yaw,speed,t,turn_seconds=2.):
    if kind not in KINDS:raise ValueError('unknown command sequence')
    if not np.isfinite(t) or t<0:raise ValueError('time must be finite and nonnegative')
    if not np.isfinite(turn_seconds) or not 0<turn_seconds<=8.:raise ValueError('turn seconds must be in (0,8]')
    if kind=='stand':return np.zeros(3),21.
    if kind=='stop':return np.array([4. if t<8. else 0.,0.,0.]),21.
    if kind=='straight':return np.array([speed,0.,0.]),12.
    return np.array([4.,0.,yaw if 4.<=t<4.+turn_seconds else 0.]),12.


class CommandSequenceEnv(RunEnv):
    def __init__(self,seed=0,forced_kind=None,turn_seconds=2.,**kwargs):
        if forced_kind is not None and forced_kind not in KINDS:raise ValueError('unknown forced kind')
        sequence_command('turn',0.,4.,0.,turn_seconds)
        self.turn_seconds=float(turn_seconds)
        self.forced_kind=forced_kind;self.sequence_rng=np.random.default_rng(seed+100000)
        self.sequence_kind='stand';self.sequence_yaw=0.;self.sequence_speed=4.
        super().__init__(seed=seed,**{**kwargs,'episode_s':22.})

    def _sample_command(self):
        # Retain the inherited RNG draws. Initial speed still uses the inherited
        # rule, now based on this sequence's initial vx rather than a random vx.
        super()._sample_command()
        return sequence_command(self.sequence_kind,self.sequence_yaw,self.sequence_speed,0.,self.turn_seconds)[0]

    def _set_command(self):
        self.command,self.sequence_duration=sequence_command(self.sequence_kind,self.sequence_yaw,
                                                            self.sequence_speed,round(float(self.data.time),12),self.turn_seconds)
        self.resample_steps=0

    def reset(self,seed=None,options=None):
        if seed is not None:self.sequence_rng=np.random.default_rng(seed+100000)
        self.sequence_kind=self.forced_kind or str(self.sequence_rng.choice(KINDS,p=PROBABILITIES))
        self.sequence_yaw=float(self.sequence_rng.choice([-1.,-.5,.5,1.]))
        self.sequence_speed=float(self.sequence_rng.choice([4.,7.]))
        _,info=super().reset(seed=seed,options=options)
        self._set_command()
        return self._observe(),info

    def step(self,action):
        self._set_command()
        _,reward,terminated,truncated,info=super().step(action)
        self._set_command()
        info={**info,'sequence_kind':self.sequence_kind}
        truncated=truncated or round(float(self.data.time),12)>=self.sequence_duration
        return self._observe(),reward,terminated,bool(truncated),info


class SequenceControlEnv(CommandSequenceEnv):
    """Matched episode kinds/durations, retaining the inherited random commands."""
    def _sample_command(self):
        return RunEnv._sample_command(self)

    def _set_command(self):
        self.sequence_duration=sequence_command(self.sequence_kind,self.sequence_yaw,self.sequence_speed,0.)[1]
        self.resample_steps=250

    def step(self,action):
        obs,reward,terminated,truncated,info=super().step(action)
        return obs,reward,terminated,truncated,{**info,'sequence_kind':'control:'+self.sequence_kind}
