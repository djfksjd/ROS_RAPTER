"""Simulation-only local model around an inverse-dynamics stand reference.

State includes quaternion tangent displacement, velocity and previous target.
It uses full simulator state; it is not an operational sensor-based controller.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
from run_env import RunEnv, LINK_ETA
from train_run import env_kwargs


class LocalStandModel:
    def __init__(self, source, candidate):
        config=json.loads((Path(source)/'args.json').read_text())
        if not config.get('arch8') or config.get('arch12'):
            raise ValueError('eight-axis source required')
        self.env=RunEnv(**{**env_kwargs(config),'episode_s':65.,'level':0.},
                        randomize=False,seed=7)
        self.env.reset()
        self.q=np.asarray(candidate['qpos'],dtype=float).copy()
        self.u=np.asarray(candidate['action'],dtype=float).copy()
        self.nv=self.env.model.nv
        self.na=len(self.env.policy_idx)
        self.nx=2*self.nv+self.na
        self._envelope=self.env._clamp
        if (self.na!=8 or self.q.shape!=(self.env.model.nq,) or self.u.shape!=(8,)
                or not np.isfinite(self.q).all() or not np.isfinite(self.u).all()
                or np.max(np.abs(self.u))>1):
            self.close()
            raise ValueError('invalid stand reference')

    def _vector(self, value, size):
        value=np.asarray(value,dtype=float)
        if value.shape!=(size,) or not np.isfinite(value).all():
            raise ValueError('finite vector with the expected dimension required')
        return value

    def restore(self, x):
        x=self._vector(x,self.nx)
        e=self.env;m,d=e.model,e.data
        e.reset();e.command=np.zeros(3);e.resample_steps=0
        d.qpos[:]=self.q
        mujoco.mj_integratePos(m,d.qpos,x[:self.nv],1.)
        d.qvel[:]=x[self.nv:2*self.nv]
        d.qacc[:]=0.;d.qacc_warmstart[:]=0.
        d.qfrc_applied[:]=0.;d.xfrc_applied[:]=0.
        previous=e.q0.copy()
        previous[e.policy_idx]=e.q0[e.policy_idx]+e.scale[e.policy_idx]*(self.u+x[2*self.nv:])
        e.ctrl_prev=previous.copy();e.last_action=self.u+x[2*self.nv:]
        d.ctrl[e.act]=previous
        e._clamp();mujoco.mj_forward(m,d)

    def state(self):
        e=self.env;dq=np.zeros(self.nv)
        mujoco.mj_differentiatePos(e.model,dq,1.,self.q,e.data.qpos)
        previous=(e.ctrl_prev[e.policy_idx]-e.q0[e.policy_idx])/e.scale[e.policy_idx]-self.u
        return np.r_[dq,e.data.qvel.copy(),previous]

    def transition(self, x, u):
        x=self._vector(x,self.nx);u=self._vector(u,self.na)
        self.restore(x)
        result=self.env.step(self.u+u)
        if result[2] or not all(result[4]['loaded'].values()):
            raise ValueError('transition left the local double-support region')
        return self.state()

    def set_stand_current_limit(self, factor=None):
        if factor is None:
            self.env._clamp=self._envelope
            return
        if not np.isfinite(factor) or not 0<factor<=1:
            raise ValueError('continuous limit factor must be within (0,1]')
        def limit():
            e=self.env;low,high=self._envelope()
            gear=e.gear_const.copy()
            gear[e.fourbar_idx]=np.interp(e.data.qpos[e.q_adr[e.fourbar_idx]],*e.knee_ratio)
            cap=factor*gear*LINK_ETA*e.motor_cont
            low=np.maximum(low,-cap);high=np.minimum(high,cap)
            e.model.actuator_forcerange[e.act,0]=low
            e.model.actuator_forcerange[e.act,1]=high
            return low,high
        self.env._clamp=limit

    def close(self):
        self.env.close()
