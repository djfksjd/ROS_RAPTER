"""ROS-free runner for an exported RL locomotion policy (numpy only; no torch in the ROS image).

The policy file (.npz, from sim/rl/export_policy.py) holds the actor MLP (ELU), the observation
normalizer and the action contract (joint order, default pose q0, per-joint action scale). The
observation matches sim/rl/raptor_env.py exactly:
  [gyro*0.25, gravity in body frame, command*(2, 2, 0.5), q-q0, qd*0.05, last action, sin/cos(2*pi*phase)]
Outputs position targets only; STOP and all safety checks stay outside this class.
"""
import math

import numpy as np


class PolicyRunner:
    def __init__(self, path):
        f = np.load(path)
        self.layers = [(f[f'w{i}'], f[f'b{i}']) for i in range(int(f['n_layers']))]
        self.mean, self.var, self.clip = f['obs_mean'], f['obs_var'], float(f['obs_clip'])
        self.joints = [str(j) for j in f['joints']]
        self.q0, self.scale = f['q0'], f['action_scale']
        self.lo, self.hi = f['joint_lo'], f['joint_hi']
        self.clock_hz, self.dt = float(f['clock_hz']), float(f['control_dt'])
        self.last_action = np.zeros(len(self.joints))
        self.phase = 0.

    @staticmethod
    def gravity_in_body(quat_wxyz):
        w, x, y, z = quat_wxyz
        # third row of the body->world rotation, i.e. R^T @ (0, 0, -1)
        return -np.array([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])

    def observation(self, gyro, quat_wxyz, command, q, qd):
        return np.concatenate([np.asarray(gyro)*.25, self.gravity_in_body(quat_wxyz),
                               np.asarray(command)*np.array([2., 2., .5]), np.asarray(q)-self.q0,
                               np.asarray(qd)*.05, self.last_action,
                               [math.sin(2*math.pi*self.phase), math.cos(2*math.pi*self.phase)]])

    def act(self, obs):
        """Deterministic action in [-1, 1] for a raw observation; advances the gait clock."""
        x = np.clip((obs-self.mean)/np.sqrt(self.var+1e-8), -self.clip, self.clip)
        for i, (w, b) in enumerate(self.layers):
            x = w @ x+b
            if i < len(self.layers)-1:
                x = np.where(x > 0, x, np.expm1(x))  # ELU
        action = np.clip(x, -1., 1.)
        self.last_action = action
        self.phase = (self.phase+self.dt*self.clock_hz) % 1.
        return action

    def targets(self, action):
        return np.clip(self.q0+self.scale*action, self.lo, self.hi)
