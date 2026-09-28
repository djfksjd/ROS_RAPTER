"""Gazebo-like position path for the MuJoCo Raptor model.

Mirrors gz_ros2_control 1.2.20: every physics step the velocity command is
-gain*(q_read - q_cmd)*update_rate, where q_read is refreshed only every control
period. The actuator ctrlrange clips it to the URDF velocity limit (like DART SERVO
setCommand). DART's separate velocity-limit constraint on externally driven motion is
not reproduced. STOP holds the last read positions; no learned motor commands here.
"""
import numpy as np
import mujoco

ACTIVE = [f'{s}_{j}_joint' for s in ('left', 'right')
          for j in ('hip_roll', 'hip_pitch', 'knee_pitch', 'ankle_pitch')] + ['tail_yaw_joint', 'tail_pitch_joint']


class GazeboLikeServo:
    def __init__(self, model, gain=.3, update_rate=100.):
        self.model = model
        self.gain, self.update_rate = gain, update_rate
        self.period_steps = max(1, round(1/(update_rate*model.opt.timestep)))
        self.qpos = np.array([model.jnt_qposadr[model.joint(n).id] for n in ACTIVE])
        self.actuators = np.array([model.actuator(n).id for n in ACTIVE])
        self.read = None
        self.command = None
        self.steps = 0

    def set_target(self, positions):
        """positions: dict joint->rad for a subset; others keep the previous command."""
        if self.command is None:
            raise RuntimeError('call step() once before set_target()')
        for name, value in positions.items():
            self.command[ACTIVE.index(name)] = value

    def stop(self):
        self.command = self.read.copy()

    def step(self, data):
        if self.steps % self.period_steps == 0:
            self.read = data.qpos[self.qpos].copy()
            if self.command is None:
                self.command = self.read.copy()
        data.ctrl[self.actuators] = -self.gain*(self.read-self.command)*self.update_rate
        mujoco.mj_step(self.model, data)
        self.steps += 1
