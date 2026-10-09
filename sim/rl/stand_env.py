"""Stand-only skill experiment; inherits RunEnv mechanics and observations.

The command is always zero. Reward replacement does not make this a moving or
braking controller, nor does a training return constitute a catalogue pass.
"""
import numpy as np
from run_env import RunEnv, CONTROL_DT


def stand_reward_terms(displacement, velocity, tilt, angular_velocity, height_error,
                       flight, unloaded_fraction, heat, action_change):
    return {
        'alive': 2.,
        'position': -40.*float(np.dot(displacement, displacement)),
        'speed': -4.*float(np.dot(velocity[:2], velocity[:2])),
        'tilt': -20.*float(tilt)**2,
        'angular_speed': -.1*float(np.dot(angular_velocity, angular_velocity)),
        'height': -.5*(float(height_error)/.1)**2,
        'flight': -2.*float(flight),
        'unloaded': -.3*float(unloaded_fraction),
        'heat': -.2*float(np.mean(np.maximum(np.asarray(heat)-1., 0.))),
        'action_change': -.01*float(np.dot(action_change, action_change)),
    }


class StandEnv(RunEnv):
    def __init__(self, *, fall_penalty=10., **kwargs):
        if not np.isfinite(fall_penalty) or fall_penalty <= 0:
            raise ValueError('fall penalty must be finite and positive')
        self.fall_penalty = float(fall_penalty)
        super().__init__(**kwargs)

    def _sample_command(self):
        return np.zeros(3)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.resample_steps = 0
        self.stand_origin = self.data.xpos[self.base][:2].copy()
        self.stand_height = float(self.data.xpos[self.base][2])
        return observation, info

    def step(self, action):
        if np.any(self.command != 0.):
            raise ValueError('StandEnv accepts only zero commands')
        previous_action = self.last_action.copy()
        obs, _, terminated, truncated, info = super().step(action)
        displacement = self.data.xpos[self.base][:2]-self.stand_origin
        angular_velocity, _ = self.body_velocity(self.base)
        terms = stand_reward_terms(
            displacement, info['v_body'], info['tilt'], angular_velocity,
            self.data.xpos[self.base][2]-self.stand_height, info['flight'],
            sum(not x for x in info['loaded'].values())/2.,
            self.heat_inst[self.policy_idx], self.last_action-previous_action)
        reward = CONTROL_DT*sum(terms.values())-(self.fall_penalty if terminated else 0.)
        info = {**info, 'stand_terms': terms,
                'stand_displacement_m': float(np.linalg.norm(displacement))}
        return obs, float(reward), terminated, truncated, info
