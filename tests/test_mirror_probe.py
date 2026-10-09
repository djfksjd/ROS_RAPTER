import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'sim/rl'))
from mirror_probe import ObservationReflection, joint_reflection, symmetric_action

class ReflectionTest(unittest.TestCase):
    def make(self, velocity=True):
        names=['left_hip_roll_joint','left_hip_pitch_joint','right_hip_roll_joint',
               'right_hip_pitch_joint','tail_yaw_joint','tail_pitch_joint']
        # Deliberately asymmetric reference poses exercise the q-q0 offset.
        return ObservationReflection(names, [0,1,2,3,4,5], [.1,.2,.3,.4,.5,.6], velocity)

    def test_involution_and_input_preservation(self):
        for velocity in [False,True]:
            mirror=self.make(velocity); raw=np.arange(mirror.size,dtype=float)/100
            original=raw.copy()
            np.testing.assert_allclose(mirror.observation(mirror.observation(raw)),raw,atol=1e-14)
            np.testing.assert_array_equal(raw,original)
            action=np.arange(6,dtype=float)
            np.testing.assert_array_equal(mirror.action(mirror.action(action)),action)

    def test_vectors_clock_and_absolute_joint_positions(self):
        mirror=self.make();raw=np.zeros(mirror.size);raw[:12]=1;raw[-2:]=[.6,.8]
        actual=mirror.observation(raw)
        np.testing.assert_array_equal(actual[:12],[-1,1,-1,1,-1,1,1,-1,-1,1,-1,1])
        np.testing.assert_allclose(actual[12:18],[-.4,.2,-.4,-.2,-1.,0.])
        np.testing.assert_array_equal(actual[-2:],[-.6,-.8])
        np.testing.assert_array_equal(mirror.action([1,2,3,4,5,6]),[-3,4,-1,2,-5,6])

    def test_averaged_policy_equivariance_with_asymmetric_normalization(self):
        mirror=self.make();raw=np.arange(mirror.size,dtype=float)/10
        class Norm:
            def normalize_obs(self,x):return np.tanh((x-np.arange(mirror.size)/20)/2)
        class Policy:
            def predict(self,x,deterministic):
                return np.tanh(x[:,:6]+x[:,-6:]),None
        a=symmetric_action(Policy(),Norm(),mirror,raw)
        b=symmetric_action(Policy(),Norm(),mirror,mirror.observation(raw))
        np.testing.assert_allclose(a,mirror.action(b),atol=1e-14)

    def test_invalid_mappings_and_dimensions_rejected(self):
        for names in [['left_hip_roll_joint'],['mystery_pitch_joint'],['tail_yaw_joint']*2]:
            with self.assertRaises(ValueError):joint_reflection(names)
        with self.assertRaises(ValueError):self.make().observation(np.zeros(5))
        with self.assertRaises(ValueError):self.make().action(np.zeros(3))

if __name__=='__main__':unittest.main()
