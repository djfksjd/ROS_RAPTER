import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/rl'))
try:from command_sequence import sequence_command,CommandSequenceEnv,SequenceControlEnv
except ImportError:CommandSequenceEnv=None

@unittest.skipIf(CommandSequenceEnv is None,'simulator dependencies unavailable')
class CommandSequenceTest(unittest.TestCase):
    def test_extended_turn_changes_only_yaw_window_and_preserves_horizons(self):
        for t in (0.,3.98,4.,5.98,6.,9.98,10.,11.98,12.):
            old,horizon=sequence_command('turn',-1.,4.,t)
            new,new_horizon=sequence_command('turn',-1.,4.,t,turn_seconds=6.)
            np.testing.assert_array_equal(old[:2],new[:2])
            self.assertEqual(horizon,new_horizon)
            self.assertEqual(new[2],-1. if 4.<=t<10. else 0.)
            for kind in ('stand','stop','straight'):
                a,ha=sequence_command(kind,-1.,7.,t)
                b,hb=sequence_command(kind,-1.,7.,t,turn_seconds=6.)
                np.testing.assert_array_equal(a,b);self.assertEqual(ha,hb)
        for invalid in (0.,-1.,9.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):sequence_command('turn',1.,4.,0.,invalid)

    def test_extended_turn_environment_observation_at_original_turn_end(self):
        env=CommandSequenceEnv(forced_kind='turn',turn_seconds=6.,randomize=False)
        try:
            env.reset(seed=0);env.data.time=5.98
            obs,_,_,_,info=env.step(np.zeros(env.action_space.shape))
            self.assertEqual(info['command'][2],env.sequence_yaw)
            self.assertEqual(obs[8],env.sequence_yaw*.5)
            self.assertEqual(env.sequence_duration,12.)
            env.data.time=9.98
            obs,_,_,_,info=env.step(np.zeros(env.action_space.shape))
            self.assertEqual(info['command'][2],env.sequence_yaw)
            self.assertEqual(obs[8],0.)
        finally:env.close()
    def test_turn_and_stop_boundaries_and_actual_zero_commands(self):
        for t,yaw in [(3.98,0),(4.,1),(5.98,1),(6.,0)]:
            command,duration=sequence_command('turn',1,4,t)
            np.testing.assert_array_equal(command,[4,0,yaw]);self.assertEqual(duration,12)
        np.testing.assert_array_equal(sequence_command('stop',1,7,7.98)[0],[4,0,0])
        np.testing.assert_array_equal(sequence_command('stop',1,7,8.)[0],[0,0,0])
        np.testing.assert_array_equal(sequence_command('stand',1,7,0)[0],[0,0,0])
        with self.assertRaises(ValueError):sequence_command('bad',1,4,0)

    def test_control_retains_random_commands_and_matched_horizon(self):
        from run_env import RunEnv
        for zero_cmd in [0.,1.]:
            env=SequenceControlEnv(forced_kind='stand',randomize=False,zero_cmd=zero_cmd)
            baseline=RunEnv(randomize=False,zero_cmd=zero_cmd)
            env.reset(seed=0);baseline.reset(seed=0)
            np.testing.assert_array_equal(env.command,baseline.command)
            if zero_cmd==1.:np.testing.assert_array_equal(env.command,[0,0,0])
            else:self.assertGreater(env.command[0],0.)
            self.assertEqual(env.sequence_duration,21.)
            self.assertEqual(env.resample_steps,250)
            _,_,_,_,info=env.step(np.zeros(env.action_space.shape))
            self.assertEqual(info['sequence_kind'],'control:stand')
            baseline.close();env.close()

    def test_reset_true_stand_and_returned_observation_current_command(self):
        env=CommandSequenceEnv(forced_kind='stand',randomize=False)
        obs,_=env.reset(seed=0)
        np.testing.assert_array_equal(obs[6:9],[0,0,0]);self.assertEqual(env.data.qvel[0],0.)
        env.close()
        env=CommandSequenceEnv(forced_kind='turn',randomize=False)
        env.reset(seed=0);env.data.time=3.98
        obs,_,_,_,info=env.step(np.zeros(env.action_space.shape))
        self.assertAlmostEqual(env.data.time,4.)
        np.testing.assert_allclose(obs[6:9],[.4,0,env.sequence_yaw*.5])
        self.assertEqual(info['command'][2],0.)
        self.assertEqual(env.resample_steps,0)
        env.close()

if __name__=='__main__':unittest.main()
