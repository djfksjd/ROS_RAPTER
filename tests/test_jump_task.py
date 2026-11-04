"""Stage 5a temporal boundaries and opt-in command/reward contracts."""
import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim/rl'))
from jump_task import JumpTask
from run_env import RunEnv
from mirror_probe import ObservationReflection


def frame(t, off=False, z=1., x=0., stable=True):
    return dict(t=t, xyz=np.array([x,0.,z]), velocity=np.zeros(3) if stable else np.array([1.,0.,0.]),
                tilt=0., ground_z=0., body_hit=False, feet_n=np.zeros(2,int) if off else np.ones(2,int),
                feet_force_n=np.zeros(2) if off else np.array([50.,50.]), torque_nm=np.zeros(2),
                current_a=np.zeros(2), current_limit_a=np.ones(2))


class JumpTaskTest(unittest.TestCase):
    def ready(self, config=None):
        task=JumpTask(config or {});self.t=0
        self.feed(task,2000)
        self.assertEqual(task.command,1.)
        return task

    def feed(self,task,n,**kw):
        for _ in range(n):
            self.t+=1;task.sample(frame(self.t*.001,**kw),1.,100.)

    def test_success_needs_same_flight_and_contiguous_hold(self):
        task=self.ready();self.feed(task,100,off=True,z=1.06)
        self.feed(task,500,x=.3);self.assertIsNone(task.success_time)
        self.feed(task,1,x=.3,stable=False);self.feed(task,999,x=.3)
        self.assertIsNone(task.success_time)
        self.feed(task,1,x=.3);self.assertIsNotNone(task.success_time)
        self.assertAlmostEqual(task.consume_bonus(),40.)
        self.feed(task,100);self.assertEqual(task.consume_bonus(),0.)
        self.assertEqual(task.command,0.)

    def test_short_intervals_do_not_add_to_flight(self):
        task=self.ready()
        for _ in range(3):self.feed(task,50,off=True,z=1.06);self.feed(task,100)
        self.feed(task,1000);self.assertIsNone(task.success_time)
        self.assertLessEqual(task.consume_bonus(),10.)

    def test_height_is_from_standing_not_crouch(self):
        task=self.ready();self.feed(task,100,z=.85)
        self.feed(task,100,off=True,z=1.03);self.feed(task,1000)
        self.assertIsNone(task.success_time)
        self.assertAlmostEqual(task.result()['apex_rise_m'],.03)

    def test_landing_error_and_one_foot_contact(self):
        task=self.ready();self.feed(task,100,off=True,z=1.06);self.feed(task,1200,x=.30001)
        self.assertIsNone(task.success_time)
        task=self.ready()
        for _ in range(100):
            self.t+=1;s=frame(self.t*.001,off=True,z=1.06);s['feet_n'][1]=1
            task.sample(s,1.,100.)
        self.feed(task,1000);self.assertIsNone(task.success_time)

    def test_no_height_duration_from_different_flights(self):
        task=self.ready();self.feed(task,50,off=True,z=1.10);self.feed(task,10)
        self.feed(task,150,off=True,z=1.02);self.feed(task,1000)
        self.assertIsNone(task.success_time)

    def test_external_request_and_unstable_start(self):
        self.t=0;task=JumpTask({'auto_request_s':None});self.feed(task,2500)
        self.assertEqual(task.command,0.);task.request();self.feed(task,1);self.assertEqual(task.command,1.)
        task=JumpTask({});self.t=0;self.feed(task,2500,stable=False);self.assertEqual(task.command,0.)

    def test_fall_blocks_success_and_negative_bonus_rejected(self):
        task=self.ready();s=frame(2.001);s['body_hit']=True;task.sample(s,1.,100.)
        self.feed(task,100,off=True,z=1.06);self.feed(task,1000)
        self.assertIsNone(task.success_time);self.assertIsNotNone(task.fall_time)
        with self.assertRaises(ValueError):JumpTask({'flight_bonus':-1})
        with self.assertRaises(ValueError):JumpTask({'success_bonus':float('nan')})

    def test_optional_input_is_appended_reflected_and_physics_passive(self):
        kw=dict(arch8=True,model_rev=2,real_mass=True,seed=3,randomize=False,obs_hold={'noise':0.,'walk':0.})
        a,b=RunEnv(**kw),RunEnv(**kw,jump={'auto_request_s':None})
        try:
            x,_=a.reset(seed=3);y,_=b.reset(seed=3)
            np.testing.assert_array_equal(x,y[:-1]);self.assertEqual(y[-1],0.)
            r=ObservationReflection(b.active,b.policy_idx,b.q0,b.obs_vel,b.obs_contact,len(b.scan_dx),True,True)
            y[-4:]=[.2,.3,.4,1.];m=r.observation(y)
            np.testing.assert_allclose(m[-4:],[.2,-.3,-.4,1.]);np.testing.assert_array_equal(r.observation(m),y)
            for _ in range(15):
                aa=np.zeros(a.action_space.shape);ra,rb=a.step(aa),b.step(aa)
                np.testing.assert_array_equal(ra[0],rb[0][:-1]);self.assertEqual(ra[1:4],rb[1:4])
                np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
        finally:a.close();b.close()

if __name__=='__main__':unittest.main()
