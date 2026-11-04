"""j2 removes anti-flight costs only, preserving the policy/physics and all gates."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'docs/evidence/199-l80-jump'
sys.path.insert(0,str(E));sys.path.insert(0,str(ROOT/'sim/rl'))
from jump_task import JumpTask
from run_env import RunEnv
from train_run import budget_chunk
from launch_j2 import plan
from check_training_j2 import stop_reason,MAX_STEPS,BLOCK


class JumpRemedyTest(unittest.TestCase):
    def test_default_off_and_active_landing_success_fall_boundaries(self):
        original=dict(stand=1.,stand_support=.5,hold=.2,slip_lin=.3,action_rate=.4,gait=.1)
        weights={k:-5. for k in original}
        for config in ({},{'suspend_stand_penalties':False},{'suspend_stand_penalties':True}):
            task=JumpTask(config)
            before=original.copy();task.adjust_stand_terms(before,weights);self.assertEqual(before,original)
            task.request_time=2.
            active=original.copy();task.adjust_stand_terms(active,weights)
            expected={**original,'stand':0.,'stand_support':0.} if config.get('suspend_stand_penalties') else original
            self.assertEqual(active,expected)
            for attr,value in (('landed',{'touchdown_s':2.4}),('success_time',3.4),('fall_time',2.5)):
                setattr(task,attr,value)
                restored=original.copy();task.adjust_stand_terms(restored,weights);self.assertEqual(restored,original)
                setattr(task,attr,None)

    def test_positive_weights_untouched_and_boolean_required(self):
        task=JumpTask({'suspend_stand_penalties':True});task.request_time=2.
        terms={'stand':1.,'stand_support':1.};task.adjust_stand_terms(terms,{'stand':5.,'stand_support':0.})
        self.assertEqual(terms,{'stand':1.,'stand_support':1.})
        for invalid in ('false',1,None):
            with self.assertRaises(ValueError):JumpTask({'suspend_stand_penalties':invalid})

    def test_real_env_same_physics_only_two_costs_removed(self):
        kw=dict(arch8=True,model_rev=2,real_mass=True,randomize=False,seed=3,
                weights={'stand':-5.,'stand_support':-5.,'hold':-3.})
        a=RunEnv(**kw,jump={'auto_request_s':None})
        b=RunEnv(**kw,jump={'auto_request_s':None,'suspend_stand_penalties':True})
        changed=False
        try:
            a.reset(seed=3);b.reset(seed=3)
            # Identical accepted external requests; the test isolates reward from request readiness.
            for env in (a,b):
                env.command=np.zeros(3);env.resample_steps=0
                env.jump_task.request_time=0.;env.jump_task.standing_height=float(env.data.xpos[env.base,2])
                # A shared airborne fixture exercises the anti-flight costs, without learning.
                env.data.qpos[2]+=.1;mujoco.mj_forward(env.model,env.data)
            for k in range(100):
                action=np.zeros(a.action_space.shape)
                ra,rb=a.step(action),b.step(action)
                np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
                np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
                np.testing.assert_array_equal(ra[0],rb[0]);self.assertEqual(ra[2:4],rb[2:4])
                self.assertEqual(ra[4]['jump'],rb[4]['jump'])
                delta=0.
                for key,value in ra[4]['terms'].items():
                    other=rb[4]['terms'][key]
                    if key in ('stand','stand_support'):
                        delta+=.02*a.weights[key]*(other-value)
                    else:self.assertEqual(value,other,key)
                self.assertGreaterEqual(delta,0.);self.assertAlmostEqual(rb[1]-ra[1],delta)
                changed |= delta>0
                if ra[2] or ra[3]:break
            self.assertTrue(changed,'trajectory must actually exercise the removed cost')
        finally:a.close();b.close()

    def test_only_task_conditional_mask_changes_in_launch_plan(self):
        old=json.loads((ROOT/'sim/rl/runs/l80_j1/args.json').read_text());new=plan()
        changed={k for k in old if old[k]!=new[k]}
        self.assertEqual(changed,{'name','steps','init','checkpoint_check','jump'})
        self.assertEqual(json.loads(new['jump']),{**json.loads(old['jump']),'suspend_stand_penalties':True})
        self.assertIsNone(new['resume_action_std']);self.assertEqual(new['action_lpf'],6.)

    def test_budget_stagnation_and_full_gate_priority(self):
        done=0;chunks=[]
        while n:=budget_chunk(4_000_000-done,8192):done+=n;chunks.append(n)
        self.assertEqual(chunks,[BLOCK]*4);self.assertEqual(done,MAX_STEPS)
        history=[dict(passed=0,progress=0.),dict(passed=0,progress=.02)]
        self.assertIn('no success-count gain',stop_reason(3,[600.]*60,history,dict(passed=0,progress=.049)))
        self.assertEqual(stop_reason(3,[600.]*60,history,dict(passed=0,progress=.05)),'continue')
        self.assertIn('zero quick',stop_reason(4,[600.]*60,history,dict(passed=0,progress=1.)))
        self.assertIn('4M budget',stop_reason(4,[600.]*60,history,dict(passed=1,progress=.3)))
        self.assertIn('full 5a',stop_reason(4,[600.]*60,history,dict(passed=6,progress=1.),True))
        self.assertIn('below half',stop_reason(2,[600.]*40+[200.]*10,history,dict(passed=6,progress=1.),True))
        self.assertIn('missing/nonfinite',stop_reason(1,[],[],dict(passed=0,progress=0.)))


if __name__=='__main__':unittest.main()
