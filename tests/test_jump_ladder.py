"""2026-11-05 height ladder and optional landing phase: legacy/default compatibility."""
import copy
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'docs/evidence/199-l80-jump'
sys.path[:0]=[str(ROOT/'sim/rl'),str(E)]
from jump_task import JumpTask
from run_env import RunEnv
from mirror_probe import ObservationReflection
from train_run import validate_jump_resume,budget_chunk
from gate_judge import jump_episode_ok,jump_suite_status,judge
from tests import test_jump_task as helpers
from tests.test_jump_gate import episode,suite
from launch_j4 import plan,INIT
from check_training_j4 import stop_reason,metrics,BLOCK,MAX_STEPS
from verify_j4_init import initialized,compare


class JumpHeightPhaseTest(unittest.TestCase):
    ready=helpers.JumpTaskTest.ready
    feed=helpers.JumpTaskTest.feed

    def test_valid_landing_phase_and_height_not_old_5cm(self):
        task=self.ready({'height_target_m':.2,'obs_phase':True,'success_bonus':50.})
        self.assertEqual(task.phase,0.)
        self.feed(task,100,off=True,z=1.06);self.feed(task,1000)
        self.assertEqual(task.phase,0.);self.assertIsNone(task.success_time)
        self.assertAlmostEqual(task.progress,.3)
        self.feed(task,100,off=True,z=1.200001);self.feed(task,500)
        self.assertEqual(task.phase,1.);self.assertIsNone(task.success_time)
        self.feed(task,20,off=True,z=1.1)  # hold interrupted; phase must remain landing, no second reward
        self.assertEqual(task.phase,1.);self.feed(task,999);self.assertIsNone(task.success_time)
        self.feed(task,1);self.assertIsNotNone(task.success_time)
        self.assertAlmostEqual(task.consume_bonus(),60.)
        task.reset();self.assertEqual(task.phase,0.)

    def test_three_height_levels_and_invalid_configs(self):
        for height in (.05,.2,.3):
            t=self.ready({'height_target_m':height});self.feed(t,100,off=True,z=1.+height-.00001);self.feed(t,1000)
            self.assertIsNone(t.success_time)
            self.feed(t,100,off=True,z=1.+height+.00001);self.feed(t,1000);self.assertIsNotNone(t.success_time)
        for cfg in ({'height_target_m':0},{'height_target_m':float('nan')},{'obs_phase':1}):
            with self.assertRaises(ValueError):JumpTask(cfg)

    def test_phase_is_passive_append_and_reflection_invariant(self):
        kw=dict(arch8=True,model_rev=2,real_mass=True,seed=3,randomize=False,obs_hold={'noise':0.,'walk':0.})
        a=RunEnv(**kw,jump={'auto_request_s':None});b=RunEnv(**kw,jump={'auto_request_s':None,'obs_phase':True})
        try:
            x,_=a.reset(seed=3);y,_=b.reset(seed=3)
            np.testing.assert_array_equal(x,y[:-1]);self.assertEqual(y[-1],0.)
            r=ObservationReflection(b.active,b.policy_idx,b.q0,b.obs_vel,b.obs_contact,len(b.scan_dx),True,True,True)
            y[-5:]=[.2,.3,.4,1.,1.];m=r.observation(y)
            np.testing.assert_allclose(m[-5:],[.2,-.3,-.4,1.,1.]);np.testing.assert_array_equal(r.observation(m),y)
            for _ in range(20):
                action=np.zeros(a.action_space.shape);ra,rb=a.step(action),b.step(action)
                np.testing.assert_array_equal(ra[0],rb[0][:-1]);self.assertEqual(ra[1:4],rb[1:4])
                np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
        finally:a.close();b.close()

    def test_same_height_task_without_phase_has_identical_physics_and_reward(self):
        old=JumpTask({'height_target_m':.2});new=JumpTask({'height_target_m':.2,'obs_phase':True})
        from tests.test_jump_task import frame
        for t in range(1,3101):
            f=frame(t*.001,off=2000<t<=2100,z=1.21 if 2000<t<=2100 else 1.)
            old.sample(f,1.,100.);new.sample(f,1.,100.)
        self.assertEqual(old.result(),new.result());self.assertEqual(old.consume_bonus(),new.consume_bonus())

    def test_new_height_judge_and_historical_records(self):
        e=episode();self.assertTrue(jump_episode_ok(e));self.assertFalse(jump_episode_ok(e,.2))
        for height in (.2,.3):
            e=episode();e.update(height_target_m=height,apex_rise_m=height)
            self.assertTrue(jump_episode_ok(e,height));e['apex_rise_m']=height-.00001
            self.assertFalse(jump_episode_ok(e,height))
        rows=suite()
        for row in rows[-2:]:
            row['height_target_m']=.2
            for e in row['episodes']:e.update(height_target_m=.2,apex_rise_m=.2)
        self.assertTrue(jump_suite_status(rows,.2)[0]);self.assertFalse(jump_suite_status(rows,.3)[0])
        rows[-1]['height_target_m']=.3;self.assertFalse(jump_suite_status(rows)[0])
        rows[-1]['height_target_m']=.2;rows[-1]['episodes'][0]['height_target_m']=.05
        self.assertFalse(judge(rows[-1])[0])  # mismatch cannot consume one of two allowed DR failures

    def test_exact_two_task_changes_and_four_blocks(self):
        old=json.loads((INIT/'args.json').read_text());new=plan()
        self.assertEqual({k for k in new if new[k]!=old.get(k)}, {'name','steps','init','jump','checkpoint_check'})
        self.assertEqual(json.loads(new['jump']),{**json.loads(old['jump']),'height_target_m':.2,'obs_phase':True})
        for key in ('weights','springs_override','action_lpf','scale_override','cmd','cmd_final'):
            self.assertEqual(new[key],old[key])
        done=0;chunks=[]
        while n:=budget_chunk(4_000_000-done,8192):done+=n;chunks.append(n)
        self.assertEqual(chunks,[BLOCK]*4);self.assertEqual(done,MAX_STEPS)
        self.assertEqual(done,3_997_696)

    def test_stop_rules_comparison_is_two_blocks(self):
        zero=dict(passed=0,progress=.2);one=dict(passed=1,progress=.4);two=dict(passed=2,progress=.6)
        self.assertEqual(stop_reason(1,[600]*50,[],zero,zero),'continue')
        self.assertIn('two blocks',stop_reason(2,[600]*50,[one],zero,zero))
        self.assertEqual(stop_reason(2,[600]*50,[one],dict(passed=0,progress=.25),zero),'continue')
        self.assertIn('two blocks',stop_reason(3,[600]*50,[one,two],dict(passed=1,progress=.449),zero))
        self.assertEqual(stop_reason(3,[600]*50,[one,two],dict(passed=1,progress=.45),zero),'continue')
        self.assertIn('4M budget',stop_reason(4,[600]*50,[one,two,two],dict(passed=3,progress=.6),zero))
        self.assertIn('5a-1',stop_reason(4,[600]*50,[one,two,two],zero,zero,True))
        self.assertIn('half',stop_reason(1,[600]*40+[200]*10,[],zero,zero))
        self.assertIn('nonfinite',stop_reason(1,[float('nan')],[],zero,zero))

    def test_checkpoint_rejects_mixed_height_or_seed_sets(self):
        rows=[]
        for cond,seeds in [('in_place',[9300,9301]),('in_place_dr',[9310,9311,9312,9313])]:
            es=[{**episode(seed), 'height_target_m':.2, 'apex_rise_m':.21, 'progress':1.} for seed in seeds]
            rows.append(dict(cond=cond,height_target_m=.2,episodes=es))
        self.assertEqual(metrics(rows)['passed'],6)
        changed=copy.deepcopy(rows);changed[1]['height_target_m']=.05
        with self.assertRaises(ValueError):metrics(changed)
        changed=copy.deepcopy(rows);changed[1]['episodes'][0]['seed']=9999
        with self.assertRaises(ValueError):metrics(changed)

    def test_phase_removal_fails_resume(self):
        old=dict(symmetric=True,jump=json.dumps({'obs_phase':True}))
        with self.assertRaises(ValueError):validate_jump_resume(old,{**old,'jump':'{}'})

    @unittest.skipUnless((INIT/'model.zip').exists(),'local j3 checkpoint required')
    def test_real_checkpoint_expansion_actor_critic_and_norm(self):
        torch.set_num_threads(1);model,vn,_=initialized()
        try:
            r=compare(model,vn);self.assertEqual(r['input_shape'],[56,57]);self.assertEqual(r['training_steps'],0)
        finally:vn.close()

if __name__=='__main__':unittest.main()
