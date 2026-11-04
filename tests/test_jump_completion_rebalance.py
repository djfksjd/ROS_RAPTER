"""j3 is one positive-weight change with a fixed two-block budget and 0M baseline."""
import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'docs/evidence/199-l80-jump'
sys.path.insert(0,str(E));sys.path.insert(0,str(ROOT/'sim/rl'))
from launch_j3 import plan
from check_training_j3 import baseline,stop_reason,MAX_STEPS,BLOCK
from train_run import budget_chunk
from jump_task import JumpTask
from tests.test_jump_task import frame


class JumpCompletionRebalanceTest(unittest.TestCase):
    def test_only_positive_completion_bonus_changes(self):
        old=json.loads((ROOT/'sim/rl/runs/l80_j2/args.json').read_text());new=plan()
        changed={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
        self.assertEqual(changed,{'name','steps','init','jump','checkpoint_check'})
        before=json.loads(old['jump']);after=json.loads(new['jump'])
        self.assertEqual(before['success_bonus'],30.)
        self.assertEqual(after,{**before,'success_bonus':50.})
        self.assertEqual(new['weights'],old['weights']);self.assertIsNone(new['resume_action_std'])

    def test_two_blocks_no_extension(self):
        done=0;chunks=[]
        while n:=budget_chunk(2_000_000-done,8192):done+=n;chunks.append(n)
        self.assertEqual(chunks,[BLOCK,BLOCK]);self.assertEqual(done,MAX_STEPS)
        self.assertEqual(done,1_998_848)

    def test_fixed_baseline_and_stop_rules(self):
        b=baseline();self.assertEqual(b['passed'],3);self.assertEqual(b['total'],6)
        self.assertAlmostEqual(b['progress'],.9629303400700909)
        self.assertEqual(stop_reason(1,[600.]*50,b,b),'continue')
        self.assertIn('fixed 0M baseline',stop_reason(2,[600.]*50,{**b,'progress':1.},b))
        self.assertIn('2M budget exhausted',stop_reason(2,[600.]*50,{**b,'passed':4},b))
        self.assertIn('full 5a',stop_reason(2,[600.]*50,b,b,True))
        self.assertIn('below half',stop_reason(1,[600.]*40+[200.]*10,b,b,True))
        self.assertIn('nonfinite',stop_reason(1,[float('nan')],b,b))

    def test_completion_adds_twenty_once_without_changing_success(self):
        tasks=[JumpTask({'success_bonus':bonus,'suspend_stand_penalties':True}) for bonus in (30.,50.)]
        for t in range(1,3101):
            sample=frame(t*.001,off=2000<t<=2100,z=1.06 if 2000<t<=2100 else 1.)
            for task in tasks:task.sample(sample,1.,100.)
        self.assertEqual(tasks[0].result(),tasks[1].result())
        bonuses=[task.consume_bonus() for task in tasks]
        for actual,expected in zip(bonuses,(40.,60.)):self.assertAlmostEqual(actual,expected,delta=1e-12)
        self.assertEqual(bonuses[1]-bonuses[0],20.)
        self.assertEqual([task.consume_bonus() for task in tasks],[0.,0.])


if __name__=='__main__':unittest.main()
