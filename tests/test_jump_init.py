"""Real stand-checkpoint actor/critic compatibility after the one appended input."""
import sys
import unittest
from pathlib import Path
import torch
E=Path(__file__).resolve().parents[1]/'docs/evidence/199-l80-jump'
sys.path.insert(0,str(E))
from verify_j1_init import initialized,compare,INIT
from train_run import validate_jump_resume

class JumpInitTest(unittest.TestCase):
    @unittest.skipUnless((INIT/'model.zip').exists(), 'local source checkpoint required')
    def test_zero_column_actor_critic_and_exploration_preserved(self):
        torch.set_num_threads(1)
        model,vn,_=initialized()
        try:
            result=compare(model,vn)
            self.assertEqual(result['input_shape'],[55,56])
            self.assertTrue(result['action_std_preserved'])
        finally:vn.close()

    def test_refuse_silent_symmetric_or_layout_change(self):
        old=dict(symmetric=True,obs_hold='{}')
        validate_jump_resume(old,{**old,'jump':'{}'})
        validate_jump_resume(old,{'symmetric':False}) # jump disabled: existing paths unchanged
        with self.assertRaises(ValueError):validate_jump_resume(old,dict(jump='{}',symmetric=False))
        with self.assertRaises(ValueError):validate_jump_resume(old,dict(jump='{}',symmetric=True,obs_hold=None))

if __name__=='__main__':unittest.main()
