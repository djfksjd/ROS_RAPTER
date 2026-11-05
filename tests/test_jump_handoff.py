"""The gate uses walk before request, jump through landing hold, and walk again for 1 s."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim/rl'))
import jump_eval


class JumpHandoffTest(unittest.TestCase):
    def test_both_policy_switches_use_their_own_normalizers(self):
        self.handoff(False)

    def test_optional_landing_handoff_still_requires_hold_and_return(self):
        self.handoff(True)

    def test_phase_policy_strips_two_inputs_for_walk(self):
        self.handoff(False,phase=True)

    def handoff(self,landing_walk,phase=False):
        made=[]
        class Env:
            def __init__(self,**kw):
                self.model=object();self.active=['joint'];self.q0=np.zeros(1);self.scale=np.ones(1);self.actions=[];self.k=0;self.weight=1.
                self.rz=SimpleNamespace(on={'motor':True},idx=np.array([0]),reset=lambda env:None)
                self.data=SimpleNamespace(time=0.,qvel=np.zeros(6))
                self.jump_task=SimpleNamespace(request_time=None,success_time=None,fall_time=None,landed=None,trace=[],capture=False,result=lambda:{})
                made.append(self)
            def reset(self,seed):pass
            def _observe(self):return np.zeros(57 if phase else 56)
            def step(self,action):
                self.actions.append(float(action[0]));self.k+=1;self.data.time=self.k*.02
                if self.k==100:self.jump_task.request_time=2.
                if self.k==120:self.jump_task.landed={'touchdown_s':2.4}
                if self.k==170:self.jump_task.success_time=3.4
                self.jump_task.trace.append(dict(t=self.data.time))
                return self._observe(),0.,False,False,{}
        class Norm:
            def __init__(self,n,offset):self.observation_space=SimpleNamespace(shape=(n,));self.offset=offset
            def normalize_obs(self,obs):
                self_n=self.observation_space.shape[0]
                assert obs.shape==(1,self_n)
                return obs+self.offset
            def close(self):pass
        class Policy:
            def predict(self,obs,deterministic):return obs[:,:1],None
        def load(path,env):return Policy(),Norm(55,1.) if Path(path).parent.name=='walk' else Norm(57 if phase else 56,2.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'jump';walk=root/'walk';run.mkdir();walk.mkdir()
            (run/'args.json').write_text(json.dumps({'jump':json.dumps({'obs_phase':phase})}));(walk/'args.json').write_text('{}')
            with patch.object(jump_eval,'env_kwargs',return_value={}),patch.object(jump_eval,'RunEnv',Env),\
                 patch.object(jump_eval,'load',side_effect=load),patch.object(jump_eval.mujoco,'mj_forward'):
                observed=[]
                result=jump_eval.jump_rollout(run,walk,1,False,root/'raw',landing_walk=landing_walk,
                    observer=lambda env:observed.append(env.data.time))
            self.assertEqual(made[0].actions,[1.]*100+[2.]*20+[1.]*100 if landing_walk else [1.]*100+[2.]*70+[1.]*50)
            self.assertEqual(len(observed),220)
            if landing_walk:self.assertAlmostEqual(result['landing_handoff_s'],2.4)
            else:self.assertNotIn('landing_handoff_s',result)
            self.assertEqual(result['return_to'],'walk');self.assertAlmostEqual(result['return_observed_s'],1.)

if __name__=='__main__':unittest.main()
