"""5a gate, full suite completeness and predeclared checkpoint stopping rules."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim/rl'))
from gate_judge import judge,jump_episode_ok,jump_suite_status
from tests.test_turn_gate import complete_suite as turn_suite


def episode(seed=9400):
    return dict(seed=seed,t_fall=None,request_time_s=2.,standing_height_m=.75,flight_s=.1,apex_rise_m=.05,
                landing_error_m=.3,touchdown_s=2.5,standing_hold_s=1.,success_time_s=3.5,elapsed_s=4.5,return_observed_s=1.,return_to='l80_w2',
                load_samples=4500,sample_dt_s=.001,raw_trace='local.npz',handoff_from='l80_w2',
                landing_impact_peak_n=10000.,current_peak_a=1000.,current_over_limit_peak=100.,torque_peak_nm=500.,progress=1.)

def suite():
    out=[r for r in turn_suite() if r['cond']!='moving_turn']
    for r in out:
        if r['gate']=='turn':r['gate']='jump'
    for cond,n,seed in [('in_place',8,9400),('in_place_dr',24,9500)]:
        out.append(dict(gate='jump',cond=cond,n=n,falls=0,quick=False,walk='l80_w2',episodes=[episode(seed+i) for i in range(n)]))
    return out


class JumpGateTest(unittest.TestCase):
    def test_boundaries_and_impact_current_record_only(self):
        self.assertTrue(jump_episode_ok(episode()))
        for key,value in [('flight_s',.099),('apex_rise_m',.049999),('landing_error_m',.300001),
                          ('standing_hold_s',.999),('t_fall',3.),('apex_rise_m',float('nan')),('success_time_s',None),('return_observed_s',.999),('return_to','other')]:
            e=episode();e[key]=value;self.assertFalse(jump_episode_ok(e),key)

    def test_full_suite_missing_quick_duplicate_and_raw(self):
        rows=suite();self.assertTrue(jump_suite_status(rows)[0]);self.assertFalse(jump_suite_status([])[0])
        for i in range(len(rows)):self.assertFalse(jump_suite_status(rows[:i]+rows[i+1:])[0])
        self.assertFalse(jump_suite_status(rows+[copy.deepcopy(rows[-1])])[0])
        for key,value in [('seed',9300),('load_samples',1),('raw_trace',None),('handoff_from','other')]:
            r=suite();r[-1]['episodes'][0][key]=value;self.assertFalse(jump_suite_status(r)[0],key)
        r=suite();r[-1]['quick']=True;self.assertFalse(jump_suite_status(r)[0])
        self.assertFalse(jump_suite_status(suite()+[dict(gate='other',cond='record',quick=True)])[0])

    def test_dr_allows_two_failures_but_not_idle_survival(self):
        r=suite()[-1]
        for e in r['episodes'][:2]:e['t_fall']=3.
        r['falls']=2;self.assertTrue(judge(r)[0])
        r['episodes'][2]['success_time_s']=None;self.assertFalse(judge(r)[0])
        r=suite()[-2];r['episodes'][0]['success_time_s']=None;self.assertFalse(judge(r)[0])

    def test_checkpoint_rules_and_plan(self):
        def module(name):
            spec=importlib.util.spec_from_file_location(name,ROOT/f'docs/evidence/199-l80-jump/{name}.py')
            m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
        c=module('check_training_j1');launch=module('launch_j1')
        history=[dict(passed=0,progress=.2),dict(passed=0,progress=.22)]
        self.assertIn('two blocks',c.stop_reason(3,[600]*50,history,dict(passed=0,progress=.249)))
        self.assertEqual(c.stop_reason(3,[600]*50,history,dict(passed=0,progress=.25)),'continue')
        self.assertIn('zero quick',c.stop_reason(4,[600]*50,history,dict(passed=0,progress=1.)))
        self.assertIn('full 5a',c.stop_reason(4,[600]*50,history,dict(passed=6,progress=1.),True))
        self.assertIn('half',c.stop_reason(1,[600]*40+[100]*10,[],dict(passed=0)))
        self.assertIn('8M',c.stop_reason(8,[600]*50,history,dict(passed=1,progress=1.)))
        import json
        old=json.loads((launch.INIT/'args.json').read_text());new=launch.plan()
        allowed={'name','steps','init','episode_s','jump','resume_action_std','checkpoint_check'}
        self.assertFalse({k for k in set(old)|set(new) if old.get(k)!=new.get(k)}-allowed)
        self.assertEqual(new['weights'],old['weights']);self.assertEqual(new['scale_override'],old['scale_override'])
        self.assertLessEqual(c.MAX_STEPS,8_000_000);self.assertEqual(c.MAX_STEPS,7_995_392)
        launch.verify_inputs()

if __name__=='__main__':unittest.main()
