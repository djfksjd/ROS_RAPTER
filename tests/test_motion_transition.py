"""Catalogue gates must reject speed, stance, missing stages and incomplete holds."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sim'/'rl'))
try:
    from motion_transition import command_at, summarize, stand_gate
except ImportError:
    summarize = None


@unittest.skipIf(summarize is None, 'simulator dependencies unavailable')
class MotionTransitionTest(unittest.TestCase):
    def rows(self):
        rows=[]
        for k in range(3400):
            cmd,stage,cycle=command_at(k*.02,'transition')
            rows.append(dict(t=(k+1)*.02,stage=stage,cycle=cycle,vx=cmd,speed_xy=cmd,
                             position_xy=[0,0],before_position_xy=[0,0],tilt=0.,flight=False,
                             loaded={'left':(k//10)%2==0,'right':(k//10)%2==1}))
        return rows

    def test_complete_stages_pass_and_incomplete_trial_fails(self):
        rows=self.rows()
        self.assertTrue(summarize(rows,'transition',False,68.)['pass_gate'])
        self.assertFalse(summarize(rows[:-100],'transition',False,68.)['pass_gate'])
        self.assertFalse(summarize(rows,'transition',True,68.)['pass_gate'])

    def test_lateral_motion_and_slow_stop_start_fail(self):
        rows=self.rows()
        for row in rows:
            if row['stage']=='stop':row['speed_xy']=.5
        self.assertFalse(summarize(rows,'transition',False,68.)['pass_gate'])
        rows=self.rows()
        for row in rows:
            if row['stage']=='run':row['speed_xy']=1.
        self.assertFalse(summarize(rows,'transition',False,68.)['pass_gate'])

    def test_initial_drift_or_non_alternating_walk_fail(self):
        rows=self.rows();rows[2]['position_xy']=[.10001,0]
        self.assertFalse(summarize(rows,'transition',False,68.)['pass_gate'])
        rows=self.rows()
        for row in rows:row['loaded']={'left':True,'right':True}
        self.assertFalse(summarize(rows,'transition',False,68.)['pass_gate'])

    def test_stand_origin_and_flight(self):
        row=dict(position_xy=[.10001,0],before_position_xy=[0,0],tilt=0.,flight=False)
        self.assertFalse(stand_gate([row])['pass_gate'])
        row['position_xy']=[0,0];row['flight']=True
        self.assertFalse(stand_gate([row])['pass_gate'])


if __name__=='__main__':unittest.main()
