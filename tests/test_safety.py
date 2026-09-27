import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'ai'),str(ROOT/'src/raptor_control/scripts')]
from commands import validate_action, validate_text
from mission_policy import Policy


class SafetyTests(unittest.TestCase):
    def test_stop_latches_and_cannot_be_resumed(self):
        p=Policy();self.assertEqual(p.check('STOP'),'hold')
        for action in ['STAND','RESUME','SEARCH_EAST','PAUSE']:
            self.assertEqual(p.check(action),'rejected_stop_latched')

    def test_pause_requires_explicit_resume(self):
        p=Policy();self.assertEqual(p.check('PAUSE'),'hold')
        self.assertEqual(p.check('STAND'),'rejected_paused')
        self.assertEqual(p.check('RESUME'),'hold')
        self.assertEqual(p.check('STAND'),'stand')

    def test_unverified_navigation_is_rejected(self):
        p=Policy()
        for action in ['SEARCH_EAST','MOVE_WEST','RETURN_BASE','REJECT','evil']:
            self.assertEqual(p.check(action),'rejected_unverified_mission')

    def test_model_cannot_inject_motor_values_or_code(self):
        for data in [{'action':'STAND','positions':[10]}, {'action':'__import__("os")'},
                     {'action':['STOP']}, None, []]:
            with self.assertRaises(ValueError):validate_action(data)

    def test_invalid_text(self):
        for text in ['',None,[], 'x'*501]:
            with self.assertRaises(ValueError):validate_text(text)


if __name__=='__main__':unittest.main()
