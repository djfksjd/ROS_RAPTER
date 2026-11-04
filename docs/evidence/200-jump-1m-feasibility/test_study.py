"""Independent identities, runtime parity and finite-budget evidence checks."""
import json
import unittest
from pathlib import Path
import numpy as np
from study import *


class RequirementTests(unittest.TestCase):
    def test_ballistic_energy_and_force(self):
        d=Design();p=path(d);r=evaluate(d,p)
        self.assertAlmostEqual(r['flight_rise_m']+r['extension_rise_vs_standing_m'],1.)
        self.assertAlmostEqual(r['point_mass_work_total_j'],r['kinetic_energy_j']+d.mass*G*p['stroke'],9)
        self.assertAlmostEqual(r['point_mass_work_per_leg_j']*2,r['point_mass_work_total_j'])

    def test_virtual_work_with_nonzero_hip_moment(self):
        p=path(Design())
        # Independent geometric finite difference: d(foot z)/dz = -1 along the constrained path.
        np.testing.assert_allclose(np.sum(p['lever']*p['dq_dz'],axis=0),1.,atol=2e-7)
        self.assertTrue(np.all(np.abs(p['lever'][0])>0))

    def test_fourbar_and_armature_match_runtime(self):
        import sys
        sys.path.insert(0,str(ROOT/'docs/evidence/199-l80-jump'))
        import jump_probe as jp
        env=jp.make();env.reset(seed=0)
        q=np.linspace(.65,2.6,100);ratio,n2=fourbar(q)
        np.testing.assert_array_equal(ratio,np.interp(q,*env.knee_ratio))
        idx=list(env.rz.motor_name).index('RS04')
        knee=next(i for i,j in enumerate(env.rz.idx) if 'left_knee' in env.active[j])
        self.assertAlmostEqual(n2,env.rz.ext2[knee],12)
        env.close()

    def test_dc_units_and_signed_regen_window(self):
        r=limits('RS04',np.array([0.,10.,30.]),np.array([120.,-120.,-120.]))
        self.assertAlmostEqual(r['i_limit'],120/2.1)
        self.assertNotEqual(r['i_limit'],MOD['RS04']['i_peak'])
        self.assertTrue(r['ok'][0]);self.assertTrue(r['ok'][1]);self.assertFalse(r['ok'][2])
        # Overspeed diode force: back EMF 63 V at 30 rad/s, R=.16 -> -93.75 A.
        lo,hi,over=current_window(63.,48.,.16,120/2.1)
        self.assertTrue(over);self.assertEqual(float(lo),-93.75);self.assertEqual(float(hi),-93.75)

    def test_energy_storage_is_paid_before_release(self):
        s=json.loads((E/'eval/spring.json').read_text())
        self.assertAlmostEqual(s['stored_j'],.5*s['stiffness_total_n_m']*.3**2,9)
        self.assertAlmostEqual(s['energy_check_apex_m'],1.,10)
        self.assertGreater(s['charge_time_s'],0.)

    def test_loaded_spring_energy_is_not_free(self):
        d=Design();p=path(d);r=evaluate(d,p,1.,spring_rest=1.)
        self.assertAlmostEqual(r['spring_energy_start_j'],500*d.mass/11.36*1.6**2)
        self.assertEqual(r['spring_energy_end_j'],0.)
        self.assertFalse(r['feasible'])
        saved=json.loads((E/'eval/ratchet.json').read_text())
        self.assertLess(saved['charge']['stored_total_j'],20.)
        self.assertFalse(saved['combo_loaded_landing']['feasible'])

    def test_evidence_not_adoption(self):
        s=json.loads((E/'eval/probe_summary.json').read_text())
        self.assertEqual(sum(x['episodes'] for x in s['variants']),195)
        self.assertTrue(s['hashes_unchanged'])
        for row in s['variants']:
            self.assertLessEqual(row['episodes'],65)
            self.assertLessEqual(row['best']['max_current_using_clamp_ratio'],1.+1e-10)
        self.assertEqual(hashes(),s['input_hashes'])

if __name__=='__main__':unittest.main()
