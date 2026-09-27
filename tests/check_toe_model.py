"""Run against generated URDF/SDF files: python check_toe_model.py baseline.urdf toes.urdf toes.sdf."""
import sys
import xml.etree.ElementTree as ET


def check(baseline_path, toes_path, sdf_path):
    base=ET.parse(baseline_path).getroot();toes=ET.parse(toes_path).getroot();sdf=ET.parse(sdf_path).getroot()
    active=[j.get('name') for j in toes.findall('ros2_control/joint')]
    expected=[f'{s}_{j}_joint' for s in ['left','right']
              for j in ['hip_roll','hip_pitch','knee_pitch','ankle_pitch']]+['tail_yaw_joint','tail_pitch_joint']
    assert active==expected and len(active)==10
    passive=[j for j in toes.findall('joint') if '_toe_' in j.get('name')]
    assert len(passive)==12 and not set(active)&{j.get('name') for j in passive}
    assert len([j for j in base.findall('joint') if j.get('type')=='revolute'])==10
    for joint in passive:
        dyn=sdf.find("model/joint[@name='"+joint.get('name')+"']/axis/dynamics")
        assert dyn is not None and float(dyn.findtext('spring_stiffness'))>0
        inertia=toes.find("link[@name='"+joint.find('child').get('link')+"']/inertial/inertia")
        assert all(float(inertia.get(k))>0 for k in ['ixx','iyy','izz'])
    mass=lambda r:sum(float(i.find('mass').get('value')) for i in r.findall('link/inertial'))
    assert abs(mass(base)-mass(toes))<1e-9
    for name in expected:
        assert ET.tostring(base.find("joint[@name='"+name+"']"))==ET.tostring(toes.find("joint[@name='"+name+"']"))
    print('PASS: default 10 DOF; toe variant 10 actuated + 12 passive; springs retained in SDF')
    print('PASS: active joint geometry/limits unchanged; positive toe inertias; total mass',mass(toes))


if __name__=='__main__':check(*sys.argv[1:])
