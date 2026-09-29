"""Build every R-02 artefact from the design calculator through one path (single source of truth):

  modeling/design_r02.py  ->  src/raptor_description/config/r02_design.yaml
  -> src/raptor_description/urdf/raptor_r02.urdf.xacro (xacro in the ROS image)  ->  URDF
  -> sim/build_model.py  ->  sim/raptor_r02.xml (and sim/raptor_r02_achilles.xml with the ankle spring)

Gazebo/RViz use the same Xacro (sim.launch.py leg_design:=r02), MuJoCo the generated MJCF, so all
simulators share masses, inertias, joint limits, springs and contact geometry.
sim/raptor_r02_v1.xml is the earlier directly generated model kept for reproducing evidence 83
(R-02 stages A and B were trained on it).

Usage: .venv-sim/bin/python modeling/build_r02.py   (needs Docker image raptor:jazzy-local for xacro)
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'modeling'))
import design_r02 as D  # noqa: E402

YAML = ROOT/'src/raptor_description/config/r02_design.yaml'
VARIANTS = {'raptor_r02.xml': [], 'raptor_r02_achilles.xml': ['achilles:=true']}


def main():
    D.export(D.Params(), YAML)
    table = json.loads(YAML.read_text())
    with tempfile.TemporaryDirectory(dir=ROOT/'sim') as tmp:
        for name, args in VARIANTS.items():
            urdf = Path(tmp)/name.replace('.xml', '.urdf')
            subprocess.run(['bash', str(ROOT/'sim/generate_urdf.sh'), str(urdf), *args], check=True,
                           env={**os.environ, 'XACRO_FILE': 'raptor_r02.urdf.xacro'})
            subprocess.run([sys.executable, str(ROOT/'sim/build_model.py'), str(urdf), str(ROOT/'sim'/name)], check=True)
    pose = table['nominal_pose']
    print(f"design mass {table['total_mass']} kg; nominal pose (hip, knee, ankle) = "
          f"{pose['hip_pitch']} {pose['knee_pitch']} {pose['ankle_pitch']}  (sim/rl --crouch)")


if __name__ == '__main__':
    main()
