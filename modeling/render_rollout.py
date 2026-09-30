"""Render a recorded policy rollout (sim/rl/record_rollout.py) with the R-02 visual GLB meshes in Blender.

Each link GLB (authored in its URDF link frame) gets the recorded world pose of its MuJoCo body every frame; a
ground grid, a sun light and a camera that follows the torso are added. The motion comes only from the recording.
Usage:
  blender --background --python modeling/render_rollout.py -- ROLLOUT.npz OUT_DIR [--size 960 540] [--view side|three_quarter]
Writes OUT_DIR/frame_####.png; encode with ffmpeg.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
MESHES = ROOT/'src/raptor_description/meshes/r02'
args = sys.argv[sys.argv.index('--')+1:]
rollout, out = Path(args[0]), Path(args[1])
size = (960, 540)
view = 'three_quarter'
if '--size' in args:
    i = args.index('--size'); size = (int(args[i+1]), int(args[i+2]))
if '--view' in args:
    view = args[args.index('--view')+1]
out.mkdir(parents=True, exist_ok=True)
data = np.load(rollout, allow_pickle=False)
names, pos, quat = [str(n) for n in data['names']], data['pos'], data['quat']
meta = json.loads(str(data['meta']))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
objs = {}
for n in names:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(MESHES/f'{n}.glb'))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == 'MESH']
    root = bpy.data.objects.new(n, None)
    scene.collection.objects.link(root)
    for o in new:
        if o.parent is None:
            o.parent = root
    objs[n] = root

# ground: checker-like grid material
bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, 0))
ground = bpy.context.active_object
mat = bpy.data.materials.new('ground'); mat.use_nodes = True
nodes, links = mat.node_tree.nodes, mat.node_tree.links
bsdf = nodes['Principled BSDF']
chk = nodes.new('ShaderNodeTexChecker'); chk.inputs['Scale'].default_value = 400
chk.inputs['Color1'].default_value = (.80, .82, .84, 1); chk.inputs['Color2'].default_value = (.62, .65, .68, 1)
links.new(chk.outputs['Color'], bsdf.inputs['Base Color']); bsdf.inputs['Roughness'].default_value = .9
ground.data.materials.append(mat)
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10)); sun = bpy.context.active_object
sun.data.energy = 3.5; sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(30))
world = bpy.data.worlds.new('w'); scene.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (.82, .86, .9, 1); world.node_tree.nodes['Background'].inputs[1].default_value = .9
bpy.ops.object.camera_add(); cam = bpy.context.active_object; scene.camera = cam
cam.data.lens = 35

scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items] else 'BLENDER_EEVEE'
scene.render.resolution_x, scene.render.resolution_y = size
scene.render.image_settings.file_format = 'PNG'
base = names.index('base_link')
offset = {'side': Vector((0., -2.2, .40)), 'three_quarter': Vector((1.35, -1.75, .55))}[view]
for f in range(len(pos)):
    for i, n in enumerate(names):
        q = quat[f, i]
        M = Matrix.Translation(Vector(pos[f, i])) @ Quaternion((q[0], q[1], q[2], q[3])).to_matrix().to_4x4()
        objs[n].matrix_world = M
    target = Vector(pos[f, base]); target.z = .45
    cam.location = target + offset
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scene.render.filepath = str(out/f'frame_{f:04d}.png')
    bpy.ops.render.render(write_still=True)
print('RENDERED', len(pos), 'frames', json.dumps(meta))
