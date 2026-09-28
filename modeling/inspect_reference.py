"""Inspect an AI-generated reference model and render scaled orthographic views.

Usage:
  Blender --background --python modeling/inspect_reference.py -- MODEL.glb OUT_DIR [EXCLUDE_REGEX]

Writes OUT_DIR/{side,front,top}.png and OUT_DIR/report.json. The model is not modified or
exported. Views are orthographic with a known scale (report.json: metres per pixel in the
model's own units) so joint positions can be read off the images. Reference only: an AI mesh
is not a CAD design and its proportions are targets to be reconciled with the Xacro.
"""
import json
import re
import sys
from pathlib import Path

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index('--')+1:]
model, out = Path(args[0]), Path(args[1])
exclude = re.compile(args[2]) if len(args) > 2 else None  # e.g. presentation ground planes
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
suffix = model.suffix.lower()
if suffix in ('.glb', '.gltf'):
    bpy.ops.import_scene.gltf(filepath=str(model))
elif suffix == '.fbx':
    bpy.ops.import_scene.fbx(filepath=str(model))
elif suffix == '.obj':
    bpy.ops.wm.obj_import(filepath=str(model))
else:
    raise SystemExit('unsupported format '+suffix)

for obj in [o for o in bpy.context.scene.objects if exclude and exclude.search(o.name)]:
    bpy.data.objects.remove(obj)
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
depsgraph = bpy.context.evaluated_depsgraph_get()
corners, triangles = [], 0
for obj in meshes:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    triangles += len(mesh.loop_triangles)
    corners += [obj.matrix_world @ v.co for v in mesh.vertices]
    evaluated.to_mesh_clear()
low = Vector((min(c[i] for c in corners) for i in range(3)))
high = Vector((max(c[i] for c in corners) for i in range(3)))
size, center = high-low, (high+low)/2
images = [{'name': i.name, 'size': list(i.size)} for i in bpy.data.images if i.size[0]]
armatures = [o.name for o in bpy.context.scene.objects if o.type == 'ARMATURE']

scene = bpy.context.scene
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.light = 'STUDIO'
scene.display.shading.color_type = 'TEXTURE'
resolution = 1600
scene.render.resolution_x = scene.render.resolution_y = resolution
world = bpy.data.worlds.new('bg')
world.color = (1, 1, 1)
scene.world = world
span = max(size)*1.1
camera = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
camera.data.type = 'ORTHO'
camera.data.ortho_scale = span
scene.collection.objects.link(camera)
scene.camera = camera
views = {'side': ((0, -1, 0), (1.5708, 0, 0)), 'front': ((1, 0, 0), (1.5708, 0, 1.5708)),
         'top': ((0, 0, 1), (0, 0, 0))}
for name, (direction, rotation) in views.items():
    camera.location = center+Vector(direction)*span*2
    camera.rotation_euler = rotation
    scene.render.filepath = str(out/f'{name}.png')
    bpy.ops.render.render(write_still=True)

(out/'report.json').write_text(json.dumps({
    'source': model.name, 'mesh_objects': len(meshes), 'triangles': triangles,
    'bbox_min': list(low), 'bbox_max': list(high), 'size': list(size), 'center': list(center),
    'units_per_pixel': span/resolution, 'image_center_units': list(center),
    'view_axes': {'side': 'image x = +X, image up = +Z', 'front': 'image x = +Y, image up = +Z',
                  'top': 'image x = +X, image up = +Y'},
    'textures': images, 'armatures': armatures}, indent=1))
print('report', out/'report.json')
