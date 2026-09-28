"""Split a Morphloom Raptor GLB into URDF link-local visual meshes and render a posed preview.

Usage:
  Blender --background --python modeling/morphloom/split_links.py -- ASSET.glb URDF OUT_DIR [PREVIEW_PREFIX]

Component ids are `<link>__<part>` authored at the URDF zero pose. Each link's parts are joined,
moved into that link frame with the inverse zero-pose transform and exported as OUT_DIR/<link>.glb.
The preview places links at the nominal digitigrade crouch (hip -0.10, knee 0.50, ankle -0.40)
on flat ground. Rendered preview is a visual check, not a physics result.
"""
import sys
from collections import defaultdict
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/raptor_control/scripts'))
from lateral_feasibility import Model  # noqa: E402

args = sys.argv[sys.argv.index('--')+1:]
asset, urdf, out = Path(args[0]), Path(args[1]), Path(args[2])
preview = args[3] if len(args) > 3 else None
out.mkdir(parents=True, exist_ok=True)
model = Model(urdf.read_text())
zero = model.fk({})
pose = {f'{s}_{j}_joint': v for s in ('left', 'right')
        for j, v in (('hip_pitch', -.10), ('knee_pitch', .50), ('ankle_pitch', -.40))}
posed = model.fk(pose)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(asset))
groups = defaultdict(list)
for obj in list(bpy.context.scene.objects):
    if obj.type == 'MESH' and '__' in obj.name:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
        groups[obj.name.split('__')[0]].append(obj)
for obj in [o for o in bpy.context.scene.objects if o.type == 'EMPTY']:
    bpy.data.objects.remove(obj)

links = {}
for link, objs in sorted(groups.items()):
    if link not in zero:
        raise SystemExit(f'component prefix {link} is not a URDF link')
    for obj in objs:  # bake world transforms into mesh data
        obj.data = obj.data.copy()
        obj.data.transform(obj.matrix_world)
        obj.matrix_world = Matrix.Identity(4)
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = joined.data.name = link
    joined.data.transform(Matrix(np.linalg.inv(zero[link]).tolist()))
    links[link] = joined
    bpy.ops.object.select_all(action='DESELECT')
    joined.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(out/f'{link}.glb'), export_format='GLB', use_selection=True,
                              export_yup=True, export_apply=True)
print('exported', len(links), 'links to', out)

if preview:
    for link, obj in links.items():
        obj.matrix_world = Matrix(posed[link].tolist())
    bpy.context.view_layer.update()
    low = min((obj.matrix_world @ Vector(c)).z for obj in links.values() for c in obj.bound_box)
    for obj in links.values():
        obj.matrix_world = Matrix.Translation((0, 0, -low)) @ obj.matrix_world
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y = 1400, 1000
    world = bpy.data.worlds.new('studio')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = (.82, .83, .85, 1)
    world.node_tree.nodes['Background'].inputs[1].default_value = .9
    scene.world = world
    bpy.ops.mesh.primitive_plane_add(size=6)
    floor = bpy.context.active_object
    floor.data.materials.append(bpy.data.materials.new('floor'))
    floor.data.materials[0].diffuse_color = (.55, .56, .58, 1)
    for name, rotation, energy in (('key', (0.9, 0.2, 0.6), 4.), ('rim', (1.1, 0, 3.4), 2.5)):
        sun = bpy.data.objects.new(name, bpy.data.lights.new(name, 'SUN'))
        sun.data.energy = energy
        sun.rotation_euler = rotation
        scene.collection.objects.link(sun)
    camera = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
    camera.data.lens = 55
    scene.collection.objects.link(camera)
    scene.camera = camera
    for view, location in (('side', (0.1, -3.4, 0.55)), ('three_quarter', (2.3, -2.4, 1.0)),
                           ('front', (3.2, 0.0, 0.6))):
        camera.location = location
        direction = Vector((-.12, 0, .42))-camera.location
        camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = f'{preview}_{view}.png'
        bpy.ops.render.render(write_still=True)
    print('preview', preview)
