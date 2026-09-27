"""Blender: reference-inspired cosmetic meshes using the current URDF dimensions.

These are visualization parts, NOT manufacturing-ready CAD. No collision/inertia
or actuator is added. Link-local DAE meshes preserve the existing kinematic chain.
Run: Blender --background --python modeling/build_visuals.py
"""
import bpy
import math
from mathutils import Vector, Matrix, Euler
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src/raptor_description/meshes/raptor'
OUT.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
COLORS = {'graphite': (.065,.075,.085,1), 'titanium': (.36,.40,.43,1),
          'ivory': (.72,.72,.65,1), 'copper': (.66,.29,.08,1),
          'rubber': (.025,.028,.032,1), 'lens': (.05,.26,.33,1)}
MATS = {}
for key, color in COLORS.items():
    material = bpy.data.materials.new(key)
    material.diffuse_color = color
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Metallic'].default_value = .65 if key not in ['rubber','lens'] else .1
    bsdf.inputs['Roughness'].default_value = .3
    MATS[key] = material
GROUPS = {}
active = None


def finish(obj, name, material):
    obj.name = name
    obj.data.materials.append(MATS[material])
    GROUPS.setdefault(active, []).append(obj)
    return obj


def box(name, center, size, material, bevel=.008):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = obj.modifiers.new('Machined edge', 'BEVEL')
        mod.width = bevel; mod.segments = 2
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return finish(obj, name, material)


def rod(name, a, b, radius, material, vertices=24):
    a,b = Vector(a),Vector(b)
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius,
        depth=(b-a).length, location=(a+b)/2)
    obj=bpy.context.object
    obj.rotation_mode='QUATERNION'
    obj.rotation_quaternion=(b-a).to_track_quat('Z','Y')
    return finish(obj,name,material)


def servo(center, radius=.065, width=.11, axis='y'):
    c=Vector(center); direction=Vector((0,1,0) if axis=='y' else (1,0,0))
    rod('Servo housing',c-direction*width/2,c+direction*width/2,radius,'graphite')
    for sign in [-1,1]:
        end=c+direction*sign*(width/2+.004)
        rod('Reduction gear cover',end-direction*.006,end+direction*.006,radius*.85,'ivory')
        rod('Output hub',end-direction*.009,end+direction*.009,radius*.44,'titanium')
        for angle in range(0,360,60):
            angle=math.radians(angle)
            radial=Vector((math.cos(angle),0,math.sin(angle))) if axis=='y' else Vector((0,math.cos(angle),math.sin(angle)))
            p=end+radial*radius*.67
            rod('Hex fastener',p-direction*.01,p+direction*.01,.004,'graphite',6)


active='base_link'
box('Load-bearing central chassis',(0,0,0),(.56,.25,.20),'graphite',.04)
for side in [-1,1]:
    box('Removable side armor',(.025,side*.138,.005),(.43,.025,.16),'ivory',.025)
    for x in [-.15,.19]:
        for z in [-.045,.055]:
            rod('Armor screw',(x,side*.145,z),(x,side*.16,z),.007,'titanium',6)
    for i in range(7):
        box('Cooling fin',(-.22+i*.017,side*.12,.115),(.008,.04,.035),'titanium',.002)
    rod('Power conduit',(-.22,side*.16,-.03),(.20,side*.16,-.07),.008,'copper')
box('Forward sensor housing',(.31,0,.03),(.17,.22,.13),'graphite',.035)
box('Sensor brow',(.32,0,.115),(.23,.24,.025),'ivory',.012)
for side in [-1,1]:
    rod('RGB-D lens',(.39,side*.055,.045),(.41,side*.055,.045),.029,'rubber',32)
    rod('Optical glass',(.411,side*.055,.045),(.414,side*.055,.045),.022,'lens',32)
box('Electronics pack',(-.12,0,.145),(.19,.17,.08),'titanium',.018)
rod('Sensor mast',(0,0,.10),(0,0,.19),.032,'graphite')
rod('Optional lidar visual',(0,0,.19),(0,0,.245),.045,'graphite')
box('Mast cap',(0,0,.248),(.10,.09,.014),'ivory',.006)

for side in ['left','right']:
    active=side+'_hip_roll_link'
    servo((0,0,-.045),.065,.12,'x')
    box('Hip yoke',(0,0,-.085),(.10,.10,.09),'titanium')
    active=side+'_thigh_link'
    servo((0,0,0),.067,.12)
    for y in [-.042,.042]:
        box('Thigh structural rail',(0,y,-.16),(.072,.022,.29),'titanium',.012)
        rod('Thigh tension rod',(.034,y,-.055),(.034,y,-.27),.008,'graphite')
    box('Thigh armor',(.042,0,-.16),(.025,.075,.19),'ivory',.01)
    rod('Hip motor cable',(-.036,.055,-.06),(-.035,.055,-.25),.006,'copper')
    active=side+'_shin_link'
    servo((0,0,0),.047,.105)
    for y in [-.032,.032]:
        rod('Lower leg strut',(0,y,-.035),(0,y,-.265),.016,'titanium')
    box('Shin brace',(0,0,-.16),(.042,.065,.10),'graphite')
    box('Shin armor',(.026,0,-.13),(.018,.064,.13),'ivory')
    active=side+'_foot_link'
    servo((0,0,0),.035,.10)
    box('Heel pad',(-.023,0,-.036),(.075,.105,.036),'rubber')
    for y in [-.04,0,.04]:
        rod('Toe linkage',(.005,y,-.025),(.10,y,-.038),.012,'titanium')
        box('Toe contact',(.12,y,-.047),(.10,.027,.023),'rubber',.008)
        rod('Toe hinge',(.075,y-.013,-.028),(.075,y+.013,-.028),.017,'graphite')

active='tail_yaw_link'
servo((-.035,0,0),.048,.10,'x')
box('Tail base bracket',(-.09,0,0),(.14,.065,.07),'titanium')
active='tail_link'
servo((0,0,0),.039,.095)
# Eight fixed visual vertebrae; no passive dynamics are implied by this mesh.
for i in range(8):
    x=-.033-i*.059
    radius=.045*(1-i*.07)
    box('Vertebra %02d'%i,(x,0,0),(.045,radius*1.7,radius*1.7),'ivory',.01)
    rod('Central compliant-spine visual',(x-.035,0,0),(x+.03,0,0),radius*.42,'graphite')
    for sign in [-1,1]:
        rod('Tail lateral protective rail',(x-.024,sign*radius*.72,0),
            (x+.024,sign*radius*.72,0),.005,'copper')


def export_dae(link, objects):
    # Portable COLLADA with baked link-local vertices and per-part material.
    doc=ET.Element('COLLADA',xmlns='http://www.collada.org/2005/11/COLLADASchema',version='1.4.1')
    asset=ET.SubElement(doc,'asset');ET.SubElement(asset,'unit',name='meter',meter='1')
    ET.SubElement(asset,'up_axis').text='Z_UP'
    effects=ET.SubElement(doc,'library_effects');materials=ET.SubElement(doc,'library_materials')
    for name,color in COLORS.items():
        eff=ET.SubElement(effects,'effect',id=name+'-fx');common=ET.SubElement(eff,'profile_COMMON')
        tech=ET.SubElement(common,'technique',sid='common');phong=ET.SubElement(tech,'phong')
        emission=ET.SubElement(phong,'emission');ET.SubElement(emission,'color').text='0 0 0 1'
        ambient=ET.SubElement(phong,'ambient');ET.SubElement(ambient,'color').text=' '.join(map(str,color))
        diffuse=ET.SubElement(phong,'diffuse');ET.SubElement(diffuse,'color').text=' '.join(map(str,color))
        specular=ET.SubElement(phong,'specular');ET.SubElement(specular,'color').text='0.12 0.12 0.12 1'
        shine=ET.SubElement(phong,'shininess');ET.SubElement(shine,'float').text='25'
        mat=ET.SubElement(materials,'material',id=name);ET.SubElement(mat,'instance_effect',url='#'+name+'-fx')
    geometries=ET.SubElement(doc,'library_geometries')
    scenes=ET.SubElement(doc,'library_visual_scenes');scene=ET.SubElement(scenes,'visual_scene',id='Scene')
    for index,obj in enumerate(objects):
        mesh=obj.data;mesh.calc_loop_triangles();gid='mesh'+str(index)
        geo=ET.SubElement(geometries,'geometry',id=gid);m=ET.SubElement(geo,'mesh')
        source=ET.SubElement(m,'source',id=gid+'-positions')
        points=[obj.matrix_world @ v.co for v in mesh.vertices]
        ET.SubElement(source,'float_array',id=gid+'-array',count=str(len(points)*3)).text=' '.join(str(a) for p in points for a in p)
        tc=ET.SubElement(source,'technique_common');acc=ET.SubElement(tc,'accessor',source='#'+gid+'-array',count=str(len(points)),stride='3')
        for axis in 'XYZ':ET.SubElement(acc,'param',name=axis,type='float')
        normals=[(obj.matrix_world.to_3x3().inverted().transposed() @ t.normal).normalized()
                 for t in mesh.loop_triangles]
        normal_source=ET.SubElement(m,'source',id=gid+'-normals')
        ET.SubElement(normal_source,'float_array',id=gid+'-normals-array',count=str(len(normals)*3)).text=' '.join(str(a) for n in normals for a in n)
        ntc=ET.SubElement(normal_source,'technique_common')
        nacc=ET.SubElement(ntc,'accessor',source='#'+gid+'-normals-array',count=str(len(normals)),stride='3')
        for axis in 'XYZ':ET.SubElement(nacc,'param',name=axis,type='float')
        vertices=ET.SubElement(m,'vertices',id=gid+'-vertices');ET.SubElement(vertices,'input',semantic='POSITION',source='#'+gid+'-positions')
        mat=obj.data.materials[0].name
        triangles=ET.SubElement(m,'triangles',count=str(len(mesh.loop_triangles)),material=mat)
        ET.SubElement(triangles,'input',semantic='VERTEX',source='#'+gid+'-vertices',offset='0')
        ET.SubElement(triangles,'input',semantic='NORMAL',source='#'+gid+'-normals',offset='1')
        ET.SubElement(triangles,'p').text=' '.join(str(v) for i,t in enumerate(mesh.loop_triangles) for vertex in t.vertices for v in [vertex,i])
        node=ET.SubElement(scene,'node',id=gid+'-node');inst=ET.SubElement(node,'instance_geometry',url='#'+gid)
        bind=ET.SubElement(inst,'bind_material');tc=ET.SubElement(bind,'technique_common')
        ET.SubElement(tc,'instance_material',symbol=mat,target='#'+mat)
    ET.SubElement(ET.SubElement(doc,'scene'),'instance_visual_scene',url='#Scene')
    ET.ElementTree(doc).write(OUT/(link+'.dae'),encoding='utf-8',xml_declaration=True)


bpy.context.view_layer.update()
for link,objects in GROUPS.items():
    export_dae(link,objects)
    for material in COLORS:
        subset=[obj for obj in objects if obj.data.materials[0].name==material]
        if subset:export_dae(link+'__'+material,subset)

# Assemble a bent-leg presentation pose; this pose is not a validated gait.
urdf=ET.parse(ROOT/'src/raptor_description/urdf/raptor.urdf.xacro').getroot()
transforms={'base_root':Matrix.Identity(4)}
for joint in urdf.findall('joint'):
    parent=joint.find('parent').get('link');child=joint.find('child').get('link')
    origin=joint.find('origin');xyz=[float(v) for v in origin.get('xyz','0 0 0').split()]
    angle=-.43 if 'hip_pitch' in joint.get('name') else .86 if 'knee_pitch' in joint.get('name') else -.43 if 'ankle_pitch' in joint.get('name') else 0
    transforms[child]=transforms[parent] @ Matrix.Translation(xyz) @ Euler((0,angle,0)).to_matrix().to_4x4()
for link,objects in GROUPS.items():
    for obj in objects:obj.matrix_world=transforms[link] @ obj.matrix_world

bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,.04))
plane=bpy.context.object;plane.name='Presentation ground';plane.data.materials.append(MATS['graphite'])
world=bpy.context.scene.world;world.use_nodes=True
world.node_tree.nodes['Background'].inputs[0].default_value=(.17,.19,.23,1)
world.node_tree.nodes['Background'].inputs[1].default_value=.5
for loc,power,size in [((2,-3,4),1300,4),((-3,1,3),1800,3),((1,3,2),900,2)]:
    bpy.ops.object.light_add(type='AREA',location=loc)
    light=bpy.context.object;light.data.energy=power;light.data.shape='DISK';light.data.size=size
    light.rotation_euler=(Vector((0,0,.55))-light.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=(1.9,-2.7,1.5))
cam=bpy.context.object;cam.rotation_euler=(Vector((-.15,0,.58))-cam.location).to_track_quat('-Z','Y').to_euler()
cam.data.type='ORTHO';cam.data.ortho_scale=1.8
scene=bpy.context.scene;scene.camera=cam;scene.render.engine='CYCLES';scene.cycles.samples=40
scene.render.resolution_x=1600;scene.render.resolution_y=1100;scene.render.resolution_percentage=100
scene.render.filepath=str(ROOT/'docs/evidence/raptor-concept-render.png')
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'modeling/raptor-assembly.blend'))
bpy.ops.render.render(write_still=True)
