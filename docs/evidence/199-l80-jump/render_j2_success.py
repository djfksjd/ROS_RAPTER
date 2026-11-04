"""Actual successful gate replay, optional OpenGL or evidence-197 CPU scene projection."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/raptor-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'sim/rl'));sys.path.insert(0,str(E.parent/'197-l80-run'))
from jump_eval import jump_rollout
from gate_judge import jump_episode_ok
from render_cpu import render


def plot(shots,meta,backend,original):
    labels=['Before request','Push-off','Both feet off','Apex','Descent','Landing','Hold complete','Walk returned']
    fig,axes=plt.subplots(2,4,figsize=(13,7.2))
    for ax,shot,label,m in zip(axes.flat,shots,labels,meta):
        ax.imshow(shot);ax.axis('off')
        ax.set_title(f"{label} | {m['t']:.2f} s\nFeet contacts L/R: {m['feet_contacts'][0]} / {m['feet_contacts'][1]}",fontsize=9)
    fig.suptitle('l80_j2 | DR seed 9515 | actual successful gate replay\n'
                 f"Flight {original['flight_s']*1000:.0f} ms, rise {original['apex_rise_m']*100:.2f} cm, landing {original['landing_error_m']*100:.2f} cm, hold 1 s\n"+backend,fontsize=11)
    fig.subplots_adjust(left=.015,right=.985,bottom=.02,top=.79,wspace=.07,hspace=.42)
    fig.savefig(E/'media/j2_success_sequence.png',dpi=160);plt.close(fig)


def main():
    if '--plot-only' in sys.argv:
        record=json.loads((E/'eval/j2_full/frame_sequence.json').read_text())
        plot([render(s) for s in record['scenes']],record['frames'],record['backend'],record['result'])
        return
    torch.set_num_threads(1)
    data=json.loads((E/'eval/j2_full/analysis.json').read_text())
    chosen=next(e for e in data['episodes'] if e['seed']==9515)
    original=chosen['gate_episode'];event=chosen['representative_flight']
    with np.load(original['raw_trace']) as raw:
        mask=(raw['t']>=event['takeoff_s'])&(raw['t']<event['touchdown_s'])
        ids=np.flatnonzero(mask);apex=float(raw['t'][ids[np.argmax(raw['xyz'][ids,2])]])
    times=[1.8,event['takeoff_s']-.10,event['takeoff_s']+.02,apex,
           event['touchdown_s']-.02,event['touchdown_s']+.04,original['success_time_s'],original['elapsed_s']]
    steps=[round(t/.02) for t in times]
    shots=[];scenes=[];meta=[];renderer=None;scene=None;camera=None;error=None

    def observer(env):
        nonlocal renderer,scene,camera,error
        step=round(env.data.time/.02)
        if step not in steps:return
        if camera is None:
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:]=[0.,0.,.45];camera.distance=2.4;camera.elevation=-12.;camera.azimuth=90.
            env.model.vis.headlight.ambient[:]=.45
            try:renderer=mujoco.Renderer(env.model,360,420)
            except Exception as ex:
                error=f'{type(ex).__name__}: {ex}';print('OpenGL unavailable:',error,flush=True)
            scene=mujoco.MjvScene(env.model,maxgeom=1000)
        if renderer is not None:
            renderer.update_scene(env.data,camera);shots.append(renderer.render().copy())
        else:
            mujoco.mjv_updateScene(env.model,env.data,mujoco.MjvOption(),None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
            frame=dict(geoms=[dict(type=int(g.type),size=g.size.tolist(),pos=g.pos.tolist(),mat=g.mat.reshape(3,3).tolist(),rgba=g.rgba.tolist())
                       for g in scene.geoms[:scene.ngeom] if g.rgba[3]>0],
                       camera_pos=np.mean([c.pos for c in scene.camera],axis=0).tolist(),
                       camera_forward=scene.camera[0].forward.tolist(),camera_up=scene.camera[0].up.tolist(),
                       fovy=float(env.model.vis.global_.fovy))
            scenes.append(frame);shots.append(render(frame))
        sample=env.jump_task.last_sample
        meta.append(dict(t=float(env.data.time),base_xyz=sample['xyz'].tolist(),feet_contacts=sample['feet_n'].tolist(),
                         feet_force_n=sample['feet_force_n'].tolist()))
    try:
        result=jump_rollout(ROOT/'sim/rl/runs/l80_j2',ROOT/'sim/rl/runs/l80_w2',9515,True,
            E/'eval/j2_full/frame_replay',observer=observer)
    finally:
        if renderer is not None:renderer.close()
    assert jump_episode_ok(result)
    for key,value in original.items():
        if key!='raw_trace':assert result[key]==value,(key,result[key],value)
    with np.load(original['raw_trace']) as a,np.load(result['raw_trace']) as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    assert len(shots)==8
    backend='CPU projection of MuJoCo scene (evidence 197)' if scenes else 'MuJoCo OpenGL'
    plot(shots,meta,backend,original)
    (E/'eval/j2_full/frame_sequence.json').write_text(json.dumps(dict(backend=backend,opengl_error=error,
        result=result,frames=meta,exact_raw_replay=True,scenes=scenes),indent=2)+'\n')
    print(backend,'8 frames; exact replay; gate PASS',flush=True)


if __name__=='__main__':main()
