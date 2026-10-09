from pathlib import Path
import sys,json,gzip,hashlib,subprocess,ast
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path.cwd();OUT=ROOT/'docs/assets/latest';sys.path.insert(0,str(ROOT/'sim/rl'))
FONT='/Library/Fonts/Arial Unicode.ttf'
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
font_manager.fontManager.addfont(FONT);plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'axes.unicode_minus':False,'font.size':10,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
TEAL='#147F86';ORANGE='#BC6333';GRAY='#81939F'
summary=json.loads((ROOT/'docs/evidence/143-turn-yaw-input-filter/summary.json').read_text())
fig,ax=plt.subplots(figsize=(9,4.5),dpi=150);xx=np.arange(3)
for r,col,shift,label in [(summary[0],TEAL,-.17,'무필터 보정'),(summary[1],GRAY,.17,'0.1초 필터')]:
 vals=[r['angle_pass'],r['speed_pass'],r['joint_pass']];bars=ax.bar(xx+shift,vals,.32,color=col,label=label)
 for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.25,f'{v}/20',ha='center')
ax.set(xticks=xx,xticklabels=['회전량','속도 범위','동시 통과'],ylim=(0,23),ylabel='통과 횟수 / 조건당 20회',title='4 m/s 회전: 각도 통과와 속도 유지의 차이');ax.legend();ax.grid(axis='y',alpha=.2);fig.text(.02,.02,'증거143 · 공통 개발 조건 · 전도 각0/20 · 시뮬레이션, 독립 최종검증/실물 아님',fontsize=9);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'turn-gates.png');plt.close(fig)
trials=[json.loads(x) for x in gzip.decompress((ROOT/'docs/evidence/163-knee-vertical-support/support-capture.jsonl.gz').read_bytes()).decode().splitlines()]
fig,axes=plt.subplots(1,2,figsize=(10,4.5),dpi=150)
for seed,col in [(83002,TEAL),(86013,ORANGE)]:
 r=next(x for x in trials if x['candidate']=='support_only' and x['seed']==seed);rows=r['rows'];t=np.array([x['t'] for x in rows]);pos=np.array([x['position_xy'] for x in rows]);origin=np.array(rows[0]['before_position_xy']);drift=np.linalg.norm(pos-origin,axis=1)
 axes[0].plot(t,drift,color=col,label=f'seed {seed}'+(' (60초 통과)' if seed==83002 else ' (0.9초 실패)'));axes[1].plot(t,np.degrees([x['tilt'] for x in rows]),color=col)
for ax,thr,title,ylabel in [(axes[0],.1,'기립 위치 유지','초기 위치로부터 거리 (m)'),(axes[1],10,'몸통 기울기','기울기 (도)')]:
 ax.axhline(thr,color=ORANGE,linestyle='--',label='통과 한계');ax.set(xlabel='시간 (s)',ylabel=ylabel,title=title,xlim=(0,60));ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.text(.02,.02,'증거163 · support_only · 원개발21조건 중1통과 · 성공 한 조건이 일반 기립 성공률을 뜻하지 않음',fontsize=9);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'stand-success-failure.png');plt.close(fig)
pulse=json.loads((ROOT/'docs/evidence/163-knee-vertical-support/knee-response.json').read_text());fig,ax=plt.subplots(figsize=(9,4.5),dpi=150)
for seed,col in [(83002,TEAL),(86013,ORANGE)]:
 ts=[];vals=[]
 for t in [.1,.2,.24,.3]:
  rr=[x for x in pulse if x['mode']=='paired_pitch_only' and x['seed']==seed and x['time']==t];base=next(x for x in rr if x['knee_pulse_normalized']==0.);plus=next(x for x in rr if x['knee_pulse_normalized']==.01);ts.append(t);vals.append(plus['body_point_vz_world_estimate_mps']-base['body_point_vz_world_estimate_mps'])
 ax.plot(ts,vals,'o-',color=col,label=f'seed {seed}')
ax.axhline(0,color=GRAY,lw=1);ax.set(xlabel='pulse 시작 시각 (s)',ylabel='다음20ms 수직속도 변화 (m/s)',title='같은 무릎 입력 +0.01의 효과: 접촉 상태에 따라 부호 변화');ax.legend();ax.grid(alpha=.2);fig.text(.02,.02,'증거163 · 기존 paired 궤적 상태 재생 · 무변경 대비 · 오프라인 국소 반응, 보편 제어 gain 아님',fontsize=9);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(OUT/'knee-response.png');plt.close(fig)
from local_stand_model import LocalStandModel
from run_env import RunEnv
from train_run import env_kwargs
from evaluate import load
from evaluate_run import episode
candidate=next(x for x in map(json.loads,(ROOT/'docs/evidence/148-static-equilibrium/margin-search.jsonl').read_text().splitlines()) if x['start_knee']==1.1)
run=ROOT/'sim/rl/runs/kl005_lr3e5_update64k_20261009'
class Writer:
 def __init__(self,name,fps=25):
  self.path=OUT/name;self.n=0;self.proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','640x560','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)],stdin=subprocess.PIPE)
 def add(self,frame,title,detail,snapshot=None):
  im=Image.new('RGB',(640,560),'#132C42');im.paste(Image.fromarray(frame),(0,80));d=ImageDraw.Draw(im);d.text((14,10),title,font=ImageFont.truetype(FONT,18),fill='white');d.text((14,39),detail,font=ImageFont.truetype(FONT,14),fill='#C9DFE6');d.text((14,61),'MuJoCo 동역학 모델 · 실물/제작 도면 아님',font=ImageFont.truetype(FONT,12),fill='#C9DFE6');self.proc.stdin.write(im.tobytes());self.n+=1
  if snapshot:im.save(OUT/snapshot)
  return im
 def close(self):self.proc.stdin.close();assert self.proc.wait()==0

def visual(e):
 e.model.vis.headlight.ambient[:]=[.6,.6,.6];e.model.vis.headlight.diffuse[:]=[.8,.8,.8]
 for i in range(e.model.ngeom):
  e.model.geom_rgba[i]=[.23,.3,.35,1.] if e.model.geom_bodyid[i]==0 else [.72,.88,.91,1.]
verification={'source_checkpoint':'a6de8c7a59b758b3ae2c85331ac7bea9fb9c9291','goal_status':'paused','mode':'existing trajectory/evaluation visualization; no new learning or acceptance trials','stand_replays':[]}
for seed in [83002,86013]:
 r=next(x for x in trials if x['candidate']=='support_only' and x['seed']==seed);p=LocalStandModel(run,candidate);e=p.env;e.init_seed=seed;e.reset();e.command=np.zeros(3);e.resample_steps=0;visual(e);writer=Writer(f'stand-{seed}.mp4');gif=[]
 try:
  for i,row in enumerate(r['rows']):
   if i==80:p.set_stand_current_limit(.995)
   _,_,fell,_,info=e.step(np.array(row['action']))
   assert e.data.xpos[e.base][:2].tolist()==row['position_xy'] and float(info['tilt'])==row['tilt'] and bool(info['flight'])==row['flight'],(seed,i)
   sample=(i%8==0 or i==len(r['rows'])-1) if seed==83002 else True
   if sample:
    frame=e.render();title=f'기립 행동 재생 | seed {seed} | '+('개발 단일 성공' if seed==83002 else '실패 사례')
    im=writer.add(frame,title,f't={row["t"]:.2f}s / '+('약4배속 · 60초 시험' if seed==83002 else '0.5배속 · 0.9초 종료'),f'stand-{seed}-snapshot.png' if (seed==83002 and i==48) or (seed==86013 and i==len(r['rows'])-1) else None)
    if seed==83002 and row['t']<=4.:gif.append(im.resize((480,420)))
  if gif:gif[0].save(OUT/'stand-preview.gif',save_all=True,append_images=gif[1:],duration=40,loop=0,optimize=True)
  verification['stand_replays'].append({'seed':seed,'rows_exact':len(r['rows']),'frames':writer.n,'stored_pass':r['original_start_stand_pass'],'speedup':4 if seed==83002 else .5})
 finally:writer.close();p.close()
 print('stand rendered',seed,flush=True)
# Extract the original helper definitions without running the 40-trial experiment script.
tree=ast.parse((ROOT/'docs/evidence/143-turn-yaw-input-filter/probe.py').read_text());classes=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[]);exec(compile(classes,'evidence143_helpers','exec'),globals())
e=MeasuredEnv(**{**env_kwargs(json.loads((run/'args.json').read_text())),'episode_s':15.,'level':0.},randomize=False,seed=7);e.init_seed=67001;model,norm=load(str(run/'model.zip'),e);writer=Writer('run-turn-4mps.mp4');gifs=[]
class Frames:
 def __init__(self):self.n=0
 def append(self,frame):
  self.n+=1
  if self.n==1:visual(e)
  if self.n%2:return
  t=float(e.data.time);im=writer.add(frame,'주행·회전 재생 | 4 m/s | seed 67001',f't={t:.2f}s / yaw +1 rad/s 명령: 4-6초 / 1배속','run-turn-snapshot.png' if self.n==276 else None)
  if 4<=t<=6:gifs.append(im.resize((480,420)))
try:
 result=episode(Adapter(model,1.1,True,0.),norm,e,[4.,0.,0.],10.,turn=(1.,4.,2.),frames=Frames())
 original=next(x for x in map(json.loads,(ROOT/'docs/evidence/143-turn-yaw-input-filter/raw.jsonl').read_text().splitlines()) if x['filter_tau_seconds']==0 and x['seed']==67001 and x['yaw']==1.)
 assert result==original['result'] and max(e.turn_vx)==original['measured_interval_max_vx']
 verification['turn_replay']={'seed':67001,'yaw':1.,'seconds':10,'exact_original_result':True,'angle_pass':original['angle_pass'],'min_interval_vx':result['raw']['turn_interval_min_vx'],'max_interval_vx':max(e.turn_vx),'frames':writer.n}
 if gifs:gifs[0].save(OUT/'run-turn-preview.gif',save_all=True,append_images=gifs[1:],duration=40,loop=0,optimize=True)
finally:writer.close();norm.close()
verification['assets']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.iterdir() if p.suffix in ['.png','.gif','.mp4']}
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n');print('all media verified',verification,flush=True)
