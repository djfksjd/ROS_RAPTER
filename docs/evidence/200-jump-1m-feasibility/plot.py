"""Two plain charts, solely from the saved calculation JSON."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import E,MOD

req=json.loads((E/'eval/requirements.json').read_text())
detail=json.loads((E/'eval/details.json').read_text())
sweep=json.loads((E/'eval/sweep.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
c=MOD['RS04']['tn_curve'];ax[0].plot(np.array(c['rpm'])*np.pi/30,c['nm'],color='black',label='RS04 motoring T-N cap')
for r,label,color in [(req['deep'],'Frozen drives, ratchet released: 1 m','#b55b4b'),(detail['combo_launch'],'1.0 m / 13 kg / belts, released: 1 m','#356c9b')]:
 a=r['samples'];ax[0].plot(np.abs(a['module_speed'][1]),np.abs(a['module_tau'][1]),label=label,color=color)
ax[0].set(xlabel='Knee module output speed (rad/s)',ylabel='Module torque magnitude (N m)',title='Push-off requirement; knee motor')
ax[0].legend(fontsize=8);ax[0].grid(alpha=.2)
b=np.linspace(2.1,8,400);ax[1].plot(b,1/(b-1),label='Constant force: minimum stroke',color='#356c9b')
ax[1].plot(b,1/(b/2-1),label='Triangular F vs distance (assumed)',color='#b55b4b')
ax[1].axhline(req['deep']['stroke_m'],color='black',linestyle='--',label='Frozen q: 0.8 to 2.6')
ax[1].set(xlabel='Peak total GRF / body weight',ylabel='Stopping distance (m)',ylim=(0,1.2),title='1 m fall; gravity during stopping included')
ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
fig.suptitle('Reduced-model requirements; not a validated jump or landing')
fig.savefig(E/'media/requirements.png',dpi=170);plt.close(fig)
rows={r['label']:r for r in sweep['candidates']}
keys=['frozen','tuned_belts','RS03_both','RS06_both','long_1.25','long_belts_1.25']
labels=['Frozen\n14.92 kg','Belts\n14.92 kg','RS03 + belts\n12.76 kg','RS06 + belts\n11.72 kg','1.0 m legs\n15.72 kg','1.0 m + belts\n15.72 kg']
vals=[rows[k]['best']['target_m'] for k in keys]
ratchet=json.loads((E/'eval/ratchet.json').read_text())
vals.insert(0,ratchet['best_quasistatic_preloaded_family']['target_m'])
labels[0]='Frozen drives\nratchet released'
labels.insert(0,'Frozen + ratchet\nstatic preload')
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
bars=ax.bar(range(len(vals)),vals,color='#567c9b',width=.65)
for rect,v in zip(bars,vals):ax.text(rect.get_x()+rect.get_width()/2,v+.015,f'{v:.3f}',ha='center')
ax.axhline(1,color='#a84e40',linestyle='--',label='Requested 1 m')
ax.set_xticks(range(len(vals)),labels,fontsize=9)
ax.set(ylabel='Highest feasible height in the tested path family (m)',ylim=(0,1.15),title='Direct-drive screen: ratchet released, except first bar; no safety margin')
ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper right')
fig.savefig(E/'media/design_tradeoffs.png',dpi=170);plt.close(fig)
