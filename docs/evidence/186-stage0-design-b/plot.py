"""Evidence 186 figure. b_utilisation.png: option-B actuator set (modeling/stage0_b_study.py CHOSEN) under the design stride
and under policy-calibrated loads at 6 / 8 / 11.1 m/s: the larger of peak utilisation (torque / available torque at that
speed, published envelope) and RMS / rated torque, with the 1.2 load margin and the -10 % no-load speed tolerance as
hatched bars. Run from repo root."""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E = 'docs/evidence/186-stage0-design-b'
R = json.load(open(f'{E}/chosen.json'))
AX = ['hip_pitch', 'knee', 'hip_roll', 'hip_yaw']
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for a, ld in zip(ax, ('design', 'policy')):
    for i, v in enumerate(('6', '8', '11.1')):
        for j, axn in enumerate(AX):
            x = i*5+j
            s = lambda w0, m: max(R[f'v{v}_w0x{w0}']['cases'][f'{ld} loads, published x{m}'][axn][k] for k in ('util', 'rms_over_rated'))
            a.bar(x, s('1', '1'), .8, color=f'C{j}', label=axn if i == 0 else None)
            a.bar(x, s('1', '1.2')-s('1', '1'), .8, bottom=s('1', '1'), color=f'C{j}', alpha=.45, hatch='//')
            a.plot([x-.4, x+.4], [s('0.9', '1.2')]*2, 'k-', lw=1.5)
    a.axhline(1, color='r', lw=1); a.set_ylim(0, 2.4)
    a.set_xticks([1.5, 6.5, 11.5], ['6 m/s', '8 m/s', '11.1 m/s'])
    a.set_title(f'{ld} loads: max(peak util, RMS/rated); solid = x1.0, hatched = x1.2,\nblack line = x1.2 with module no-load speed -10 %', fontsize=9)
ax[0].set_ylabel('utilisation (1 = limit)'); ax[0].legend(fontsize=8, loc='upper left')
fig.suptitle(f"option B: RS04 hip pitch (0.85 belt), RS04 knee (four-bar N 0.77-1.28), RS06 hip roll (2.5 belt), RS06 hip yaw (1.5 belt); "
             f"total {R['v11.1_w0x1']['mass_kg']} kg, actuators {int(100*R['v11.1_w0x1']['actuator_fraction'])} %", fontsize=9)
fig.tight_layout(); fig.savefig(f'{E}/media/b_utilisation.png', dpi=100)
