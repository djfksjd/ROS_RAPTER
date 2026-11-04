"""Read-only diagnosis of the pre-step clamp vs post-step reported current."""
import json
import time
from unittest.mock import patch
import numpy as np
import jump_probe as jp


class CurrentRecorder(jp.Recorder):
    def __init__(self, env):
        super().__init__(env)
        for key in ('speed_before', 'speed_after', 'ratio_before', 'ratio_after', 'current_using_clamp_state_a'):
            self.rows[key] = []

    def clamp(self):
        self.qd_before = self.env.rz._motor_speed(self.env.data).copy()
        self.ratio_before = self.env.rz.ratio(self.env.data.qpos[self.env.rz.q_adr_m]).copy()
        return super().clamp()

    def sample(self, env, dt):
        super().sample(env, dt)
        rz = env.rz; tau = self.rows['drive_tau_nm'][-1]
        before = np.abs(np.where(tau*self.qd_before >= 0, tau/(self.ratio_before*rz.eta), tau*rz.eta_b/self.ratio_before))/rz.kt
        values = dict(speed_before=self.qd_before.copy(), speed_after=rz._motor_speed(env.data).copy(),
                      ratio_before=self.ratio_before.copy(), ratio_after=rz.ratio(env.data.qpos[rz.q_adr_m]).copy(),
                      current_using_clamp_state_a=before)
        for key,value in values.items(): self.rows[key].append(value)


def main():
    start = time.process_time(); source = json.loads((jp.E/'eval/search_summary.json').read_text())
    rows = []
    for mode,label in [('vertical','extreme'),('forward','score')]:
        env = jp.make()
        with patch.object(jp, 'Recorder', CurrentRecorder):
            result,a = jp.rollout(env, source['selected'][mode][label]['x'], capture=True)
        jp.save_trace(jp.E/f'eval/current_audit_{mode}.npz', a, env)
        ratio = a['current_a']/a['current_peak_a']; i,j = np.unravel_index(np.argmax(ratio), ratio.shape)
        rows.append(dict(mode=mode, joint=env.active[env.rz.idx[j]], time_s=float(a['t'][i]),
            current_a=float(a['current_a'][i,j]), current_using_clamp_state_a=float(a['current_using_clamp_state_a'][i,j]),
            current_peak_a=float(a['current_peak_a'][i,j]), speed_before=float(a['speed_before'][i,j]),
            speed_after=float(a['speed_after'][i,j]), ratio_before=float(a['ratio_before'][i,j]),
            ratio_after=float(a['ratio_after'][i,j]), torque_nm=float(a['drive_tau_nm'][i,j]),
            limit_nm=float(a['limit_nm'][i,j]), overcurrent_flag=bool(a['overcurrent'][i,j]),
            max_current_using_clamp_state_over_peak=float((a['current_using_clamp_state_a']/a['current_peak_a']).max())))
        env.close()
    (jp.E/'eval/current_audit.json').write_text(json.dumps(dict(rows=rows, cpu_s=time.process_time()-start), indent=2)+'\n')

if __name__ == '__main__': main()
