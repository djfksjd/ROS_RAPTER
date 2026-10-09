# 103 — 반사 배우 branch clipping 진단

카탈로그 정책·seed45001에서yaw−1/0/+1,10초씩각 shadow on/off 총6rollout. 각회전구간100개정책입력에대해원branch와반사branch의clamp이전평균을기록했다. 계산만추가했고원정책출력·물리는바꾸지않았다. 전체행동·qpos·qvel이 shadow on/off에서정확히일치,전도0/6.

의심한메커니즘: 두branch가반대부호로±1밖에있으면clip뒤평균은0인데각출력방향의국소미분도0이될수있다. 실행 autograd검사로이수학적사례는확인했다. 그러나 실제3조건×100입력에서두branch가동시에clip밖인비율은모든8축에서0이다. 따라서현재표본에서는이메커니즘으로회전저추종을설명하지못한다. 배우clamp·분포를수정하지않는다. 다른출력·상태·공유가중치의학습까지영구정지했다고주장하지않는다.

검증: 전체129중66통과·63환경의존생략,sim집중8통과. 원본raw branch와physical control은Git제외runs/screen_logs에보존하고manifest에해시를남겼다. 이진단은회전통과·실물검증증거가아니다.
