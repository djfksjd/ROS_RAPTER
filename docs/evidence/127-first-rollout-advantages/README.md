# 127 — 첫 PPO 배치의 value/advantage 계측

train_symmetric의실제초기화/4환경/seed0/1024rollout경로를그대로사용했다. 계측은SB3train()진입직후optimizer실행전에rolloutbuffer를저장하고정상적인진단종료로멈춘다. 4096개씩random/기존catalogue명령을비교했다. 두모델각24초기tensor정확일치·_n_updates=0·원모델미교체. 최초하니스는workerimport중폴더중복생성으로실패, 정책갱신없이종료·실패코드/로그보존. 수정진입점에서는주프로세스만폴더생성한다.

random첫배치는3.5≤vx명령≤4.5 및|yaw명령|≥.5인목표근방회전0개다. 모든movingturn1285개·nearstraight250개, 표준화advantage양수비율49.57/82.0%. 이는첫배치의표본이며전체64k나프로젝트전체에목표샘플이없다는주장은아니다. 명령동시에횡속도가있는경우도있어원평가4/0/yaw와다르다. 원초기critics전역explainedvariance는GAEtarget대비0.790528로, 전역붕괴가확인된것은아니다.

catalogue첫배치는목표근방회전200개·nearstraight1537개·정지2359개, 회전표준화advantage양수69.5%, GAEtarget대비EV0.872618이다. 명령/에피소드프로토콜이바뀌므로같은상태의critic우열실험은아니다. advantage의부호만으로미래정책의성공을보장하지않는다. values/returns는VecNormalize훈련정규화단위이고returns는bootstrap을포함한GAEtarget이며raw평가reward나완전한MonteCarloreturn과같지않다. 추가raw Gaussian action·observation·episode-start·logprob를npz에보존했다.

다음은같은sym_init/보상/물리/seed에서기존카탈로그경로와KL.005를결합한64k후보를117random-KL기준과비교한다. 카탈로그만변경했던과거98/113실패를보존하며현재실험의근거는첫배치목표데이터분포차이다. 기존기준선미교체·전체goal active.
