# 디지티그레이드 다리와 Morphloom 외형 — 2026-09-28

브랜치 `feature/digitigrade-appearance`. 사용자 결정: 외형 작업, 다리 비율 변경 포함, morphloom으로 직접 제작.
기존 설계는 `leg_design:=legacy`(기본값)로 그대로 재현된다. 새 설계는 `leg_design:=digitigrade`.

## 1. 물리 설계 (evidence/63)

- 능동축 10개 유지. 발 링크 = 20° 앞으로 기운 발등뼈 0.24m + 강체 패드 0.15×0.10m + 기존 수동 발가락 12개.
- 허벅지 0.32→0.26m. 링크 질량은 그대로, 새 관성은 박스 공식.
- 앞뒤 지지 여유(패드만, 발가락 제외): 기존 crouch 4.2cm → 새 기본 자세(hip -0.10, knee 0.50, ankle -0.40) 7.1cm.
  발등뼈를 더 기울이면 패드가 앞으로 가서 여유가 줄었다(β 0.45에서 2.1cm).
- MuJoCo: 정지 기울기 0.0131→0.0047rad, 흔들기 9/9 유지, 보폭 걷기 kv30 깨끗한 걸음 통과(+5.5m/60s),
  kv100에서도 전도 없음(미끄럼/보폭 0.33로 기준 0.25는 미달). 발목 ±0.05 계단 입력은 여전히 전도.
- 꼬리 0.5→0.95m 후 재검증(evidence/66): 정지 기울기 0.0069rad, 흔들기 9/9, kv30 +5.82m 미끄럼/보폭 0.20,
  kv100 전도 없음(0.34, 기준 미달). 결과는 꼬리 변경 전과 같은 수준이다.
- Gazebo: 올바른 spawn(0.1075m)에서 정지 기립 확인(evidence/64, 꼬리 변경 후 evidence/66).
  **정정:** 처음 기록한 Gazebo 결과는 launch 파일이 `leg_design`을 전달하지 않아 legacy 모델이었다. 수정 후 재검증했다.

## 2. 외형 제작 경로

| 단계 | 파일 | 결과 |
|---|---|---|
| 설계 도면 | `modeling/morphloom/design_drawing.py` | Xacro collision의 측면·정면 투영(1px=1mm) |
| 부품 IR | `modeling/morphloom/raptor_parts.py` | 349개 부품(쐐기형 머리, 꼬리 16마디), id=`<링크>__<부품>`, URDF zero pose 배치 |
| 작업 파일 | `modeling/morphloom/make-job.ts` (morphloom `work/raptor/`에서 실행) | 근거 팩: 설계 도면(CAD, measured), 콘셉트(AI, 스타일만) |
| 빌드 | morphloom a200aa4 `npm run morphloom -- build` | review-pass (evidence 64 run report) |
| 분할 | `modeling/morphloom/split_links.py` (Blender 5.2) | 링크별 GLB 23개 → `meshes/digitigrade/` (8.5MB, 텍스처 제외) |

- morphloom 판정은 **review 초안**이다. delivery 릴리스에 필요한 fidelity·부품 분해 계약은 아직 없다.
- 입력 적합도는 morphloom의 결정적 공식을 실제 픽셀로 계산했다(임의 점수 입력 없음).
- glTF는 Y-up이고 Gazebo가 변환하지 않아 각 visual에 `rpy="pi/2 0 0"`을 둔다. RViz에서도 같은 방향으로 표시됨(evidence/66).
- 줄무늬 음영 원인: micro normal·roughness 맵이 `KHR_texture_transform` 타일링(예 12×48)에 의존하는데
  Gazebo가 이를 적용하지 않아 늘어난 무늬로 보였다. 탄젠트 추가로는 해소되지 않았다.
  시뮬레이션용 GLB는 이미지 없이 출력한다(색·금속·거칠기 계수는 유지). 해소 확인 evidence/66.
- 외부 3D 생성 모델(Meshy/Tripo/TRELLIS)은 사용하지 않았다. Tripo 영상 프레임은 비율 참고용으로만 봤다.

## 3. 구분

- 생성 이미지: README 콘셉트, Tripo 영상(참고).
- Blender 렌더: `evidence/64-digitigrade-morphloom-blender-*.png`(같은 메시, 같은 crouch).
- 실제 Gazebo: `evidence/64-digitigrade-morphloom-gazebo.png`, `evidence/66-gazebo-*.png`. RViz: `evidence/66-rviz-digitigrade-glb.png`. 정지 기립만 확인. 보행은 아직 Gazebo에서 검증하지 않았다.

## 재현

```bash
bash sim/generate_urdf.sh sim/raptor_digitigrade.urdf leg_design:=digitigrade passive_toes:=true sensors:=true
.venv-sim/bin/python modeling/morphloom/design_drawing.py sim/raptor_digitigrade.urdf ~/Documents/morphloom/work/raptor/sources/design.png
.venv-sim/bin/python modeling/morphloom/raptor_parts.py sim/raptor_digitigrade.urdf ~/Documents/morphloom/work/raptor/assembly.json
# morphloom 저장소에서 (concept.png는 docs/assets/raptor-target-concept.png 복사, make-job.ts는 work/raptor/로 복사)
npx vite-node work/raptor/make-job.ts && npm run morphloom -- build --job work/raptor/job.json --out outputs/raptor-NNN
# raptor 저장소에서
/Applications/Blender.app/Contents/MacOS/Blender --background --python modeling/morphloom/split_links.py -- \
  ~/Documents/morphloom/outputs/raptor-NNN/asset.glb sim/raptor_digitigrade.urdf src/raptor_description/meshes/digitigrade /tmp/preview
RAPTOR_PASSIVE_TOES=true bash scripts/start_local.sh --experiment leg_design:=digitigrade crouched_start:=true crouch_hip_pitch:=-0.10 spawn_z:=0.1075
```
