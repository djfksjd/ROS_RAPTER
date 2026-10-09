from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape
import json
ROOT=Path.cwd();OUT=ROOT/'output/pdf/Raptor_Interim_Report_20261009.pdf'
pdfmetrics.registerFont(TTFont('KR','/Library/Fonts/Arial Unicode.ttf'))
NAVY=colors.HexColor('#132C42');TEAL=colors.HexColor('#137F86');GRAY=colors.HexColor('#526371');LIGHT=colors.HexColor('#EDF3F6');ORANGE=colors.HexColor('#B75C20');W,H=A4;CW=W-84
styles={
 'body':ParagraphStyle('body',fontName='KR',fontSize=10,leading=16,textColor=NAVY,wordWrap='CJK',spaceAfter=8),
 'small':ParagraphStyle('small',fontName='KR',fontSize=8.5,leading=13,textColor=GRAY,wordWrap='CJK',spaceAfter=5),
 'title':ParagraphStyle('title',fontName='KR',fontSize=25,leading=34,textColor=NAVY,spaceAfter=16),
 'h1':ParagraphStyle('h1',fontName='KR',fontSize=19,leading=27,textColor=NAVY,spaceAfter=12),
 'h2':ParagraphStyle('h2',fontName='KR',fontSize=12.5,leading=19,textColor=TEAL,spaceBefore=12,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='KR',fontSize=9,leading=14,wordWrap='CJK',textColor=NAVY),
 'headcell':ParagraphStyle('headcell',fontName='KR',fontSize=9,leading=14,wordWrap='CJK',textColor=colors.white),
 'kpi':ParagraphStyle('kpi',fontName='KR',fontSize=22,leading=30,textColor=TEAL,wordWrap='CJK'),
}
def P(t,style='body'):return Paragraph(t,styles[style])
def table(headers,rows,widths):
 data=[[P(escape(str(x)),'headcell') for x in headers]]+[[P(str(x),'cell') for x in row] for row in rows]
 t=Table(data,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('ROWBACKGROUNDS',(0,1),(-1,-1),[LIGHT,colors.white]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#D4DFE5'))]));return t
story=[]
def add(t,style='body'):story.append(P(t,style))
def section(n,title):story.append(PageBreak());add(f'{n:02d}  {title}','h1')
def note(t):
 t=Table([[P(t)]],colWidths=[CW]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#C9D9E1')),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t)
class Bars(Flowable):
 def __init__(self):Flowable.__init__(self);self.width=CW;self.height=155
 def draw(self):
  c=self.canv;c.setFont('KR',9);start=143
  for label,a,b in [('회전량',20,16),('속도 범위',11,12),('두 기준 동시',11,10)]:
   c.setFillColor(NAVY);c.drawString(0,start,label)
   for j,(count,col) in enumerate([(a,TEAL),(b,colors.HexColor('#9AAEBB'))]):
    y=start-13-j*14;c.setFillColor(LIGHT);c.rect(95,y,320,9,fill=1,stroke=0);c.setFillColor(col);c.rect(95,y,320*count/20,9,fill=1,stroke=0);c.setFillColor(NAVY);c.drawString(425,y-1,f'{count}/20')
   start-=48
class Architecture(Flowable):
 def __init__(self):Flowable.__init__(self);self.width=CW;self.height=168
 def draw(self):
  c=self.canv
  labels=[('운영자 명령 / 장면 정보','상위 AI: 허용 행동 선택'),('실행 가능성 검사 / STOP 우선','행동 전환과 실패 처리'),('보행·자세 정책 / 상태 피드백','관절 목표각 생성'),('로컬 관절 서보 / 구동 한계','PD 토크와 물리·접촉 계산')]
  for i,(a,b) in enumerate(labels):
   y=133-i*43;c.setFillColor(LIGHT if i%2==0 else colors.HexColor('#F8FAFB'));c.roundRect(0,y,CW,35,5,fill=1,stroke=0);c.setFillColor(NAVY);c.setFont('KR',10);c.drawString(12,y+21,a);c.setFillColor(GRAY);c.setFont('KR',8.5);c.drawString(12,y+8,b)
   if i<3:c.setStrokeColor(TEAL);c.line(CW/2,y,CW/2,y-8)

add('RAPTOR / DEVELOPMENT REVIEW','small');add('랩터 로봇<br/>개발 중간 보고서','title')
add('2026년 10월 9일 체크포인트 기준 · 증거 87-163 · 사용자 요청으로 일시정지','small')
kpis=Table([[P('20/20','kpi'),P('11/20','kpi'),P('1/21','kpi')],[P('4 m/s 회전량 기준\n개발 조건 통과','cell'),P('회전 중 속도까지\n동시 통과','cell'),P('원시작 기립\n기존 개발 성공','cell')]],colWidths=[CW/3]*3);kpis.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]));story.append(kpis);story.append(Spacer(1,15))
note('현재 판정: 회전 방향·회전량 제어에는 진전이 있으나 속도 유지, 원시작 기립, 가속·감속·STOP을 함께 만족하는 통합 정책은 아직 확보하지 못했다. 기준 정책을 교체하지 않았으며 실물 성능이나 40 km/h 달성을 주장하지 않는다. [S1, S3, S8]')
add('이번 개발에서 확인한 것','h2')
add('후방 골반 8축 구성은 가정한 구동기·열 모델의 60초 시험에서 직진 지속 능력을 보였다. 이후 회전 보정을 통해 개발 조건의 회전량 기준을 모두 통과했지만, 20회 중 9회는 속도 기준이 미달이었다. [S2, S3]')
add('정적 평형을 찾고 그 주변에서 관측 기반 제어를 유지하는 데 성공했다. 하지만 원래 시작 자세·속도·접촉 상태에서 그 평형으로 들어가는 과정은 여전히 실패가 많다. 실패 조건을 제외하거나 초기조건을 쉽게 바꾸지 않았다. [S4-S8]')
add('읽는 기준','h2')
add('이 보고서의 수치는 MuJoCo 및 코드·로그 검증 결과다. 전도 없음, 목표 추종, 가상 발열 기준, 실물 검증은 각각 별개다. 서로 다른 시험군의 성공률을 하나의 성능 수치로 합산하지 않는다. 기록의 “가상 부하”는 섭씨 온도가 아니다.','small')

section(2,'시스템과 기구 구성')
add('상위 명령 선택과 하위 운동 제어를 분리한다. 보행 정책은 관절 목표각을 출력하고, 로컬 서보가 구동기 한계 안에서 토크를 계산한다. 상위 AI가 직접 모터 토크를 생성하는 운영 경로는 사용하지 않는다. [S10, S11]')
story.append(Architecture());story.append(Spacer(1,10))
story.append(table(['항목','현재 개발 기준 / 구분'],[
 ['능동축','프로젝트 기본 10축, 사용자 선택 12축 변형과 비교한다. 이번 고도화 목표는 후방 골반 8축 후보이며 다른 구조로 대체하지 않는다.'],
 ['8축 구성','각 다리의 hip roll·hip pitch·knee pitch 3축 + 꼬리 yaw·pitch 2축. ankle pitch는 무릎에 연동, ankle roll은 잠금 조건.'],
 ['수동 요소','발가락 수동 회전관절, 스프링·접촉·연동 구속은 능동 모터 수와 구분한다.'],
 ['질량·배치','8축 동역학 모델 약 11.29 kg, 설계 강성 기준 질량 11.36 kg. 골반 위치는 몸통 대비 뒤로 0.2 m인 후보.'],
 ['관측·주기','원래 46개 관측 / 정책 20 ms / 물리·서보 1 ms. 기립 관측기는 기존 관측 중 41개를 사용한다.']],[100,CW-100]))
add('실행 환경','h2')
add('ROS 통합은 Ubuntu 24.04 / ROS 2 Jazzy / Gazebo Harmonic의 ARM64 컨테이너 경로다. 보행 학습·평가는 Mac의 별도 MuJoCo + Stable-Baselines3 PPO CPU 환경을 사용한다. 병렬 시뮬레이션 환경과 Docker 컨테이너 수는 같은 뜻이 아니다. [S10]')
add('현재 위치 서보·시뮬레이션 STOP은 실물 전원 차단이나 고속 안전 정지를 보장하지 않는다. 스프링·걸쇠 기능은 동역학 모델에서 평가하며 제작 치수·내구성·실제 부품의 반복 운전 성능은 미확정이다. [S1, S10]','small')

section(3,'직진·회전 개발 결과')
add('직진 능력은 확인했지만 회전 중 속도 유지가 남아 있다.','h2')
story.append(table(['시험','결과','해석'],[
 ['후방 8축 / 7 m/s 명령','추가 30개 출발: 전도 0, 가상 열 기준 30/30. 실제 속도 6.167-6.194 m/s.','60초 가정 모델 시험. 7 m/s 명령의 속도 오차 10% 기준은 미달. [S2]'],
 ['기본 8축 / 6 m/s 명령','추가 30개 출발: 전도 0, 가상 열 기준 29/30.','선별 결과가 모든 시작 조건으로 일반화되지 않았다. [S2]'],
 ['40 km/h / 약 11.1 m/s','구조 비교에서 통과 후보 없음.','검토한 구성·정책·가정의 결과이며 모든 설계의 불가능 증명이 아니다. [S2]']],[126,191,CW-317]))
add('회전 입력 필터 대조: 동일 개발 조건 20회씩 [S3]','h2')
story.append(Bars());add('청록: 무필터 보정 / 회색: 시정수 0.1초 필터. 양쪽 모두 전도 0/20. 숫자는 각 시험 기준을 통과한 횟수이며 실물 성공률이 아니다.','small')
add('원 기준은 실제 명령 4 m/s, yaw ±0.5·±1 rad/s, 2초 회전이다. 회전량 오차 ±20%와 회전 중 속도 3.6-4.4 m/s를 동시에 검사했다. 무필터는 11/20, 필터는 10/20으로 동시 통과가 줄어 필터를 채택하지 않았다. 새 독립 최종 시험은 아니다. [S3]')
note('회전량 통과 20/20을 “회전 기능 완성”으로 보고하지 않는다. 통합 후보에는 직진·가상 발열의 퇴행 없음과 기립·가감속·STOP 전환 검증이 추가로 필요하다. [S1, S3]')

section(4,'기립과 초기 접근의 핵심 병목')
add('평형 유지와 평형 진입은 다른 문제다.','h2')
story.append(table(['단계','실제 확인','판정'],[
 ['153 / 관측 기반 기립','평형 근처 개발 20조건 × 60초 통과. 최대 이동 0.00252 m, 기울기 약 4.53도, 최대 가상 부하 0.90250.','원초기 4조건 모두 실패. 몸통 속도는 시뮬레이터 관측. [S4]'],
 ['156 / 제동 접근','원시작 개발 seed 83002의 60초 기립 성공·3,000행 재현. 고정 후 새 20조건 모두 실패.','새 조건 0/20, 일반화 미달·미채택. [S5]'],
 ['161 / 초기구간 검사','21개 개발 시작 중 16개는 첫 0.6초에 이동·기울기·비행 기준을 이미 위반.','전환 이후 제어만으로 원시험을 통과시킬 수 없음. [S6]'],
 ['162 / hip-tail 분배','105회 대조. 감속 보정은 새 기립 성공 없음. 계측 두 조건에서 토크 포화 없이 발이 뜸.','수평 감속과 수직 지지의 결합 문제 확인. [S7]'],
 ['163 / 무릎 지지','48개 pulse 저장 + 구간 보정 42회. 대조 1/21 기존 성공 유지, paired 0/21.','작은 보정만으로 개선되지 않음. [S8]']],[100,255,CW-355]))
add('무릎 입력 반응은 접촉 상태에 따라 달라졌다.','h2')
add('0.10·0.20초와 0.24초에서 같은 무릎 입력이 수직 속도에 미치는 영향의 부호가 달랐다. 감속 보정군에 제한된 무릎 폄 보정을 넣자 두 조건의 초기 비행 표본은 4→3, 9→7로 줄었지만 각각 1.22초·0.76초에 전도했다. 비행 감소는 기립 통과와 같지 않다. [S8]')
note('원 기립 기준: 60초 유지, 최대 이동 0.1 m 이하, 기울기 10도 이하, 비행 표본 0, 마지막 40초 가상 부하 1 이하. 실패한 짧은 시험에는 정상상태 열 값을 부여하지 않는다. [S1, S6-S8]')

section(5,'채널·논문 조사와 랩터 적용')
story.append(table(['조사 항목','확인 범위'],[
 ['Engiuniverse 목록','121개: 일반 영상 79개 + Shorts 42개'],
 ['자막 전문 분석','43편 전체 구간 검토'],
 ['영상·음성 완시청','0편. 자막 분석과 화면·음성 완시청을 구분'],
 ['남은 내용 확인','78개. 자막 HTTP 429 / 영상·블로그 접근 실패 포함'],
 ['근거 저장','출처 258개 / 근거 188개 / 주장 188개. 독립 실험 수가 아님']],[150,CW-150]))
add('조사 결과를 개발 검증 항목으로 연결했다. [S9]','h2')
story.append(table(['내용','랩터에 연결할 항목','적용 한계'],[
 ['몸·다리·스프링','구동기 응답 식별, 스프링 하중 경로, 접촉 신호','형상·강성·실물 피로수명 별도 검증'],
 ['Isaac / Sim2Real','좌표·단위·관측·지연·마찰·접촉 일치','기존 MuJoCo 경로 이관은 결정하지 않음'],
 ['Agentic / ER2','행동 API, 작업 ID, 결과 확인, STOP 우선','언어 판단이 로컬 균형 제어를 대신하지 않음'],
 ['촉각·힘 / T-Rex','발 닿음·하중·미끄럼, 원시각 이력 구분','손 조작 실험 성능을 발 보행으로 이전하지 않음'],
 ['ANYmal / OP3 회복','별도 회복 정책, 접촉 순서·핵심 자세','신체와 자유도가 다른 로봇의 성능 수치 제외'],
 ['AnyBipe / Bimo 코드','고정 평가 지표, 지연·유격·속도-토크 모델','코드 존재는 실제 랩터 부품 식별의 대체가 아님']],[116,198,CW-314]))
add('추가 영어·중국어 원 논문, 대학 논문 초록, 공식 GitHub, 개발자 Reddit 경험을 대조했다. 직접 읽은 절·초록·README·검색 스니펫을 각각 표시했으며 외부 벤치마크를 재현한 것으로 계산하지 않는다. [S9]','small')
add('전편 조사 요청은 미완료다. 현재 조사 보존본에는 영상별 범위, 미확인 큐, 출처·근거·주장 ledger가 있다. 자동자막 원문·외부 논문 PDF·사용자 노트 원본은 공개 저장소에 넣지 않았다. [S1, S9]','small')

section(6,'재개 계획과 체크포인트')
add('다음 한 단계: 초기 접촉 상태의 결합 반응 측정','h2')
add('원시작을 재생해 0.10·0.20·0.24초의 동일 상태에서 고관절·무릎·꼬리 입력이 수평 속도, 세계 수직 속도, pitch 각속도에 미치는 응답 행렬을 측정한다. 감속과 접촉 유지가 함께 가능한 입력 영역을 확인한 뒤 최소 제어 변경을 비교한다. 진단용 실제 상태를 온라인 관측기에 주입하지 않는다. [S8]')
story.append(table(['검증 순서','완료를 인정할 조건'],[
 ['1. 원시작 기립','원초기 속도·관절 노이즈·접촉 조건을 유지. 60초 위치·기울기·비행·가상 부하 gate 통과.'],
 ['2. 통합 회전 후보','4 m/s, 좌우 ±0.5·±1 rad/s에서 회전량 오차 20% 이하. 속도 기준 동시 통과, 전도 없음.'],
 ['3. 퇴행·독립 평가','직진과 가상 발열 퇴행 없음. 이미 본 개발 조건과 새 독립 시작 조건을 분리.'],
 ['4. 동작 전환','기립·느린 걷기·가속·주행·감속·STOP. 실제 주행 속도에서 정지 후 안정 유지.']],[135,CW-135]))
add('기술·제작 의사결정의 남은 항목','h2')
add('실제 모터의 토크-속도·제동·열 특성, 전달계 손실·유격, 스프링·걸쇠 제작 구조, 회복의 실제 가동 범위, 속도·접촉 상태추정, 지형·적재량은 아직 최종 사양으로 확정되지 않았다. 기구 후보와 시뮬레이션 값은 실물 구매·제작 사양과 구분한다. [S1, S9-S11]')
add('안전하게 이어가기 위한 보존 상태','h2')
add('목표 상태는 paused다. 기준 scr_a8_h2와 작업 정책 131은 별도로 보존했고 기준선을 교체하지 않았다. 핵심 복원 파일 12개는 SHA256 검사 및 빈 임시 폴더 복원 후 재검증을 통과했다. 환경 버전과 실패 기록도 함께 남겼다. [S1]')
add('코드 검사: 시뮬레이션 환경 154개 모두 통과. AI 환경은 77개 통과·77개 환경 생략. 이는 목표 성능 달성이나 실물 인증과 별개의 코드 검증이다. 이 보고서 작성 중 실험을 재개하지 않았다. [S1]')

section(7,'근거·재개 문서 안내')
add('기준 시점: 2026-10-09 일시정지 체크포인트. 보고서는 아래 저장소 문서·원시 결과를 요약했다. 링크는 보존 커밋 2ce9621의 내용으로 고정해 이후 변경과 구분한다.','small')
refs=[('S1','일시정지 체크포인트·검사·복원','docs/checkpoints/2026-10-09-pause/README.md'),('S2','8축/12축 구조 비교·30개 출발 재확인','docs/evidence/87-structure-comparison/README.md'),('S3','4 m/s 회전 입력 필터 대조','docs/evidence/143-turn-yaw-input-filter/README.md'),('S4','평형 근처 관측 기반 기립','docs/evidence/153-stand-observer/summary.json'),('S5','원시작 제동 접근과 독립20조건','docs/evidence/156-stand-peak-envelope/README.md'),('S6','초기0.6초 위반과 평형 후보','docs/evidence/161-equilibrium-family-and-prefix/README.md'),('S7','hip-tail 분배·1ms 접촉 계측','docs/evidence/162-early-hip-tail-capture/README.md'),('S8','무릎 pulse·수직 지지 대조','docs/evidence/163-knee-vertical-support/README.md'),('S9','채널 조사·원 자료 추가 분석','docs/research/engiuniverse-20261009/research-refresh.ko.md'),('S10','ROS·Mac 시뮬레이션 실행 환경','docs/local-development.md'),('S11','동역학 환경·관측·구동 한계 코드','sim/rl/run_env.py')]
rows=[]
for sid,title,path in refs:
 url='https://github.com/djfksjd/ROS_RAPTER/blob/2ce9621ae310e7529e26f439a64ef521d53ebe7e/'+path
 rows.append([sid,title,f'<link href="{url}" color="#137F86">근거 열기</link>'])
story.append(table(['표기','원 자료','링크'],rows,[48,CW-122,74]))
add('원시 근거를 확인할 때','h2')
add('각 evidence 폴더의 README와 verification·manifest·원시 JSON/압축 JSONL을 함께 읽는다. 측정 시간, 시작 seed, 실제 실행한 행동, 전도·미달·부하 null을 보존한 기록이 판정 근거다. 서로 다른 구조·시작 조건·학습 seed의 수치를 같은 대조 시험으로 취급하지 않는다.')
add('재개 문서의 “다음 한 단계”부터 진행하며 사용자 재개 요청 전에는 새 실험을 시작하지 않는다. 영상 완시청, 실물 안정성, 목표 속도, 적재량, 제작 도면, 고속 STOP은 미검증 상태를 유지한다.','small')

def footer(c,doc):
 c.saveState();c.setFillColor(TEAL);c.rect(42,H-34,28,3,fill=1,stroke=0);c.setFont('KR',8);c.setFillColor(GRAY);c.drawString(79,H-34,'RAPTOR  |  INTERIM DEVELOPMENT REPORT');c.setStrokeColor(colors.HexColor('#D4DFE5'));c.line(42,36,W-42,36);c.drawString(42,23,'2026-10-09 checkpoint  |  시뮬레이션·개발 기록 기준');c.drawRightString(W-42,23,str(doc.page));c.restoreState()
doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=53,bottomMargin=49,title='랩터 로봇 개발 중간 보고서 - 2026-10-09',author='Raptor Project',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
