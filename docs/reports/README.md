# 중간 보고서

Raptor_Interim_Report_20261009.pdf: 2026-10-09 paused 체크포인트와증거87-163기준7쪽보고서. 실행결과와미검증/재개점을구분한다. 보고서근거링크는2ce9621커밋에고정되어있다.

Mac에서reportlab과Arial Unicode폰트를사용해 `python3 docs/reports/build_interim_report.py`로재생성한다. 출력은ignored `output/pdf/`에생긴다. 문서의수치/판정이변하면근거대조후수정하고PDF를렌더링해확인해야한다. 다른OS의폰트경로는별도조정이필요하다.
