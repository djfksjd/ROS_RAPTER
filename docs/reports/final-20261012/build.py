"""Render report.ko.md to PDF and HTML (ReportLab). Adapted from the 1.0 build script; self-contained in this folder."""
from pathlib import Path
import re,html,json,hashlib,shutil
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from PIL import Image as PIL
R=Path(__file__).resolve().parent;OUT=R.parent
pdfmetrics.registerFont(TTFont('AU','/Library/Fonts/Arial Unicode.ttf'))
style={k:ParagraphStyle(k,fontName='AU',fontSize=s,leading=l,spaceAfter=a,wordWrap='CJK',textColor=colors.HexColor('#20333f')) for k,s,l,a in [('body',9.3,14.5,9),('h1',18,25,15),('h2',12,18,9),('cell',8.0,12,0),('eq',8.7,14,11),('cap',8,12,8)]}
style['eq'].backColor=colors.HexColor('#edf3f6');style['eq'].borderPadding=7
W=A4[0]-104
source=(R/'report.ko.md').read_text();pages=source.split('---PAGE---');story=[];tableissues=[]
def fmt(t):
 t=html.escape(t)
 return re.sub(r'(https?://[^\s<]+)',r'<link href="\1" color="#215775">\1</link>',t)
def P(t,k='body'):return Paragraph(fmt(t),style[k])
toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc',fontName='AU',fontSize=8.5,leading=13.7,leftIndent=0,firstLineIndent=0,spaceBefore=2,textColor=colors.HexColor('#20333f'),wordWrap='CJK')]
for pi,page in enumerate(pages):
 if pi==1:
  story.extend([PageBreak(),P('목차','h1'),P('항목과 페이지 번호를 누르면 해당 절로 이동합니다. 표와 수식은 모델 값·설명용 가정·미해결 항목을 함께 읽으세요.','cap'),toc,PageBreak()])
 elif pi>1:story.append(PageBreak())
 lines=page.strip().splitlines();i=0
 while i<len(lines):
  line=lines[i].strip()
  if not line:i+=1;continue
  if line.startswith('# '):
   title=line[2:];display=(f'{pi:02d}. '+title) if pi>0 and not title.startswith('참고문헌') else title
   p=P(display,'h1');p.bookmark='s'+str(pi);p.chapterTitle=display;story.append(p);i+=1;continue
  if line.startswith('### ') or line.startswith('## '):story.append(P(line.lstrip('# '),'h2'));i+=1;continue
  if line.startswith('!['):
   m=re.match(r'!\[(.*?)\]\((.*?)\)',line);assert m,line
   img=R/m.group(2);pw,ph=PIL.open(img).size;sc=min(W/pw,240/ph)
   story.extend([Image(str(img),width=pw*sc,height=ph*sc),P(m.group(1),'cap')]);i+=1;continue
  if line.startswith('```'):
   i+=1;eq=[]
   while i<len(lines) and not lines[i].startswith('```'):eq.append(lines[i]);i+=1
   story.append(Paragraph('<br/>'.join(html.escape(x).replace(' ','&nbsp;') for x in eq),style['eq']));i+=1;continue
  if line.startswith('|'):
   rows=[]
   while i<len(lines) and lines[i].strip().startswith('|'):
    l=lines[i].strip();i+=1
    if re.match(r'^\|[\s:|\-]+\|$',l):continue
    rows.append([x.strip() for x in l.strip('|').split('|')])
   assert len(rows)>=2 and any(re.match(r'^\|[\s:|\-]+\|$',x.strip()) for x in lines[i-len(rows)-1:i]),('table without separator',rows[0])
   n=len(rows[0])
   for ri,row in enumerate(rows):
    if len(row)!=n:tableissues.append([pi,ri,n,row])
   assert all(len(row)==n for row in rows),tableissues
   widths=[W/n]*n
   if n==2:widths=[W*.35,W*.65]
   if n==3:widths=[W*.26,W*.32,W*.42]
   tab=Table([[P(x,'cell') for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
   tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce8ee')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8fa')]),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#cad5dc')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
   story.extend([tab,Spacer(1,10)]);continue
  para=[line];i+=1
  while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','![')):para.append(lines[i].strip());i+=1
  story.append(P(' '.join(para)))
class Doc(SimpleDocTemplate):
 def afterFlowable(self,f):
  if hasattr(f,'bookmark'):
   self.canv.bookmarkPage(f.bookmark);self.canv.addOutlineEntry(f.chapterTitle,f.bookmark,0,False)
   self.notify('TOCEntry',(0,f.chapterTitle,self.page,f.bookmark))
def footer(c,d):
 c.setFont('AU',7);c.setFillColor(colors.HexColor('#526774'));c.drawString(52,812,'RAPTOR / FINAL ENGINEERING 1.8 · 2026-10-12');c.drawString(52,29,'모델·설계 후보·검증 한계 통합 | 실물 성능 보증 아님');c.drawRightString(A4[0]-52,29,str(d.page))
pdf=R/'RAPTOR_최종설계검증보고서_20261012.pdf'
Doc(str(pdf),pagesize=A4,rightMargin=52,leftMargin=52,topMargin=58,bottomMargin=52,title='RAPTOR 최종 설계·검증 보고서 1.8',author='Raptor Project').multiBuild(story,onFirstPage=footer,onLaterPages=footer)
shutil.copy2(pdf,OUT/pdf.name)
# Markdown and HTML maintain a portable, editable copy with relative asset links.
try:
 import markdown
 h=markdown.markdown(source.replace('---PAGE---','<hr>'),extensions=['tables','fenced_code'])
except ImportError:
 h='<pre>'+html.escape(source)+'</pre>'
(R/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>RAPTOR 최종 설계 검증 1.8</title><style>body{max-width:1000px;margin:45px auto;font:16px/1.8 sans-serif;color:#20333f}img{max-width:100%}pre{white-space:pre-wrap;background:#edf3f6;padding:15px}td,th{border:1px solid #ccd;padding:8px}table{border-collapse:collapse}hr{margin:50px 0}a{color:#215775}</style><body>'+h+'</body></html>')
print(json.dumps({'pdf':str(pdf),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()},ensure_ascii=False))
