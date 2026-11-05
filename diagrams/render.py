"""Regenerate AP examples as editable SVG and optimized WebP."""
import io,json,sys
from fractions import Fraction
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from detection_ap import class_ap,report
plt.rcParams.update({'svg.hashsalt':'wifa313','svg.fonttype':'none','text.color':'#163548','font.size':12})
def case(n): return json.loads((D.parent/'examples'/f'{n}.json').read_text())
def chart(title,sub):
 f,a=plt.subplots(figsize=(7.5,4.5),facecolor='#eef6f7')
 f.subplots_adjust(left=.14,right=.94,bottom=.16,top=.73)
 f.text(.055,.91,title,fontsize=20,weight='bold'); f.text(.055,.82,sub,fontsize=12)
 a.set_facecolor('#eef6f7'); a.spines[['top','right']].set_visible(False); a.grid(alpha=.25)
 return f,a
def save(f,n):
 f.savefig(D/f'{n}.svg',metadata={'Date':None})
 b=io.BytesIO();f.savefig(b,format='png',dpi=144)
 with Image.open(b) as im: im.convert('RGB').save(D/f'{n}.webp',quality=86,method=6)
 plt.close(f)
if __name__=='__main__':
 d=case('duplicate'); ap,points=class_ap(d['truth'],d['predictions'],'gear',Fraction(1,2))
 f,a=chart('A duplicate lowers precision','Three ranked detections · two labeled objects · synthetic')
 a.plot([float(r) for r,p,m in points],[float(p) for r,p,m in points],'o-',label='Ranked detections',color='#a34826')
 a.plot([0,.5,.5,1],[1,1,2/3,2/3],'--',label='Precision envelope',color='#09645a')
 a.set(xlabel='Recall',ylabel='Precision',xlim=(-.03,1.04),ylim=(.4,1.1))
 a.legend(loc='lower right',frameon=False,fontsize=10)
 save(f,'map-duplicate-precision')
 r=report(case('localization')); f,a=chart('One box, ten overlap thresholds','Synthetic box has exact IoU 0.70 with its reference')
 d=r['ap_by_class_and_iou']; a.bar(list(d),[x['gear'] for x in d.values()],color='#09645a')
 a.set(xlabel='Required IoU',ylabel='Average precision',ylim=(0,1.12))
 save(f,'map-localization-thresholds')
