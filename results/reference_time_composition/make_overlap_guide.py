"""Draw the exact three-design source-time example from fixed coefficients."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make(out):
 out.mkdir(parents=True,exist_ok=False)
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
 c=[F(1),F(0),F(0),F(0),F(-1),F(0)]
 ds=[[F(0),-F(1,2),F(1),F(0),-F(1,2),F(0)],[-F(1,2),F(0),F(1),F(0),F(0),-F(1,2)],[-F(1,2),F(0),F(1),F(0),-F(1,2),F(0)]]
 names=['Positive overlap','Negative overlap','Zero overlap']
 fig,axes=plt.subplots(1,3,figsize=(10.8,4.1));records=[]
 for ax,name,d in zip(axes,names,ds):
  ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off');ax.text(.5,.97,name,ha='center',va='top',weight='bold')
  xs=[.08,.28,.5,.7,.9]
  for x,label in zip(xs,['Source','Peer','c','d','c × d']):ax.text(x,.84,label,ha='center')
  ax.plot([.01,.99],[.80,.80],color='.25',lw=.7)
  for k,(a,b) in enumerate(zip(c,d)):
   y=.74-k*.081;v=a*b
   values=[str(k//2+1),str(k%2+1),str(a),str(b),str(v)]
   for z,(x,value)in enumerate(zip(xs,values)):
    color=('#176b91' if v>0 else '#9b4d16') if z==4 and v else '.15'
    ax.text(x,y,value,ha='center',color=color,weight='bold' if z==4 and v else 'normal')
  overlap=sum((a*b for a,b in zip(c,d)),F());mean=F(5,54)*overlap
  ax.plot([.01,.99],[.27,.27],color='.25',lw=.7)
  ax.text(.5,.20,'Total overlap: '+str(overlap),ha='center',weight='bold')
  ax.text(.5,.11,'Additional mean: '+str(mean),ha='center')
  records.append({'design':name,'A':list(map(str,c)),'B':list(map(str,d)),'overlap':str(overlap),'noise_variance':'5/54','population_reference_mean':'0','additional_mean':str(mean)})
 fig.suptitle('Different rows can reuse the same raw observation',y=.99,fontsize=13)
 fig.text(.5,.028,'Fixed comparison coefficients; independent uniform {−2, 0, 5} source values. Each population-reference product has mean zero.',ha='center',fontsize=8.5)
 fig.subplots_adjust(left=.035,right=.985,bottom=.12,top=.88,wspace=.16)
 for ext in ['pdf','svg','png']:
  kw={'dpi':180}if ext=='png'else{}
  if ext=='pdf':kw['metadata']={'CreationDate':None,'ModDate':None}
  if ext=='svg':kw['metadata']={'Date':None}
  fig.savefig(out/('source_time_overlap.'+ext),**kw)
 plt.close(fig)
 (out/'source_time_overlap.json').write_text(json.dumps({'scope':'Exact fixed-design coefficient illustration, not a real-panel calibration result','designs':records},indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);make(p.parse_args().output)
