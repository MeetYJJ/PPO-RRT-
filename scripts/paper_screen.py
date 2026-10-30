import sys,os,statistics
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler
from src.coupled_sampling import coupled_sampler
from src.coverage_sampling import coverage_gap_sampler
from src.metrics import evaluate,write_csv
rows=[]
for d in ['simple','medium','hard']:
 for seed in range(10):
  for a in ['rrtstar','fixed_unc','coupled','coverage_gap']:
   e=Env.demo(d,seed); c=CollisionChecker(e); sm=None
   if a=='fixed_unc':sm=make_sampler('goal',e,.05)
   elif a=='coupled':sm=coupled_sampler(e,.05,.20)
   elif a=='coverage_gap':sm=coverage_gap_sampler(e,.05,.20)
   p=RRTStar(e,c,seed,max_iter=160,sampler=sm); path=p.run(); r=evaluate(path,e,p); sc=r.pop('sample_counts',{}); r.update({'difficulty':d,'seed':seed,'algorithm':a,'samples_target':sum(v for k,v in sc.items() if 'target' in k or 'goal' in k)}); rows.append(r)
write_csv(rows,'results/raw/paper_screen.csv')
for d in ['simple','medium','hard','all']:
 print('\n',d)
 for a in ['rrtstar','fixed_unc','coupled','coverage_gap']:
  x=[r for r in rows if r['algorithm']==a and (d=='all' or r['difficulty']==d)]; ok=[r['cost'] for r in x if r['success']]; print(a,'success',round(sum(r['success'] for r in x)/len(x),3),'cost',round(statistics.mean(ok),3) if ok else None,'ext',round(statistics.mean(r['extensions'] for r in x),1))
