import sys,os,statistics
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler
from src.coupled_sampling import coupled_sampler
from src.metrics import evaluate,write_csv
rows=[]
for d in ['simple','medium','hard']:
 for seed in range(10):
  for alg in ['fixed_unc','coupled']:
   e=Env.demo(d,seed); c=CollisionChecker(e)
   sm=make_sampler('goal',e,0.05) if alg=='fixed_unc' else coupled_sampler(e,0.05,0.20)
   p=RRTStar(e,c,seed,max_iter=160,sampler=sm); path=p.run(); r=evaluate(path,e,p); r.pop('sample_counts',None); r.update({'difficulty':d,'seed':seed,'algorithm':alg}); rows.append(r)
write_csv(rows,'results/raw/coupled_screen.csv')
for d in ['simple','medium','hard']:
 for a in ['fixed_unc','coupled']:
  x=[r for r in rows if r['difficulty']==d and r['algorithm']==a]; ok=[r['cost'] for r in x if r['success']]; print(d,a,'success',sum(r['success'] for r in x)/10,'cost',statistics.mean(ok) if ok else None,'ext',statistics.mean(r['extensions'] for r in x))
