import sys,os,math
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from src.environment import Env
from src.collision_checker import CollisionChecker
from src.rrt_star import RRTStar
from src.sampling_policies import make_sampler
from src.candidate_sampling import uncertainty_sampler
from src.metrics import evaluate,write_csv
rows=[]
# smoke run: 5 paired layouts per difficulty and uncertainty level
for d in ['simple','medium','hard']:
 for seed in range(5):
  for level in [0.02,0.05,0.10,0.20]:
   for alg in ['no_uncertainty','fixed_uncertainty','feedback_uncertainty']:
    e=Env.demo(d,seed); c=CollisionChecker(e); sm=None
    if alg=='fixed_uncertainty': sm=uncertainty_sampler(e,level,False)
    if alg=='feedback_uncertainty': sm=uncertainty_sampler(e,level,True)
    p=RRTStar(e,c,seed,max_iter=120,sampler=sm); path=p.run(); r=evaluate(path,e,p); sc=r.pop('sample_counts',{}); r.update({'difficulty':d,'seed':seed,'uncertainty_level':level,'algorithm':alg,'prediction_error_model':'synthetic_gaussian','samples_goal':sc.get('goal',0),'samples_global':sc.get('global',0),'samples_local':sc.get('local',0)}); rows.append(r)
write_csv(rows,'results/raw/uncertainty_ablation_smoke.csv'); print('rows',len(rows),'written')
