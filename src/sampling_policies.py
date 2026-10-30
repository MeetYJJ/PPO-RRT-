from .geometry_se3 import random_quat

def make_sampler(mode,env,unc=0.05):
 def f(rrt):
  rrt.sample_counts[mode]=rrt.sample_counts.get(mode,0)+1
  rng=rrt.rng; b=env.bounds
  if mode=='goal':
   p=tuple(max(b[i][0],min(b[i][1],env.goal[0][i]+rng.gauss(0,.12))) for i in range(3)); return (p,env.goal[1])
  if mode=='local':
   n=rng.choice(rrt.nodes).pose; p=tuple(max(b[i][0],min(b[i][1],n[0][i]+rng.gauss(0,.15))) for i in range(3)); return (p,n[1])
  if mode=='uncertainty':
   p=tuple(max(b[i][0],min(b[i][1],env.goal[0][i]+rng.gauss(0,unc))) for i in range(3)); return (p,env.goal[1])
  return ((rng.uniform(*b[0]),rng.uniform(*b[1]),rng.uniform(*b[2])),random_quat(rng))
 return f

def make_faroni_inspired_sampler(env, local_radius=.12):
 """Literature-inspired hybrid: informed global samples after a solution, otherwise global; reward is recent cost improvement."""
 def f(rrt):
  import math
  rrt.sample_counts['faroni_global']=rrt.sample_counts.get('faroni_global',0)
  rrt.sample_counts['faroni_local']=rrt.sample_counts.get('faroni_local',0)
  if rrt.best is None:
   rrt.sample_counts['faroni_global']+=1
   return make_sampler('global',env)(rrt)
  # adaptive exploitation probability rises when a solution recently improved
  hist=[x for x in rrt.history[-20:] if math.isfinite(x)]
  improved=len(hist)>1 and hist[-1] < min(hist[:-1])-1e-9
  p_local=.65 if improved else .35
  if rrt.rng.random()<p_local:
   rrt.sample_counts['faroni_local']+=1
   n=rrt.rng.choice(rrt.nodes).pose; b=env.bounds
   p=tuple(max(b[i][0],min(b[i][1],n[0][i]+rrt.rng.gauss(0,local_radius))) for i in range(3))
   return (p,n[1])
  rrt.sample_counts['faroni_global']+=1
  return make_sampler('global',env)(rrt)
 return f
