import math
from .sampling_policies import make_sampler
from .geometry_se3 import qdist,random_quat

def coverage_gap_sampler(env,pos_sigma=.05,rot_sigma=.2):
    glob=make_sampler('global',env); calls=[0]; last=[float('inf')]; stall=[0]
    def f(rrt):
        calls[0]+=1; cur=rrt.best.cost if rrt.best else float('inf')
        if cur<last[0]-1e-7: stall[0]=0; last[0]=cur
        else: stall[0]+=1; last[0]=cur
        near=[n.pose for n in rrt.nodes if math.sqrt(sum((n.pose[0][i]-env.goal[0][i])**2 for i in range(3)))<.25]
        pos_gap=1.0 if not near else min(1.0,min(math.sqrt(sum((p[0][i]-env.goal[0][i])**2 for i in range(3))) for p in near)/.25)
        rot_gap=1.0 if not near else min(1.0,min(qdist(p[1],env.goal[1]) for p in near)/math.pi)
        # if tree lacks pose coverage, target sampling; if covered, local optimization
        explore=min(.7,.18+.35*min(1,stall[0]/25)+.20*pos_gap)
        if rrt.rng.random()<explore:return glob(rrt)
        if pos_gap>rot_gap:
            p=tuple(max(env.bounds[i][0],min(env.bounds[i][1],env.goal[0][i]+rrt.rng.gauss(0,pos_sigma))) for i in range(3)); q=env.goal[1]
        else:
            p=tuple(max(env.bounds[i][0],min(env.bounds[i][1],env.goal[0][i]+rrt.rng.gauss(0,pos_sigma*.5))) for i in range(3)); q=random_quat(rrt.rng)
        rrt.sample_counts['coverage_target']=rrt.sample_counts.get('coverage_target',0)+1
        return p,q
    return f
