import math
from .sampling_policies import make_sampler
from .geometry_se3 import slerp,random_quat

def coupled_sampler(env, pos_sigma=0.05, rot_sigma=0.10):
    glob=make_sampler('global',env); local=make_sampler('local',env)
    calls=[0]; stall=[0]; last=[float('inf')]
    def sample(rrt):
        calls[0]+=1; cur=rrt.best.cost if rrt.best else float('inf')
        if cur < last[0]-1e-7: stall[0]=0; last[0]=cur
        else: stall[0]+=1
        stagnation=min(1.0,stall[0]/25.0)
        # uncertainty and stagnation interact: high rotation uncertainty makes local exploitation less useful
        rot_factor=min(1.0,rot_sigma/0.4); pos_factor=min(1.0,pos_sigma/0.2)
        p_global=min(.75,max(.15,.20 + .40*stagnation + .15*rot_factor))
        if rrt.best is None: p_global=min(.55,p_global)
        u=rrt.rng.random(); b=env.bounds
        if u<p_global: return glob(rrt)
        if rrt.best is not None and u<p_global+.25*(1-rot_factor): return local(rrt)
        # coupled terminal sample: position and orientation are perturbed together
        p=tuple(max(b[i][0],min(b[i][1],env.goal[0][i]+rrt.rng.gauss(0,pos_sigma*(1+0.5*stagnation)))) for i in range(3))
        qrand=random_quat(rrt.rng); alpha=min(1.0,abs(rrt.rng.gauss(0,rot_sigma))/math.pi)
        q=slerp(env.goal[1],qrand,alpha)
        rrt.sample_counts['coupled_uncertainty']=rrt.sample_counts.get('coupled_uncertainty',0)+1
        return (p,q)
    return sample
