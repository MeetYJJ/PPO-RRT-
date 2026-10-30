from .sampling_policies import make_sampler

def uncertainty_sampler(env, level=0.05, feedback=False):
    glob=make_sampler('global',env); goal=make_sampler('goal',env,level); local=make_sampler('local',env)
    calls=[0]; stall=[0]; last=[float('inf')]
    def f(rrt):
        calls[0]+=1; cur=rrt.best.cost if rrt.best else float('inf')
        if cur < last[0]-1e-7: stall[0]=0; last[0]=cur
        else: stall[0]+=1
        # covariance changes target spread; feedback changes exploration/exploitation
        p_unc=min(.65,.25+level*3); p_global=.35 if not feedback else min(.7,.25+stall[0]/50)
        if rrt.rng.random()<p_global:return glob(rrt)
        if rrt.rng.random()<p_unc:return goal(rrt)
        return local(rrt)
    return f
