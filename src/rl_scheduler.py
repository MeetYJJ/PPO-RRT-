"""RL-ready resource scheduler. PPO controls sampling allocation; RRT* remains unchanged."""
import math

def observation(rrt, env, pos_sigma=0.05, rot_sigma=0.2):
    near=[n.pose for n in rrt.nodes if math.sqrt(sum((n.pose[0][i]-env.goal[0][i])**2 for i in range(3)))<.25]
    pos_gap=1.0 if not near else min(1.0,min(math.sqrt(sum((p[0][i]-env.goal[0][i])**2 for i in range(3))) for p in near)/.25)
    rot_gap=1.0 if not near else min(1.0,min(2*math.acos(min(1,abs(sum(p[1][i]*env.goal[1][i] for i in range(4))))) for p in near)/math.pi)
    hist=[x for x in rrt.history[-20:] if math.isfinite(x)]
    improve=(hist[-2]-hist[-1]) if len(hist)>1 else 0.0
    valid=rrt.extensions/max(1,len(rrt.history))
    return [pos_gap,rot_gap,min(1,len(rrt.nodes)/500),valid,min(1,len(rrt.history)/rrt.max_iter),1.0 if rrt.best else 0.0,min(1,abs(improve)),min(1,pos_sigma/.2),min(1,rot_sigma/.8)]

def action_to_probabilities(action):
    """Map 6 bounded policy outputs to valid resource allocation and step scale."""
    a=[max(-1,min(1,float(x))) for x in action]
    e=[math.exp(x) for x in a[:4]]; s=sum(e); p=[max(.05,x/s) for x in e]; z=sum(p); p=[x/z for x in p]
    return {'global':p[0],'goal_position':p[1],'local':p[2],'goal_orientation':p[3],'step_scale':0.5+0.5*(a[4]+1)/2,'uncertainty_scale':0.25+0.75*(a[5]+1)/2}

def reward_delta(prev, current, first_solution=False, collision=False, budget_exhausted=False):
    r=0.0
    if first_solution:r+=5.0
    if prev is not None and current is not None and math.isfinite(prev) and math.isfinite(current):r+=2.0*max(-1.0,min(1.0,prev-current))
    if collision:r-=0.05
    if budget_exhausted:r-=1.0
    return r
