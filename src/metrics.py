import math, csv, os, statistics
from .geometry_se3 import path_metrics,pose_dist,qdist
def evaluate(path,env,planner):
 if not path:return {'success':0,'failure_reason':'no_terminal_connection','trans_length':float('nan'),'rot_change':float('nan'),'cost':float('nan'),'goal_pos_err':float('nan'),'goal_rot_err':float('nan'),'extensions':planner.extensions,'collision_checks':planner.chk.calls,'first_solution_iter':float('nan'),'min_clearance':float('nan'),'sample_counts':dict(planner.sample_counts)}
 t,r,c=path_metrics(path); clear=min((math.sqrt(sum((p[0][i]-o.c[i])**2 for i in range(3)))-o.r-max(env.body) for p in path for o in env.obstacles),default=float('inf')); return {'success':1,'failure_reason':'','trans_length':t,'rot_change':r,'cost':c,'goal_pos_err':math.sqrt(sum((path[-1][0][i]-env.goal[0][i])**2 for i in range(3))),'goal_rot_err':qdist(path[-1][1],env.goal[1]),'extensions':planner.extensions,'collision_checks':planner.chk.calls,'first_solution_iter':planner.first_solution_iter,'min_clearance':clear,'sample_counts':dict(planner.sample_counts)}
def write_csv(rows,path):
 os.makedirs(os.path.dirname(path),exist_ok=True); keys=sorted({k for r in rows for k in r});
 with open(path,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def summary(rows):
 out={};
 for k in ['success','trans_length','rot_change','cost','extensions','collision_checks']:
  x=[r[k] for r in rows if isinstance(r.get(k),(int,float)) and math.isfinite(r[k])]; out[k+'_mean']=statistics.mean(x) if x else float('nan'); out[k+'_n']=len(x)
 return out
