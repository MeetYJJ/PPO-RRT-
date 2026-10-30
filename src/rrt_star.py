import random, math
from dataclasses import dataclass
from .geometry_se3 import *
@dataclass
class Node: pose: tuple; parent: int; cost: float
class RRTStar:
 def __init__(self,env,checker,seed=0,max_iter=500,step=.15,goal_bias=.05,sampler=None,iteration_callback=None): self.env,self.chk,self.rng,self.max_iter,self.step,self.goal_bias,self.sampler=env,checker,random.Random(seed),max_iter,step,goal_bias,sampler; self.iteration_callback=iteration_callback; self.nodes=[Node(env.start,-1,0)]; self.best=None; self.history=[]; self.extensions=0; self.first_solution_iter=None; self.sample_counts={}
 def sample(self):
  if self.rng.random()<self.goal_bias:self.sample_counts['goal_bias']=self.sample_counts.get('goal_bias',0)+1; return self.env.goal
  if self.sampler:
   s=self.sampler(self); return s
  self.sample_counts['global']=self.sample_counts.get('global',0)+1
  b=self.env.bounds; return ((self.rng.uniform(*b[0]),self.rng.uniform(*b[1]),self.rng.uniform(*b[2])),random_quat(self.rng))
 def run(self):
  for it in range(self.max_iter):
   s=self.sample(); ni=min(range(len(self.nodes)),key=lambda i:pose_dist(self.nodes[i].pose,s)); np=steer(self.nodes[ni].pose,s,self.step)
   if not self.chk.edge(self.nodes[ni].pose,np):
    self.history.append(self.best.cost if self.best else float('inf'))
    if self.iteration_callback is not None and self.iteration_callback(self,it): break
    continue
   self.extensions+=1; rad=max(self.step*2, .25*math.sqrt(math.log(len(self.nodes)+1)/(len(self.nodes)+1))); neigh=[i for i,n in enumerate(self.nodes) if pose_dist(n.pose,np)<=rad]
   parent=ni; cost=self.nodes[ni].cost+pose_dist(self.nodes[ni].pose,np)
   for i in neigh:
    c=self.nodes[i].cost+pose_dist(self.nodes[i].pose,np)
    if c<cost and self.chk.edge(self.nodes[i].pose,np): parent,cost=i,c
   self.nodes.append(Node(np,parent,cost)); idx=len(self.nodes)-1
   for i in neigh:
    if self.nodes[idx].cost+pose_dist(np,self.nodes[i].pose)<self.nodes[i].cost and self.chk.edge(np,self.nodes[i].pose): self.nodes[i].parent=idx; self.nodes[i].cost=self.nodes[idx].cost+pose_dist(np,self.nodes[i].pose)
   pos_err=math.sqrt(sum((np[0][j]-self.env.goal[0][j])**2 for j in range(3))); rot_err=qdist(np[1],self.env.goal[1])
   if pos_err<=self.env.pos_tol and rot_err<=self.env.rot_tol and self.chk.edge(np,self.env.goal):
    gc=cost+pose_dist(np,self.env.goal)
    if self.best is None or gc<self.best.cost:
     self.nodes.append(Node(self.env.goal,idx,gc)); self.best=self.nodes[-1]
     if self.first_solution_iter is None:self.first_solution_iter=it+1
   self.history.append(self.best.cost if self.best else float('inf'))
   if self.iteration_callback is not None and self.iteration_callback(self,it): break
  return self.path()
 def path(self):
  if not self.best:return None
  out=[]; i=len(self.nodes)-1
  while i>=0: out.append(self.nodes[i].pose); i=self.nodes[i].parent
  return list(reversed(out))
