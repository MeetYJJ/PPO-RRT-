import math
from .geometry_se3 import interpolate
class CollisionChecker:
 def __init__(self,env,resolution=.04): self.env=env; self.res=resolution; self.calls=0
 def point(self,p):
  (x,y,z)=p
  if any(x<a or x>b for (x,y,z), (a,b) in zip([p]*3,self.env.bounds)): return True
  # conservative box radius
  br=max(self.env.body)
  return any(sum((p[i]-o.c[i])**2 for i in range(3)) <= (o.r+br)**2 for o in self.env.obstacles)
 def edge(self,a,b):
  self.calls+=1; n=max(2,int(math.ceil(self._dist(a,b)/self.res)))
  return all(not self.point(interpolate(a,b,i/n)[0]) for i in range(n+1))
 def _dist(self,a,b): return math.sqrt(sum((a[0][i]-b[0][i])**2 for i in range(3)))
