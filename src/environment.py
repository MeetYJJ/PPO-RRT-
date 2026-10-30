from dataclasses import dataclass
from .geometry_se3 import qnorm
@dataclass
class Sphere: c: tuple; r: float
@dataclass
class Env:
 bounds: tuple=((0,1),(0,1),(0,1))
 obstacles: tuple=()
 start: tuple=((.1,.1,.1),(1,0,0,0))
 goal: tuple=((.9,.9,.9),(1,0,0,0))
 pos_tol: float=.08
 rot_tol: float=.25
 body: tuple=(.08,.06,.04)
 @staticmethod
 def demo(difficulty='simple',seed=0):
  import random; R=random.Random(seed); n={'simple':2,'medium':5,'hard':9}.get(difficulty,2); obs=[]
  for _ in range(n): obs.append(Sphere((R.uniform(.2,.8),R.uniform(.2,.8),R.uniform(.2,.8)), .07 if difficulty=='simple' else .1))
  return Env(obstacles=tuple(obs), start=((.1,.1,.1),(1,0,0,0)), goal=((.9,.9,.9),(1,0,0,0)))
