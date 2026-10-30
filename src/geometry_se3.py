import math, random

def qnorm(q):
 n=math.sqrt(sum(x*x for x in q)); return tuple(x/n for x in q) if n>1e-12 else (1,0,0,0)
def qdot(a,b): return sum(x*y for x,y in zip(a,b))
def qdist(a,b): return 2*math.acos(min(1.0,abs(qdot(qnorm(a),qnorm(b)))))
def slerp(a,b,t):
 a=qnorm(a); b=qnorm(b); d=qdot(a,b)
 if d<0: b=tuple(-x for x in b); d=-d
 if d>0.9995: return qnorm(tuple((1-t)*x+t*y for x,y in zip(a,b)))
 th=math.acos(max(-1,min(1,d))); s=math.sin(th)
 return tuple((math.sin((1-t)*th)/s)*x+(math.sin(t*th)/s)*y for x,y in zip(a,b))
def pose_dist(a,b,lR=0.5):
 dp=math.sqrt(sum((a[0][i]-b[0][i])**2 for i in range(3))); return math.sqrt(dp*dp+(lR*qdist(a[1],b[1]))**2)
def interpolate(a,b,t): return (tuple(a[0][i]+t*(b[0][i]-a[0][i]) for i in range(3)),slerp(a[1],b[1],t))
def steer(a,b,step,lR=0.5):
 d=pose_dist(a,b,lR); return b if d<=step else interpolate(a,b,step/d)
def path_metrics(path,lR=0.5):
 if len(path)<2:return (0,0,0)
 trans=sum(math.sqrt(sum((path[i][0][j]-path[i-1][0][j])**2 for j in range(3))) for i in range(1,len(path)))
 rot=sum(qdist(path[i-1][1],path[i][1]) for i in range(1,len(path))); return trans,rot,math.sqrt(trans*trans+(lR*rot)**2)
def random_quat(rng):
 u1,u2,u3=rng.random(),rng.random(),rng.random(); return qnorm((math.sqrt(1-u1)*math.sin(2*math.pi*u2),math.sqrt(1-u1)*math.cos(2*math.pi*u2),math.sqrt(u1)*math.sin(2*math.pi*u3),math.sqrt(u1)*math.cos(2*math.pi*u3)))
