import math, random

def project_covariance(cov, floor=1e-8):
    if len(cov)!=6 or any(len(r)!=6 for r in cov): raise ValueError('covariance must be 6x6')
    c=[[0.5*(cov[i][j]+cov[j][i]) for j in range(6)] for i in range(6)]
    # diagonal loading is a stable PSD approximation without scipy
    for i in range(6): c[i][i]=max(c[i][i],floor)
    return c

def synthetic_prediction(true_pose, level, seed=0):
    rng=random.Random(seed); sp=level; sr=max(level*2,1e-4)
    err=[rng.gauss(0,sp) for _ in range(3)]+[rng.gauss(0,sr) for _ in range(3)]
    pred=(tuple(true_pose[0][i]+err[i] for i in range(3)),true_pose[1])
    cov=project_covariance([[((sp if i<3 else sr)**2 if i==j else 0.0) for j in range(6)] for i in range(6)])
    return pred,tuple(err),cov
