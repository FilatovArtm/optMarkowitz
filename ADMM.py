import numpy as np
import cvxpy

from datetime import datetime
from collections import defaultdict

def ADMM(oracle, x_0, eta, max_iter=100, tolerance=1e-5, trace = False):
    n = x_0.shape[0]
    x_k = np.copy(x_0)
    v_k = np.zeros(n)
    alpha_k = np.zeros(n)
    beta_k = np.random.uniform(0, 1)
    
    precomp_inv = np.linalg.inv(2*oracle.Sigma_ + eta*(np.eye(n) + np.ones((n,n))))
    
    history = defaultdict(list) if trace else None
    message = 'success'
    start_time = datetime.now()
    
    for k in np.arange(max_iter):
        if history != None:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            if len(x_k) <= 2:
                history['x'].append(x_k)
        
        x_k = precomp_inv\
        .dot(oracle.income_ - alpha_k - beta_k*np.ones(n) + eta*(v_k + np.ones(n)))
        
        v_k = (np.absolute(x_k + alpha_k/eta) - oracle.lambd_/eta).clip(0) * np.sign(x_k + alpha_k/eta)
        alpha_k += eta*(x_k - v_k)
        beta_k += eta*(np.ones(n).dot(x_k) - 1)
        
    return x_k, message, history