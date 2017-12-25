from collections import defaultdict
from datetime import datetime
import numpy as np


def AcceleratedProximalGD(oracle, x_0, max_iter=100, tolerance=1e-5, trace=False):
    '''
    Function to perform accelerated proximal gradient descent.
    Args:
        oracle - class with methods: func (value of the function),
                 grad (gradient of the smooth part)
                 and projection (projection onto feasible set)
        x_0 - starting point
        max_iter - maximal number of iteration
        tolerance - stopping criteria
        trace - indicates the storing of the history
    Return:
        x_k - achieved vector
        message - status of the optimization
        history - collected history (dictionary with keys 'time', 'func' and 'x')
    '''
    x_k = np.copy(x_0)
    x_prev = np.copy(x_k)
    history = defaultdict(list) if trace else None
    message = 'success'
    start_time = datetime.now()

    iter_number = 0
    for iter_number in range(max_iter):
        if history != None:
            history['time'].append(
                (datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            if len(x_k) <= 2:
                history['x'].append(x_k)

        v_k = x_k + (iter_number - 2) / (iter_number + 1) * (x_k - x_prev)
        x_k, x_prev = oracle.proximal_mapping(
            v_k - 1 / oracle.lipshitz_constant_ * oracle.grad(v_k)), x_k
        if np.linalg.norm(x_k - x_prev) ** 2 < tolerance:
            break

    return x_k, message, history


def FISTA(oracle, x_0, max_iter=100, trace=False, tolerance=1e-5):
    y_k = np.copy(x_0)
    x_k = np.copy(x_0)
    t_k = 1

    history = defaultdict(list) if trace else None
    message = 'success'
    start_time = datetime.now()

    for i in range(max_iter):
        if history != None:
            history['time'].append(
                (datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            if len(x_k) <= 2:
                history['x'].append(x_k)

        x_next = oracle.proximal_mapping(
            y_k - 1 / oracle.lipshitz_constant_ * oracle.grad(y_k))
        t_next = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        y_k = x_next + (t_k - 1) / t_next * (x_next - x_k)
        if np.linalg.norm(x_k - x_next) ** 2 < tolerance:
            x_k = x_next
            break
        x_k = x_next
        t_k = t_next

    return x_k, message, history


def ADMM(oracle, x_0, eta, max_iter=100, tolerance=1e-8, trace=False):
    n = x_0.shape[0]
    x_k = np.copy(x_0)
    v_k = np.ones(n)
    alpha_k = np.ones(n)
    beta_k = np.random.uniform(0, 1)

    precomp_inv = np.linalg.inv(
        2 * oracle.Sigma_ + eta * (np.eye(n) + np.ones((n, n))))

    history = defaultdict(list) if trace else None
    message = 'success'
    start_time = datetime.now()

    for k in np.arange(max_iter):
        if history != None:
            history['time'].append(
                (datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            if len(x_k) <= 2:
                history['x'].append(x_k)

        x_k = precomp_inv\
            .dot(oracle.rho_ * oracle.income_ - alpha_k - beta_k * np.ones(n) + eta * (v_k + np.ones(n)))

        v_k = (np.absolute(x_k + alpha_k / eta) - oracle.lambd_ /
               eta).clip(0) * np.sign(x_k + alpha_k / eta)
        alpha_k += eta * (x_k - v_k)
        beta_k += eta * (np.ones(n).dot(x_k) - 1)
        '''
        dual_gap = -(alpha_k + beta_k*np.ones(n)).dot(x_k) + beta_k + oracle.lambd_*np.linalg.norm(v_k, 1)
        if history is not None: history['dual'].append(dual_gap)
            
        if np.abs(dual_gap) < tolerance:
            break
        '''
    return x_k, message, history
