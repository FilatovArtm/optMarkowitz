'''
Module provides the function to solve the composite constrained optimization problem 
by accelerated projected gradient descent.
'''

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
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            if len(x_k) <= 2:
                history['x'].append(x_k)

        v_k = x_k + (iter_number - 2) / (iter_number + 1) * (x_k - x_prev)
        x_k, x_prev = oracle.proximal_mapping(v_k - 1 / oracle.lipshitz_constant_ * oracle.grad(v_k)), x_k

    return x_k, message, history
