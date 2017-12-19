import numpy as np
import cvxpy
from collections import defaultdict
from datetime import datetime


def FISTA(oracle, x_0, max_iter=100, trace=False):
    y_k = np.copy(x_0)
    x_k = np.copy(x_0)
    t_k = 1

    history = defaultdict(list) if trace else None
    message = 'success'
    start_time = datetime.now()

    for i in range(max_iter):
        if history != None:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            if len(x_k) <= 2:
                history['x'].append(x_k)


        x_next = oracle.proximal_mapping(
            y_k - 1 / oracle.lipshitz_constant_ * oracle.grad(y_k))
        t_next = (1 + np.sqrt(1 + 4 * t_k ** 2)) / 2
        y_k = x_next + (t_k - 1) / t_next * (x_next - x_k)
        x_k = x_next
        t_k = t_next

    return x_k, message, history
