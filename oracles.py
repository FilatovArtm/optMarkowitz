import numpy as np
import cvxpy

class Oracle:
    '''
        Oracle for next optimization problem
            x^T Sigma x - rho * income^T x + lambda || x ||_1 rightarrow min
            sum x_i  = 1

        func: value of the function
        grad: computes the gradient of the differantible part
        proximal mapping: computes the constrained proximal operator

    '''
    def __init__(self, Sigma, income, lambd, rho):
        self.Sigma_ = np.copy(Sigma)
        self.income_ = np.copy(income)
        self.lambd_ = np.copy(lambd)
        self.rho_ = np.copy(rho)
        self.lipshitz_constant_ = 2 * np.max(np.linalg.eigvals(Sigma))

    def func(self, x):
        return x.T @ self.Sigma_ @ x - \
               self.rho_ * self.income_ @ x + \
               self.lambd_ *  np.linalg.norm(x, 1)

    def grad(self, x):
        return 2 * self.Sigma_ @ x - self.rho_ * self.income_

    def proximal_mapping(self, x):
        y = cvxpy.Variable(len(x))
        obj = cvxpy.Minimize(self.lambd_ * cvxpy.norm1(y) +
                             self.lipshitz_constant_ / 2 * cvxpy.norm(y - x) ** 2)
        constraints = [cvxpy.sum_entries(y) == 1]
        problem = cvxpy.Problem(obj, constraints)
        problem.solve()
        return np.array(y.value).reshape(-1)
