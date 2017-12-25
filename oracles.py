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
    def __init__(self, Sigma, income, lambda_1, rho, mode='cvxpy'):
        self.Sigma_ = np.copy(Sigma)
        self.income_ = np.copy(income)
        self.lambd_ = np.copy(lambda_1)
        self.rho_ = np.copy(rho)
        self.lipshitz_constant_ = 2 * np.max(np.abs(np.linalg.eigvals(self.Sigma_)))
        self.mode_ = mode

    def func(self, x):
        return x.T @ self.Sigma_ @ x - \
               self.rho_ * self.income_ @ x + \
               self.lambd_ *  np.linalg.norm(x, 1)

    def grad(self, x):
        return 2 * self.Sigma_ @ x - self.rho_ * self.income_

    def proximal_mapping(self, x):
        if self.mode_ == 'cvxpy':
            y = cvxpy.Variable(len(x))
            obj = cvxpy.Minimize(self.lambd_ * cvxpy.norm1(y) +
                                self.lipshitz_constant_ / 2 * cvxpy.norm(y - x) ** 2)
            constraints = [cvxpy.sum_entries(y) == 1]
            problem = cvxpy.Problem(obj, constraints)
            problem.solve()
            return np.array(y.value).reshape(-1)
        else:
            return self.efficient_proximal_operator(x)

    def efficient_proximal_operator(self, x):
        c_upper = (-self.lambd_ + self.lipshitz_constant_ * x) / self.lipshitz_constant_
        c_lower = (-self.lambd_ - self.lipshitz_constant_ * x) / self.lipshitz_constant_

        maxb = np.max([c_upper, c_lower])
        minb = np.min([c_upper, c_lower])

        while np.abs(maxb - minb) > 1e-10:
            gamma = (maxb + minb) / 2
            tmp_1 = c_upper - gamma / self.lipshitz_constant_
            tmp_2 = c_lower + gamma / self.lipshitz_constant_
            first_part = np.maximum(tmp_1, 0, tmp_1)
            second_part = np.maximum(tmp_2, 0, tmp_2)
            result = np.sum(first_part - second_part) - 1

            if result < 0:
                maxb = gamma
            elif result > 0:
                minb = gamma
            else:
                break

        tmp_1 = c_upper - gamma / self.lipshitz_constant_
        tmp_2 = c_lower + gamma / self.lipshitz_constant_
        first_part = np.maximum(tmp_1, 0, tmp_1)
        second_part = np.maximum(tmp_2, 0, tmp_2)
    
        return first_part - second_part

