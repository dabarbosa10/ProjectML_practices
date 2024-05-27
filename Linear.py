import numpy as np


class LinearReg:
    def __init__(self, alpha, n_iter, theta0, theta1, target, m, x):
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta0 = theta0
        self.theta1 = theta1
        self.target = target
        self.m = m
        self.x = x

    def lin_reg(self):
        return self.theta0 + self.theta1 * self.x

    def cost_f(self):
        return (1 / 2 * self.m) * np.sum((self.lin_reg() - self.target) ** 2)

    def grad_desc(self):
        self.theta0 = self.theta0 - self.alpha * (1 / 2 * self.m) * np.sum(self.lin_reg() - self.target)
        self.theta1 = self.theta1 - self.alpha * (1 / 2 * self.m) * np.sum(self.lin_reg() - self.target) * self.x
        return self.theta0, self.theta1

            


# y=2+2*x
# x=1 -> y=4

Lin1 = LinearReg(1, 1, 2, 2, 3, 4, 1)
val = Lin1.lin_reg()
print(val)

