import numpy as np


class Linreg:

    def __init__(self, n_iter=1000, eta=0.05):
        """

        :type n_iter: object
        """
        self.theta = None
        self.cost = None
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        self.cost = []
        m = x.shape[0]
        self.theta = np.zeros(x.shape[1])

        for i in range(self.n_iter):
            h = np.dot(x, self.theta)
            self.theta = self.theta - (self.eta/m)*np.dot(x.T, h-y)
            cost_f = (1/2*m)*np.sum((h-y) ** 2)
            self.cost = np.append(self.cost, cost_f)

        return self

    def predict(self, x):
        return np.dot(x. self.theta)






