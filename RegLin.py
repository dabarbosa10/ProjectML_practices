import numpy as np
import pandas as pd

class Linreg:

    def __init__(self, n_iter: int = 1000, eta: float =0.05) -> object:
        """_summary_

        Args:
            n_iter (int, optional): _description_. Defaults to 1000.
            eta (float, optional): _description_. Defaults to 0.05.

        Returns:
            object: _description_
        """        

        self.theta = None
        self.cost = None
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x: pd.DataFrame, y: list[float])-> object:
        """_summary_

        Args:
            x (pd.DataFrame): _description_
            y (list[float]): _description_

        Returns:
            object: _description_
        """        
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
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """        
        return np.dot(x. self.theta)






