import numpy as np


class Problem:

    def __init__(self, lb=None, ub=None):
        self.n_dims = None
        self.lb = np.array(lb).flatten()
        self.ub = np.array(ub).flatten()
        if len(self.lb) == len(self.ub):
            self.n_dims = len(self.lb)
            if len(self.lb) < 1:
                raise ValueError(
                    f"Length of lb and ub must be greater than 0. {len(self.lb)} != {len(self.ub)}.")
        else:
            raise ValueError(
                f"Length of lb and ub must be equal. {len(self.lb)} != {len(self.ub)}.")

    def fit_func(self, x):
        raise NotImplementedError
