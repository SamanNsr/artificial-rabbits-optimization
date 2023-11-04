from aro import ARO
from problem import Problem
import math


class DerivedProblem(Problem):
    def fit_func(self, solution):
        x1 = solution[0]
        x2 = solution[1]
        return 21.5 + x1 * math.sin(4.0 * 3.14 * x1) + x2 * math.sin(20.0 * 3.14 * x2)


def __main__():
    problem_dict = {
        "lb": [-3.0, 4.1],
        "ub": [12.1, 5.8],
    }
    prob = DerivedProblem(**problem_dict)

    aro = ARO(epoch=1000, pop_size=50, problem=prob, seed=27)
    best, worst = aro.solve()
    print("Best solution: {}".format(best))
    print("Worst solution: {}".format(worst))


__main__()
