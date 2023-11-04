# artificial-rabbits-optimization

This repository contains the code for the paper ["Artificial Rabbits Optimization: A New Metaheuristic Algorithm for Numerical Optimization"](https://doi.org/10.1016/j.engappai.2022.105082).

## Requirements

Install the requirements using the following command:

```bash
pip install -r requirements.txt
```

## Run

```bash
python index.py
```

## Problem Definition

Derive Problem class and implement the fit_func method. you can see the example in the [index.py]

## Parameters

| Parameter  | Description                            |
| ---------- | -------------------------------------- |
| `pop_size` | Number of rabbits                      |
| `epoch`    | Maximum number of iterations           |
| `lb`       | Lower bound of the problem             |
| `ub`       | Upper bound of the problem             |
| `seed`     | Random number seed for reproducibility |

## Usage

```python
from aro import ARO
from problem import Problem
import math


class DerivedProblem(Problem):
    def fit_func(self, solution):
        x1 = solution[0]
        x2 = solution[1]
        return 21.5 + x1 * math.sin(4.0 * 3.14 * x1) + x2 * math.sin(20.0 * 3.14 * x2)

problem_dict = {
    "lb": [-3.0, 4.1],
    "ub": [12.1, 5.8],
}
prob = DerivedProblem(**problem_dict)

aro = ARO(epoch=1000, pop_size=50, problem=prob, seed=27)
best, worst = aro.solve()
print("Best solution: {}".format(best))
print("Worst solution: {}".format(worst))

```
