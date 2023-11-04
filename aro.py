import numpy as np
from problem import Problem


class ARO():
    POSITION_INDEX = 0
    FITNESS_INDEX = 1

    def __init__(self, problem: Problem, epoch=10000, pop_size=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.pop, self.solution = None, None
        self.epoch = epoch
        self.pop_size = pop_size
        self.problem = problem
        self.create_solution = None
        self.pop = self.create_population(self.pop_size)

    def create_population(self, pop_size=None):
        if pop_size is None:
            pop_size = self.pop_size
        pop = []
        for _ in range(0, pop_size):
            pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])
        return pop

    def evolve(self, epoch):
        theta = 2 * (1 - (epoch+1)/self.epoch)
        for idx in range(0, self.pop_size):
            L = (np.exp(1) - np.exp((epoch / self.epoch)**2)) * \
                (np.sin(2*np.pi*np.random.rand()))
            R = np.zeros(self.problem.n_dims)
            random_index = np.random.choice(np.arange(0, self.problem.n_dims), int(
                np.ceil(np.random.rand()*self.problem.n_dims)), replace=False)
            R[random_index] = 1
            R = L * R        # Eq 2
            A = 2 * np.log(1.0 / np.random.rand()) * theta      # Eq. 15

            if A > 1:   # detour foraging strategy
                rand_agent_idx = np.random.randint(0, self.pop_size)
                pos_new = self.pop[rand_agent_idx][self.POSITION_INDEX] + R * (self.pop[idx][self.POSITION_INDEX] - self.pop[rand_agent_idx][self.POSITION_INDEX]) + np.round(
                    0.5 * (0.05 + np.random.rand())) * np.random.normal(0, 1)      # Eq. 1

            else:       # Random hiding stage
                g = np.zeros(self.problem.n_dims)
                random_index = np.random.choice(np.arange(0, self.problem.n_dims), int(
                    np.ceil(np.random.rand() * self.problem.n_dims)), replace=False)
                g[random_index] = 1        # Eq. 12

                H = np.random.normal(0, 1) * (epoch / self.epoch)       # Eq. 8

                b = self.pop[idx][self.POSITION_INDEX] + H * g * \
                    self.pop[idx][self.POSITION_INDEX]        # Eq. 13

                pos_new = self.pop[idx][self.POSITION_INDEX] + R * (
                    np.random.rand() * b - self.pop[idx][self.POSITION_INDEX])      # Eq. 11

            pos_new = np.clip(pos_new, self.problem.lb, self.problem.ub)
            fit = self.problem.fit_func(pos_new)
            self.pop[idx] = [pos_new, fit]
            print(f"Epoch: {epoch}, Agent: {idx}, Position: {pos_new}")

    def solve(self):
        best = []
        worst = []
        # TODO: its only support maximization problem, add minimization support
        for epoch in range(0, self.epoch):
            self.evolve(epoch)

            sorted_pop = sorted(
                self.pop, key=lambda temp: temp[self.FITNESS_INDEX])
            if len(worst) == 0 or worst[self.FITNESS_INDEX] > sorted_pop[0][self.FITNESS_INDEX]:
                worst = sorted_pop[0]
            if len(best) == 0 or best[self.FITNESS_INDEX] < sorted_pop[-1][self.FITNESS_INDEX]:
                best = sorted_pop[-1]
        return best, worst
