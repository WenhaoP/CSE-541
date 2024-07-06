import numpy as np
from math import sqrt

class BernoulliBandit:
    # accepts a list of K >= 2 floats, each lying in [0,1]
    def __init__ (self, means, seed=541):
        self.num_of_arms = len(means)
        assert self.num_of_arms >= 2, "Number of arms must be >= 2!"

        self.arm_means = means
        self.best_arm_idx = np.argmax(means)
        self.best_arm_mean = np.max(means)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pulled_arm_history = []
        self.pulled_arm_mean_history = []
        self.reward_history = []

    # Function should return the number of arms
    def K(self):
        return self.num_of_arms

    # Accepts a parameter 0 <= a <= K-1 and returns the
    # realisation of random variable X with P(X = 1) being
    # the mean of the (a+1) th arm .
    def pull(self, a):
        pulled_arm_mean = self.arm_means[a]
        reward = self.rng.choice([0, 1], p=[1 - pulled_arm_mean, pulled_arm_mean])

        self.pulled_arm_history.append(a)
        self.pulled_arm_mean_history.append(pulled_arm_mean)
        self.reward_history.append(reward)
        return reward

    # Returns the regret incurred so far.
    def regret(self):
        random_regret = len(self.reward_history) * self.best_arm_mean - np.sum(self.reward_history)
        pseudo_regret = len(self.reward_history) * self.best_arm_mean - np.sum(self.pulled_arm_mean_history)
        return {"random": random_regret, "pseudo": pseudo_regret}
    

class GaussianBandit:

    # accepts two lists of K >= 2 floats that are means and variances of K arms
    def __init__ (self, means, vars, seed=541):
        self.num_of_arms = len(means)
        assert self.num_of_arms >= 2, "Number of arms must be >= 2!"

        self.arm_means = means
        self.arm_vars = vars
        self.best_arm_idx = np.argmax(means)
        self.best_arm_mean = np.max(means)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pulled_arm_history = []
        self.pulled_arm_mean_history = []
        self.reward_history = []

    # Function should return the number of arms
    def K(self):
        return self.num_of_arms

    # Accepts a parameter 0 <= a <= K-1 and returns the
    # realisation of random variable X followed a Gaussian distribution.
    def pull(self, a):
        pulled_arm_mean = self.arm_means[a]
        pulled_arm_var = self.arm_vars[a]
        reward = self.rng.normal(pulled_arm_mean, sqrt(pulled_arm_var))

        self.pulled_arm_history.append(a)
        self.pulled_arm_mean_history.append(pulled_arm_mean)
        self.reward_history.append(reward)
        return reward

    # Returns the regret incurred so far.
    def regret(self):
        random_regret = len(self.reward_history) * self.best_arm_mean - np.sum(self.reward_history)
        pseudo_regret = len(self.reward_history) * self.best_arm_mean - np.sum(self.pulled_arm_mean_history)
        return {"random": random_regret, "pseudo": pseudo_regret}
