"""Empirical tests of the agents"""
from typing import List
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
from joblib import delayed, Parallel

class Agent:
    """Exploratory agent
    
    Properties:
     T - max time to run
     arms - vector of scipy distrobutions
     log_regret_every_n - None or int, when to save regret
    
    Attributes:
     T - int, max time
     t - time, starts at 0
     arms - vector of scipy distrobutions
     emp_means - vector of current empirical means for arms
     Tis - vector of pull counts for arms
     regret - float, current regret
     means - vector of true means
     true_best - index of true best arm
     delts - vector of mean differences from true best arm
    """
    def __init__(self, T: int, arms: List[object], log_regret_every_n: int = None):
        self.T = T
        self.arms = arms
        self.n = len(arms)
        self.emp_means = np.zeros((self.n,1))
        self.Tis = np.zeros((self.n,1), dtype=int)
        self.log_regret_every_n = log_regret_every_n
        self.regret = 0.0
        self.regret_log = []
        self._t = 0
        self.startup()
        return
    
    @property
    def t(self):
        return self._t
    
    def startup(self):
        raise NotImplemented()
    
    def pull(self, arm_num):
        # pull the arm and update the mean for that arm
        observation = self.arms[arm_num].rvs()
        self.update_emp_mean(arm_num, observation)
        # mark that we have pulled
        self.Tis[arm_num] += 1
        # update regret
        self.update_regret(arm_num)
        self._t += 1
        
        # check for logging regret
        if self.log_regret_every_n is not None:
            if self.t % self.log_regret_every_n == 0:
                self.regret_log.append((self.t, self.regret))
        return
    
    @property
    def means(self):
        return np.array([arm.mean() for arm in self.arms])
    
    @property
    def true_best(self):
        return np.argmax(self.means)
    
    @property
    def delts(self):
        best = max(self.means)
        return best - self.means
    
    def update_emp_mean(self, arm_num, observation):
        """Update empirical for an arm given a new observation
        
        Properties:
         arm_num - int index of arm to update
         obervation - float of observation from that arm
        """
        old_mean = self.emp_means[arm_num]
        new_mean = (old_mean * self.Tis[arm_num] + observation)/(self.Tis[arm_num] + 1)
        self.emp_means[arm_num] = new_mean
        return
    
    def update_regret(self, arm_num):
        """Update regret by specifying which arm was pulled.
        
        Properties:
         arm_num - int index of arm pulled
        """
        self.regret += self.delts[arm_num]
        return
    
    def step(self):
        raise NotImplemented()
    
    def run(self):
        """Run the algorithm until max time"""
        for t in range(self.T-self.t):
            self.step()
            
            
class UCB(Agent):
    """Selects arm with highest upper confidence bound at each step."""
    def __init__(self, T: int, arms: List[object], log_regret_every_n: int = None):
        super().__init__(T, arms, log_regret_every_n)
        return
    
    def startup(self):
        """Pull each arm once"""
        for i in range(self.n):
            self.pull(i)
        return
    
    @property
    def ucbs(self):
        """values of the ucbs, same length as arms"""
        cbs = np.sqrt(2*np.log(2*self.n*self.T**2)/self.Tis)
        ucbs = self.emp_means + cbs
        return ucbs
    
    def step(self):
        """determine the arm with largest ucb and pull it"""
        It = np.argmax(self.ucbs)
        self.pull(It)
        return
    
class ETC(Agent):
    """Pulls each arm m time and then chooses the best arm forever after"""
    def __init__(self, T: int, arms: List[object], m: int,  log_regret_every_n: int = None):
        super().__init__(T, arms,  log_regret_every_n)
        self.m = m
        return
    
    def startup(self):
        # no additional initialization needed
        return
    
    def step(self):
        # this class indexex time starting at 0 instead of 1
        t_ = self.t+1
        if t_ <= self.m*self.n:
            It = (t_ % self.n)
        else:
            It = np.argmax(self.emp_means)
        self.pull(It)
        return
        
        
class GaussianThompson(Agent):
    """Assumes posterior distributions for arms are independant.
    
    Note that the conjugate liklihood to the normal prior is also normal
    """
    def __init__(self, T: int, arms: List[object], prior_mean: float, prior_var: float, log_regret_every_n: int = None):
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        super().__init__(T, arms, log_regret_every_n)
        return
    
    def startup(self):
        self.posteriors = [scipy.stats.norm(loc=self.prior_mean, scale=np.sqrt(self.prior_var)) for i in range(self.n)]
        return
    
    def update_posterior(self, arm_num):
        old_precision = 1/self.posteriors[arm_num].var()
        old_mean = self.posteriors[arm_num].mean()
        
        # update mean
        new_mean = ((old_mean * old_precision) + (self.Tis[arm_num] * self.emp_means[arm_num]))/(old_precision + self.Tis[arm_num])
        new_precision = old_precision + 1
        self.posteriors[arm_num] = scipy.stats.norm(loc=new_mean, scale=np.sqrt(1/new_precision))
        return
    
    def step(self):
        sample_means = [post.rvs() for post in self.posteriors]
        It = np.argmax(sample_means)
        self.pull(It)
        self.update_posterior(It)
        return
    
###################################################################################

def run_agent(agent):
    agent.run()
    return agent

# Problem 4.1
T = 150000
mus = [.1]
for i in range(9):
    mus.append(0)

arms = [scipy.stats.norm(loc=mu) for mu in mus]
    
agents = [
    UCB(T, arms, log_regret_every_n=10),
    ETC(T, arms, m=5, log_regret_every_n=10),
    ETC(T, arms, m=10, log_regret_every_n=10),
    ETC(T, arms, m=100, log_regret_every_n=10),
    GaussianThompson(T, arms, prior_mean=0, prior_var=1, log_regret_every_n=10)
]

ucb, etc1, etc2, etc3, gt = Parallel(n_jobs=5)(delayed(run_agent)(agent) for agent in agents)

# create dataframe
t, r_ucb = np.array(ucb.regret_log).T
_, r_etc1 = np.array(etc1.regret_log).T
_, r_etc2 = np.array(etc2.regret_log).T
_, r_etc3 = np.array(etc3.regret_log).T
_, r_gt = np.array(gt.regret_log).T

df = pd.DataFrame({
    't': t,
    'UCB': r_ucb,
    f'ETC {etc1.m}': r_etc1,
    f'ETC {etc2.m}': r_etc2,
    f'ETC {etc3.m}': r_etc3,
    'GT': r_gt
})

df.to_csv('p4.1.csv')

# Problem 4.2
T = 150000
mus = [1]
for i in range(2,41):
    mus.append(1-1/np.sqrt(i-1))
print(mus)
arms = [scipy.stats.norm(loc=mu) for mu in mus]
    
agents = [
    UCB(T, arms, log_regret_every_n=10),
    ETC(T, arms, m=5, log_regret_every_n=10),
    ETC(T, arms, m=10, log_regret_every_n=10),
    ETC(T, arms, m=100, log_regret_every_n=10),
    GaussianThompson(T, arms, prior_mean=0, prior_var=1, log_regret_every_n=10)
]

ucb, etc1, etc2, etc3, gt = Parallel(n_jobs=5)(delayed(run_agent)(agent) for agent in agents)

# create dataframe
t, r_ucb = np.array(ucb.regret_log).T
_, r_etc1 = np.array(etc1.regret_log).T
_, r_etc2 = np.array(etc2.regret_log).T
_, r_etc3 = np.array(etc3.regret_log).T
_, r_gt = np.array(gt.regret_log).T

df = pd.DataFrame({
    't': t,
    'UCB': r_ucb,
    f'ETC {etc1.m}': r_etc1,
    f'ETC {etc2.m}': r_etc2,
    f'ETC {etc3.m}': r_etc3,
    'GT': r_gt
})

df.to_csv('p4.2.csv')


