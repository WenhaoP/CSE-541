import scipy.stats
from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style('whitegrid')
sns.set_context('talk')

def sample_X(d: int, a: float, n: int):
    def sample_xi(d: int, a: float):
        sigma2s = [j**(-a) for j in range(1,d+1)]
        return np.array(np.random.normal(0, np.sqrt(sigma2s))).reshape(1,-1)
    return np.concatenate([sample_xi(d, a) for i in range(n)])

class FrankWolfe:
    """
    Attributes
    ----------
    X : (n, d) matrix of arms
    XXT : (n, d, d) matrix of self interaction
    lamb : (n,1) vector of arm distribution
    A : (d, d) matrix of weighted self interactions
    """
    def __init__(self, X: np.ndarray, N: int):
        d = X.shape[1]
        n = X.shape[0]
        
        self.d = d

        self.n = n
        self.N = N
        self.X = X
        self.startup()
        return
    
    def startup(self):
        # pull first 2d arms and set lambda
        pulls = np.random.choice(list(range(self.n)), size=(2*self.d))
        values, counts_ = np.unique(pulls, return_counts=True)
        counts = []
        for i in range(self.n):
            if i in values:
                counts.append(counts_[np.argwhere(values==i)][0][0])
            else:
                counts.append(0)
        self.lamb = (np.array(counts)/(2*self.d)).reshape(-1,1)
        # start time after startup
        self.t = 2*self.d
        # the raw XXT matrix
        xs = []
        for x in self.X:
            xs.append(np.matmul(x.reshape(-1,1), x.reshape(1,-1)))
        self.XXT = np.array(xs)
        return
    
    @property
    def A(self):
        """Function of current lambda"""
        return np.sum(self.lamb.reshape(-1,1,1)*self.XXT, axis=0)
    
    @property
    def D_lamb(self):
        return -np.log(np.linalg.det(self.A))
        
    def update_for_next(self, arm_index):
        """Update lambda and time at the end of the iteration."""
        indicator = np.zeros((self.n,1))
        indicator[arm_index] = 1
        # at this lambda and t are both at their end of previous iteration state, eg t has not been updated
        self.lamb = (self.lamb*self.t + indicator)/(self.t+1)
        self.t += 1
        return
    
    def g_prime_i(self, i):
        """Compute gradient of design equation for one example's lambda"""
        return -np.matmul(np.matmul(self.X[i].reshape(1,-1), np.linalg.inv(self.A)),  self.X[i].reshape(-1,1))
    
    def take_step(self):
        # get the gradient vector
        self.g_prime = np.concatenate([self.g_prime_i(i) for i in range(self.n)])
        It = np.argmin(self.g_prime)
        self.update_for_next(It)
        # print(f'Time {self.t}, pull {It}, D = {self.D_lamb}')
        return 
    
    def run(self):
        while self.t < self.N:
            self.take_step()
        return self.lamb, self.D_lamb
    
    
##### 
import numpy as np
n =300
X = np . concatenate ( ( np . linspace (0 ,1 ,50) , 0.25+ 0.01* np . random . randn (250) ) , 0)
X = np . sort (X)

K = np . zeros ((n ,n))
for i in range ( n):
    for j in range ( n):
        K[i ,j] = 1+ min (X[i] ,X[j ])
e , v = np . linalg . eigh (K ) # eigenvalues are increasing in order
d = 30
Phi = np . real (v @ np . diag ( np . sqrt ( np . abs (e ))) ) [: ,(n -d ) ::]

def f(x):
    return -x**2 + x*np.cos(8*x)+np.sin(15*x)

f_star = f(X)
theta = np.linalg.lstsq(Phi, f_star, rcond=None)[0]
f_hat = Phi @ theta

def observe(idx):
    return f(X[idx]) + np.random.randn(len(idx))

def sample_and_estimate(X, lbda, tau):
    n, d = X.shape
    reg = 1e-6
    idx = np.random.choice(np.arange(n), size=tau, p=lbda)
    y = observe(idx)
    
    XtX = X[idx].T @ X[idx]
    XtY = X[idx].T @ y
    
    theta = np.linalg.lstsq(XtX + reg*np.eye(d), XtY, rcond=None)[0]
    print(theta.shape)
    y_hat = Phi @ theta
    print(y_hat.shape)
    return y_hat, XtX

class Agent:
    """Exploratory agent. Modified from HW1
    
    Properties:
     T - max time to run
     X - matrix of current arm features shape (number arms, d)
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
    def __init__(self, T: int, X: np.ndarray, arms: List[object], log_regret_every_n: int = None):
        self.T = T
        assert len(X) == len(arms)
        self.X = X
        self.d = X.shape[1]
        self.arms = np.array(arms)
        self.log_regret_every_n = log_regret_every_n
        self.regret = 0.0
        self.regret_log = []
        self._t = 0
        self.S = np.zeros((self.d,1))
        self.V = np.eye(self.d)
        self.startup()
        return
    
    @property
    def n(self):
        return len(self.X)
    
    @property
    def t(self):
        return self._t
    
    def startup(self):
        raise NotImplemented()
    
    def pull(self, arm_num):
        # pull the arm and update the mean for that arm
        observation = self.arms[arm_num].rvs()
        self.update_states(arm_num, observation)
        # update regret
        self.update_regret(observation)
        self._t += 1
        print('Timestep t=', self._t)
        
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
    
    def update_states(self, arm_num, observation):
        """Update V and S
        
        Properties:
         arm_num - int index of arm to update
         obervation - float of observation from that arm
        """
        raise NotImplemented()
        return
    
    def update_regret(self, observation):
        """Update regret by specifying which arm was pulled.
        
        Properties:
         arm_num - int index of arm pulled
        """
        self.regret += max(self.means) - observation
        return
    
    def step(self):
        raise NotImplemented()
    
    def run(self):
        """Run the algorithm until max time"""
        while self.t < self.T:
            self.step()
            
class Eliminator(Agent):
    
    def startup(self):
        self.tau = 100
        self.del_ = 1/self.T
        self.frank_size = 200
        self.V0 = np.array(self.V)
        return
    
    def step(self):
        frank = FrankWolfe(self.X, self.frank_size)
        lbda, _ = frank.run()
        arms_indexes = np.random.choice(range(self.n), size = self.tau, replace=True, p=lbda.reshape(-1))
        for arm_num in arms_indexes:
            self.pull(arm_num)
        
        theta_k = np.linalg.inv(self.V) @ self.S
        
        # now we need to test which arms are out of bounds
        # find index of max arm
        y_hats = self.X @ theta_k
        i_star = np.argmax(y_hats)
        diff = self.X[i_star] - self.X
        # arm performance difference
        e_hats = diff @ theta_k
        # get the cutoff constant for this round
#         print('det V: ', np.linalg.det(self.V))
        beta = np.sqrt(2*np.log(1/self.del_)+np.log(np.linalg.det(self.V)/np.linalg.det(self.V0)))
        # determine the cuttoff bound
        bound = []
        for x_diff in diff:
            bound.append(beta * np.sqrt(x_diff.reshape(1,-1) @ np.linalg.inv(self.V) @ x_diff.reshape(-1,1)))
        bound = np.array(bound).reshape(-1,1)
        # flag the bad arms
        passed = e_hats < bound
        passed[i_star] = True
#         print('Arms failed with means: ', self.means[~passed.reshape(-1)])
        indexes_passed = np.argwhere(passed)[:,0]
        # remove these arms
        self.X = np.array(self.X[indexes_passed])
        self.arms = np.array(self.arms[indexes_passed])
        return y_hats, e_hats, bound, passed
    
    def update_states(self, arm_num, observation):
        xt = self.X[arm_num]
        self.V = self.V + xt.reshape(-1,1) @ xt.reshape(1,-1)
        self.S = self.S + xt.reshape(-1,1) * observation
        return
    
    def run(self):
        """Run the algorithm until max time"""
        while self.t < self.T:
            if self.n == 1:
                self.pull(0)
            else:
                self.step()
        return
    
class UCB(Agent):
    def startup(self):
        self.del_ = 1/self.T
        self.V0 = np.array(self.V)
        return
    
    def step(self):
        beta = np.sqrt(2*np.log(1/self.del_)+np.log(np.linalg.det(self.V)/np.linalg.det(self.V0)))
        theta = np.linalg.inv(self.V) @ self.S
        y_hats = self.X @ theta
        bound = []
        for x in self.X:
            bound.append(beta * np.sqrt(x.reshape(1,-1) @ np.linalg.inv(self.V) @ x.reshape(-1,1)))
        bound = np.array(bound).reshape(-1,1) 
        It = np.argmax(y_hats + bound)
        self.pull(It)
        return
    
    def update_states(self, arm_num, observation):
        xt = self.X[arm_num]
        self.V = self.V + (xt.reshape(-1,1) @ xt.reshape(1,-1))
        self.S = self.S + (xt.reshape(-1,1) * observation)
        return
    
class Thompson(Agent):
    def startup(self):
        self.V0 = np.array(self.V)
        return
    
    def step(self):
        mu = (np.linalg.inv(self.V) @ self.S).reshape(-1)
        cov = np.linalg.inv(self.V)
        posterior = scipy.stats.multivariate_normal(mean=mu, cov=cov)
        theta = posterior.rvs()
        It = np.argmax(self.X @ theta.reshape(-1,1))
        self.pull(It)
        return
    
    def update_states(self, arm_num, observation):
        xt = self.X[arm_num]
        self.V = self.V + xt.reshape(-1,1) @ xt.reshape(1,-1)
        self.S = self.S + xt.reshape(-1,1) * observation
        return
    