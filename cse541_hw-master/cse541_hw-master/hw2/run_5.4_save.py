import utils as ut
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats

from joblib import Parallel, delayed

T = 40000
replicates = 5
arms = [scipy.stats.norm(loc=f_y, scale=1) for f_y in ut.f(ut.X)]

def evaluate_alg(algorithm_class):
    regrets = []
    for i in range(replicates):
        algorithm = algorithm_class(T, ut.Phi, arms, log_regret_every_n=100)
        algorithm.run()
        regret = np.array(algorithm.regret_log)
        time = regret[:,0]
        regret = regret[:,1]
        regrets.append(regret.reshape(-1))
    rmeans = np.mean(regrets, axis=0)
    rstds = np.std(regrets, axis=0)
    return np.array([time, rmeans, rstds]).T


out = Parallel(n_jobs=3)(delayed(evaluate_alg)(alg) for alg in [ut.Eliminator, ut.UCB, ut.Thompson])

np.save('elim.npy', out[0])
np.save('ucb.npy', out[1])
np.save('thomp.npy', out[2])
