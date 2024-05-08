import sys
import agents
import sklearn.decomposition
import numpy as np
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed



def run(T,d,gamma):

    sys.stdout = open(f'./outs/T{T}_d{d}_gamma{gamma}.out', 'w')

    C = np.load('C.npy')
    y = np.load('y.npy')
    
    C = C/256
    
    if d is not None:
        pca = sklearn.decomposition.PCA(d)
        C = pca.fit_transform(C)
    C = C / np.linalg.norm(C, axis=1).reshape(-1,1)

    ucb = agents.UCB(C, y, T, gamma=gamma, beta_type='det', max_bound=False)
    ucb.run()

    fig, ax = plt.subplots()
    ax.plot(*np.array(ucb.R_log).T)
    plt.savefig(f'./figs/T{T}_d{d}_gamma{gamma}.png', bbox_inches='tight')

    sys.stdout.close()
    return


def run_combos():
    gammas = [.1,.5,.9,1.0,1.1,2,10]
    ds = [10, 50, 200, 700, None]
    Ts = [10000]
    perms = itertools.product(Ts, ds, gammas)
    Parallel(n_jobs=10)(delayed(run)(*i) for i in perms)
    return

if __name__ == '__main__':
    run_combos()
    
    

