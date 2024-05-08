import sys

import numpy as np
import sklearn.decomposition
import sklearn.linear_model
from joblib import Parallel, delayed

import agents

n_replicates = 3
n_jobs = int(sys.argv[1])

class UCB_maxed(agents.UCB):
    maxed = True


kwargs = {
    'ETC_world': {'tau': 6422},
    'ETC_bias': {'tau': 6422},
    'FTL': {'tau': 6422},
    'UCB': dict(gamma = 2, beta_type='det', max_bound=False),
    'UCB_maxed': dict(gamma = 2, beta_type='det', max_bound=True),
    'Thompson': {}
}

C = np.load('C.npy')
y = np.load('y.npy')

def test_agent_in_replicate(agent_class, kwargs):
    R_logs = []
    
    for i in range(n_replicates):
        agent = agent_class(C, y, 50000, **kwargs)
        agent.run()
        R_logs.append(np.array(agent.R_log)[:,1])
        t_vec = np.array(agent.R_log)[:,0]
    
    R_logs = np.array(R_logs).T
    np.save(f'{agent_class.__name__}.npy', R_logs)
    np.save('t.npy', t_vec)
    
if __name__ == '__main__':
    inputs = zip(
        [
            agents.ETC_world, agents.ETC_bias, agents.FTL, agents.UCB, agents.Thompson, UCB_maxed
        ],
        [
            kwargs['ETC_world'], kwargs['ETC_bias'], kwargs['FTL'], kwargs['UCB'], kwargs['Thompson'], kwargs['UCB_maxed']
        ]
    )
    Parallel(n_jobs=n_jobs)(delayed(test_agent_in_replicate)(*i) for i in inputs)
    
