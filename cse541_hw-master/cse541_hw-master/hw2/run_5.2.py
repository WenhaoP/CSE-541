import utils as ut
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style('whitegrid')
sns.set_context('talk')

n = np.array([10+2**i for i in range(1,11)])
a = np.array([0.0, .5, 1.0, 2.0])
d = 10
N = 1000

replicates = 3
Ds_means = np.zeros((len(a), len(n)))
Ds_std = np.zeros((len(a), len(n)))

complete = 0
for i, a_ in enumerate(a):
    for j, n_ in enumerate(n):
        D = []
        for k in range(replicates):
            X = ut.sample_X(d, a_, n_)
            alg = ut.FrankWolfe(X, N)
            _, D_ = alg.run()
            D.append(D_)
        Ds_means[i,j] = np.mean(D)
        Ds_std[i,j] = np.std(D)
        complete += n_
        print(f'a = {a_}, n = {n_}. D = {Ds_means[i,j]} ({Ds_std[i,j]}). {complete/(np.sum(n)*3)*100}% done.')
        
dfs = []
for i, a_ in enumerate(a):
    df = pd.DataFrame(data=n, columns=['n'])
    df['mean'] = Ds_means[i]
    df['std'] = Ds_std[i]
    df['a'] = a_
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
fig, ax = plt.subplots()
for i, a_ in enumerate(a):
    ax.errorbar(n, Ds_means[i], yerr=2*Ds_std[i], label=a_)
ax.set_xlabel('n')
ax.set_ylabel('D optimality')
ax.set_title('D optimality for different values of a. Frank Wolfe algorithm in three replicates with N=1000')
plt.legend(title='a')
plt.savefig('5.2.png', bbox_inches='tight', dpi=300)