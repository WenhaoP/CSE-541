import utils as ut
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style('whitegrid')
sns.set_context('talk')

X = ut.X

frank = ut.FrankWolfe(ut.Phi, 1000)
lbda_frank, D = frank.run()

lbda_frank = lbda_frank.reshape(-1)
f_G_Phi, A = ut.sample_and_estimate(ut.Phi, lbda_frank, 1000)
conf_G = np . sqrt ( np .sum ( ut.Phi @ np . linalg . inv ( A) * ut.Phi , axis =1) )

lbda = np . ones (ut.n)/ ut.n
f_unif_Phi , A = ut.sample_and_estimate( ut.Phi , lbda , 1000)
conf_unif = np . sqrt ( np . sum ( ut.Phi @ np . linalg . inv (A) * ut.Phi , axis =1) )

fig, ax = plt.subplots()
ax.plot(X, np.cumsum(lbda), label = "uniform", c='tab:blue')
ax.plot(X, np.cumsum(lbda_frank), label = "D optimal", c='tab:orange')
ax.set_xlabel('x')
ax.set_ylabel('Cumulative Probability')
ax.set_title('CDF of design space using uniform sampling and using D optimal sampling')
plt.legend()
plt.savefig('5.3a.png', bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.plot(X, ut.f_star, label ="True function", color='k', lw=3)
ax.plot(X, f_unif_Phi, label ="Uniform sampling of kernel space", color='tab:red', lw=1, ls='--')
ax.plot(X, f_G_Phi, label ="D optimal sampling of kernel space", color='tab:blue', lw=1, ls='--')

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.legend(bbox_to_anchor=(2.0,.7))
ax.set_title('Estimation of underlying function using kernel regression, \n with training data sampled uniformly and from d optimal space.')
plt.savefig('5.3b.png', bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(15,10))
ax.plot(X, np.abs(ut.f_star - f_G_Phi), label="D optimal error", color='tab:red')
ax.plot(X, np.abs(ut.f_star - f_unif_Phi), label="Uniform error", color='tab:red', ls='-.')
ax.plot(X, np.ones(X.shape)*np.sqrt(ut.d/ut.n), color='k', label="root(d/n)")
ax.plot(X, conf_G, label="D optimal confidence", color='tab:blue')
ax.plot(X, conf_unif, label="uniform confidence", color='tab:blue', ls='-.')

ax.set_xlabel('x')
plt.legend()

plt.savefig('5.3c.png', bbox_inches='tight', dpi=300)
plt.close()
