"""Empirical tests of the agents - plots"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('p4.1.csv', index_col = 0)
t = df['t'].values
r_ucb = df['UCB'].values
r_etc1 = df['ETC 5'].values
r_etc2 = df['ETC 10'].values
r_etc3 = df['ETC 100'].values
r_gt = df['GT'].values

fig, ax = plt.subplots()

ax.plot(t, r_ucb, c='tab:red', label='ucb')
ax.plot(t, r_etc1, c='tab:blue', label=f'etc, m=5', alpha = .4)
ax.plot(t, r_etc2, c='tab:blue', label=f'etc, m=10', alpha = .6)
ax.plot(t, r_etc3, c='tab:blue', label=f'etc, m=100')
ax.plot(t, r_gt, c='tab:green', label=f'thompson')
ax.set_xlabel('timestep')
ax.set_ylabel('regret')
plt.legend()
plt.savefig('p4.1.png', bbox_inches='tight', dpi=600)

#############
df = pd.read_csv('p4.2.csv', index_col = 0)
t = df['t'].values
r_ucb = df['UCB'].values
r_etc1 = df['ETC 5'].values
r_etc2 = df['ETC 10'].values
r_etc3 = df['ETC 100'].values
r_gt = df['GT'].values
fig, ax = plt.subplots()

ax.plot(t, r_ucb, c='tab:red', label='ucb')
ax.plot(t, r_etc1, c='tab:blue', label=f'etc, m=5', alpha = .4)
ax.plot(t, r_etc2, c='tab:blue', label=f'etc, m=10', alpha = .6)
ax.plot(t, r_etc3, c='tab:blue', label=f'etc, m=100')
ax.plot(t, r_gt, c='tab:green', label=f'thompson')
ax.set_xlabel('timestep')
ax.set_ylabel('regret')
plt.legend()
plt.savefig('p4.2.png', bbox_inches='tight', dpi=600)
