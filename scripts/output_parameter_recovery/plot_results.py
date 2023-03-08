import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', family='serif')


data = pd.read_csv('./sim_k5_t10_n100/sim_k5_t10_n100.csv')
data['t'] = 10
data['k'] = 5
data['n'] = 100

df = pd.read_csv('./sim_k5_t50_n100/sim_k5_t50_n100.csv')
df['t'] = 50
df['k'] = 5
df['n'] = 100
data = pd.concat((data, df))

df = pd.read_csv('./sim_k5_t100_n100/sim_k5_t100_n100.csv')
df['t'] = 100
df['k'] = 5
df['n'] = 100
data = pd.concat((data, df))

df = pd.read_csv('./sim_k10_t10_n100/sim_k10_t10_n100.csv')
df['t'] = 10
df['k'] = 10
df['n'] = 100
data = pd.concat((data, df))

df = pd.read_csv('./sim_k20_t10_n100/sim_k20_t10_n100.csv')
df['t'] = 10
df['k'] = 20
df['n'] = 100
data = pd.concat((data, df))

df = pd.read_csv('./sim_k5_t10_n50/sim_k5_t10_n50.csv')
df['t'] = 10
df['k'] = 5
df['n'] = 50
data = pd.concat((data, df))

df = pd.read_csv('./sim_k5_t10_n200/sim_k5_t10_n200.csv')
df['t'] = 10
df['k'] = 5
df['n'] = 200
data = pd.concat((data, df))

df = pd.read_csv('./sim_k5_t10_n500/sim_k5_t10_n500.csv')
df['t'] = 10
df['k'] = 5
df['n'] = 500
data = pd.concat((data, df))

df = pd.read_csv('./sim_k5_t10_n1000/sim_k5_t10_n1000.csv')
df['t'] = 10
df['k'] = 5
df['n'] = 1000
data = pd.concat((data, df))


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4), sharey=True)
axes[0].set_ylim(0.5 * 1e-6, 1)

color = 'gray'
fontsize = 24
xlabel = 16
labelsize = 16
dpi = 100
ext = 'pdf'

sns.boxplot(x='k', y='X_rel', data=data.query('t == 10').query('n == 100'), color=color, ax=axes[0])
axes[0].set(yscale='log')
axes[0].set_ylabel('Relative Error', fontsize=fontsize)
axes[0].set_xlabel('Number of Layers', fontsize=xlabel)
axes[0].tick_params(axis='y', labelsize=labelsize)
axes[0].tick_params(axis='x', labelsize=labelsize)
axes[0].set_title(r'$\mathcal{X}_{1:T}$', fontsize=24)

sns.boxplot(x='k', y='lambda_rel', data=data.query('t == 10').query('n == 100'), color=color, ax=axes[1])
axes[1].set(yscale='log')
axes[1].set_ylabel('')
axes[1].set_xlabel('Number of Layers', fontsize=xlabel)
axes[1].tick_params(axis='y', labelsize=labelsize)
axes[1].tick_params(axis='x', labelsize=labelsize)
axes[1].set_title(r'$\Lambda_{1:K}$', fontsize=24)

sns.boxplot(x='k', y='delta_rel', data=data.query('t == 10').query('n == 100'), color=color, ax=axes[2])
axes[2].set(yscale='log')
axes[2].set_ylabel('')
axes[2].set_xlabel('Number of Layers', fontsize=xlabel)
axes[2].tick_params(axis='y', labelsize=labelsize)
axes[2].tick_params(axis='x', labelsize=labelsize)
axes[2].set_title(r'$\mathbf{\delta}_{1:K, 1:T}$', fontsize=24)

sns.boxplot(x='k', y='proba_rel', data=data.query('t == 10').query('n == 100'), color=color, ax=axes[3])
axes[3].set(yscale='log')
axes[3].set_ylabel('')
axes[3].set_xlabel('Number of Layers', fontsize=xlabel)
axes[3].tick_params(axis='y', labelsize=labelsize)
axes[3].tick_params(axis='x', labelsize=labelsize)
axes[3].set_title(r'$P_{ijt}^k$', fontsize=24)

fig.savefig('relative_error_layers.{}'.format(ext), dpi=dpi, bbox_inches='tight')


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4), sharey=True)
axes[0].set_ylim(0.5 * 1e-6, 1)

sns.boxplot(x='t', y='X_rel', data=data.query('k == 5').query('n == 100'), color=color, ax=axes[0])
axes[0].set(yscale='log')
axes[0].set_ylabel('Relative Error', fontsize=fontsize)
axes[0].set_xlabel('Number of Time Steps', fontsize=xlabel)
axes[0].tick_params(axis='y', labelsize=labelsize)
axes[0].tick_params(axis='x', labelsize=labelsize)
axes[0].set_title(r'$\mathcal{X}_{1:T}$', fontsize=24)

sns.boxplot(x='t', y='lambda_rel', data=data.query('k == 5').query('n == 100'), color=color, ax=axes[1])
axes[1].set(yscale='log')
axes[1].set_ylabel('')
axes[1].set_xlabel('Number of Time Steps', fontsize=xlabel)
axes[1].tick_params(axis='y', labelsize=labelsize)
axes[1].tick_params(axis='x', labelsize=labelsize)
axes[1].set_title(r'$\Lambda_{1:K}$', fontsize=24)

sns.boxplot(x='t', y='delta_rel', data=data.query('k == 5').query('n == 100'), color=color, ax=axes[2])
axes[2].set(yscale='log')
axes[2].set_ylabel('')
axes[2].set_xlabel('Number of Time Steps', fontsize=xlabel)
axes[2].tick_params(axis='y', labelsize=labelsize)
axes[2].tick_params(axis='x', labelsize=labelsize)
axes[2].set_title(r'$\mathbf{\delta}_{1:K, 1:T}$', fontsize=24)

sns.boxplot(x='t', y='proba_rel', data=data.query('k == 5').query('n == 100'), color=color, ax=axes[3])
axes[3].set(yscale='log')
axes[3].set_ylabel('')
axes[3].set_xlabel('Number of Time Steps', fontsize=xlabel)
axes[3].tick_params(axis='y', labelsize=labelsize)
axes[3].tick_params(axis='x', labelsize=labelsize)
axes[3].set_title(r'$P_{ijt}^k$', fontsize=24)

fig.savefig('relative_error_time.{}'.format(ext), dpi=dpi, bbox_inches='tight')


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4), sharey=True)
axes[0].set_ylim(0.5 * 1e-6, 1)

sns.boxplot(x='n', y='X_rel', data=data.query('k == 5').query('t == 10'), color=color, ax=axes[0])
axes[0].set(yscale='log')
axes[0].set_ylabel('Relative Error', fontsize=fontsize)
axes[0].set_xlabel('Number of Nodes', fontsize=xlabel)
axes[0].tick_params(axis='y', labelsize=labelsize)
axes[0].tick_params(axis='x', labelsize=labelsize)
axes[0].set_title(r'$\mathcal{X}_{1:T}$', fontsize=24)

sns.boxplot(x='n', y='lambda_rel', data=data.query('k == 5').query('t == 10'), color=color, ax=axes[1])
axes[1].set(yscale='log')
axes[1].set_ylabel('')
axes[1].set_xlabel('Number of Nodes', fontsize=xlabel)
axes[1].tick_params(axis='y', labelsize=labelsize)
axes[1].tick_params(axis='x', labelsize=labelsize)
axes[1].set_title(r'$\Lambda_{1:K}$', fontsize=24)

sns.boxplot(x='n', y='delta_rel', data=data.query('k == 5').query('t == 10'), color=color, ax=axes[2])
axes[2].set(yscale='log')
axes[2].set_ylabel('')
axes[2].set_xlabel('Number of Nodes', fontsize=xlabel)
axes[2].tick_params(axis='y', labelsize=labelsize)
axes[2].tick_params(axis='x', labelsize=labelsize)
axes[2].set_title(r'$\mathbf{\delta}_{1:K, 1:T}$', fontsize=24)

sns.boxplot(x='n', y='proba_rel', data=data.query('k == 5').query('t == 10'), color=color, ax=axes[3])
axes[3].set(yscale='log')
axes[3].set_ylabel('')
axes[3].set_xlabel('Number of Nodes', fontsize=xlabel)
axes[3].tick_params(axis='y', labelsize=labelsize)
axes[3].tick_params(axis='x', labelsize=labelsize)
axes[3].set_title(r'$P_{ijt}^k$', fontsize=24)

fig.savefig('relative_error_nodes.{}'.format(ext), dpi=dpi, bbox_inches='tight')
