import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
#plt.rc('ytick', labelsize=16)


data = pd.read_csv('./sim_k5_t10_n100/sim_k5_t10_n100.csv')
data['t'] = 10
data['k'] = 5
data['n'] = 100
data['Simulation'] = r'$\sigma = 0.5 \, / \, T$'

df = pd.read_csv('../output_parameter_recovery/sim_k5_t10_n100/sim_k5_t10_n100.csv')
df['t'] = 10
df['k'] = 5
df['n'] = 100
df['Simulation'] = r'$\sigma = 0.05$'
data = pd.concat((data, df))

df = pd.read_csv('./sim_k5_t50_n100/sim_k5_t50_n100.csv')
df['t'] = 50
df['k'] = 5
df['n'] = 100
df['Simulation'] = r'$\sigma = 0.5 \, / \, T$'
data = pd.concat((data, df))

df = pd.read_csv('../output_parameter_recovery/sim_k5_t50_n100/sim_k5_t50_n100.csv')
df['t'] = 50
df['k'] = 5
df['n'] = 100
df['Simulation'] = r'$\sigma = 0.05$'
data = pd.concat((data, df))

df = pd.read_csv('./sim_k5_t100_n100/sim_k5_t100_n100.csv')
df['t'] = 100
df['k'] = 5
df['n'] = 100
df['Simulation'] = r'$\sigma = 0.5 \, / \, T$'
data = pd.concat((data, df))

df = pd.read_csv('../output_parameter_recovery/sim_k5_t100_n100/sim_k5_t100_n100.csv')
df['t'] = 100
df['k'] = 5
df['n'] = 100
df['Simulation'] = r'$\sigma = 0.05$'
data = pd.concat((data, df))

color = 'gray'
fontsize = 24
xlabel = 16
labelsize = 16
dpi = 100
ext = 'pdf'

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4), sharey=True)
axes[0].set_ylim(1e-5, 10)

sns.boxplot(x='t', y='X_rel', data=data.query('k == 5').query('n == 100'), hue='Simulation', ax=axes[0], palette='Set2', hue_order=[r'$\sigma = 0.05$', r'$\sigma = 0.5 \, / \, T$'])
axes[0].set(yscale='log')
axes[0].set_ylabel('Relative Error', fontsize=fontsize)
axes[0].set_xlabel('Number of Time Steps', fontsize=xlabel)
axes[0].tick_params(axis='y', labelsize=labelsize)
axes[0].tick_params(axis='x', labelsize=labelsize)
axes[0].set_title(r'$\mathcal{X}_{1:T}$', fontsize=24)

sns.boxplot(x='t', y='lambda_rel', data=data.query('k == 5').query('n == 100'), hue='Simulation', ax=axes[1], palette='Set2', hue_order=[r'$\sigma = 0.05$', r'$\sigma = 0.5 \, / \, T$'])
axes[1].set(yscale='log')
axes[1].set_ylabel('')
axes[1].set_xlabel('Number of Time Steps', fontsize=xlabel)
axes[1].tick_params(axis='y', labelsize=labelsize)
axes[1].tick_params(axis='x', labelsize=labelsize)
axes[1].set_title(r'$\Lambda_{1:K}$', fontsize=24)

sns.boxplot(x='t', y='delta_rel', data=data.query('k == 5').query('n == 100'), ax=axes[2], hue='Simulation', palette='Set2', hue_order=[r'$\sigma = 0.05$', r'$\sigma = 0.5 \, / \, T$'])
axes[2].set(yscale='log')
axes[2].set_ylabel('')
axes[2].set_xlabel('Number of Time Steps', fontsize=xlabel)
axes[2].tick_params(axis='y', labelsize=labelsize)
axes[2].tick_params(axis='x', labelsize=labelsize)
axes[2].set_title(r'$\mathbf{\delta}_{1:K, 1:T}$', fontsize=24)

sns.boxplot(x='t', y='proba_rel', data=data.query('k == 5').query('n == 100'), ax=axes[3], hue='Simulation', palette='Set2', hue_order=[r'$\sigma = 0.05$', r'$\sigma = 0.5 \, / \, T$'])
axes[3].set(yscale='log')
axes[3].set_ylabel('')
axes[3].set_xlabel('Number of Time Steps', fontsize=xlabel)
axes[3].tick_params(axis='y', labelsize=labelsize)
axes[3].tick_params(axis='x', labelsize=labelsize)
axes[3].set_title(r'$P_{ijt}^k$', fontsize=24)

fig.savefig('relative_error_time.{}'.format(ext), dpi=dpi, bbox_inches='tight')
