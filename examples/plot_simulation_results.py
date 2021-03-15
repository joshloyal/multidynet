import os
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.rc('font', family='serif')
color = 'gray'
fontsize = 16
labelsize = 16
dpi = 100
ext = 'pdf'


# process the results of the simulations
for dir_name in glob.glob('sim_k*'):
    data = None
    for file_name in glob.glob(os.path.join(dir_name, '*.csv')):
        if data is None:
            data = pd.read_csv(file_name)
        else:
            data = pd.concat((data, pd.read_csv(file_name)))

    data.to_csv(os.path.join(dir_name,  dir_name + '.csv'), index=False)


# read results
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


# create figures
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 6), sharey=True)


sns.boxplot(x='k', y='X_rel', data=data.query('t == 10').query('n == 100'),
            color=color, ax=axes[0])
axes[0].set(yscale='log')
axes[0].set_ylabel('Relative Error of Latent Positions', fontsize=fontsize)
axes[0].set_xlabel('Number of Layers', fontsize=fontsize)
axes[0].tick_params(axis='y', labelsize=labelsize)
axes[0].tick_params(axis='x', labelsize=labelsize)

sns.boxplot(x='k', y='lambda_rel', data=data.query('t == 10').query('n == 100'),
            color=color, ax=axes[1])
axes[1].set(yscale='log')
axes[1].set_ylabel(
    'Relative Error of Homophily Coefficients', fontsize=fontsize)
axes[1].set_xlabel('Number of Layers', fontsize=fontsize)
axes[1].tick_params(axis='y', labelsize=labelsize)
axes[1].tick_params(axis='x', labelsize=labelsize)

sns.boxplot(x='k', y='delta_rel', data=data.query('t == 10').query('n == 100'),
            color=color, ax=axes[2])
axes[2].set(yscale='log')
axes[2].set_ylabel('Relative error of Sociality Effects', fontsize=fontsize)
axes[2].set_xlabel('Number of Layers', fontsize=fontsize)
axes[2].tick_params(axis='y', labelsize=labelsize)
axes[2].tick_params(axis='x', labelsize=labelsize)

sns.boxplot(x='k', y='proba_rel', data=data.query('t == 10').query('n == 100'),
            color=color, ax=axes[3])
axes[3].set(yscale='log')
axes[3].set_ylabel(
    'Relative Error of Connection Probabilities', fontsize=fontsize)
axes[3].set_xlabel('Number of Layers', fontsize=fontsize)
axes[3].tick_params(axis='y', labelsize=labelsize)
axes[3].tick_params(axis='x', labelsize=labelsize)

fig.savefig(
    'relative_error_layers.{}'.format(ext), dpi=dpi, bbox_inches='tight')

fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

sns.boxplot(
    x='k', y='insample_auc', data=data.query('t == 10').query('n == 100'),
    color=color, ax=axes[0])
axes[0].set_ylabel('In-Sample AUC', fontsize=fontsize)
axes[0].set_xlabel('Number of Layers', fontsize=fontsize)

sns.boxplot(
    x='k', y='holdhout_auc', data=data.query('t == 10').query('n == 100'),
    color=color, ax=axes[1])
axes[1].set_ylabel('Held-Out AUC', fontsize=fontsize)
axes[1].set_xlabel('Number of Layers', fontsize=fontsize)

fig.savefig('perf_layers.{}'.format(ext), dpi=dpi)


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 6), sharey=True)

sns.boxplot(x='t', y='X_rel', data=data.query('k == 5').query('n == 100'),
            color=color, ax=axes[0])
axes[0].set(yscale='log')
axes[0].set_ylabel('Relative Error of Latent Positions', fontsize=fontsize)
axes[0].set_xlabel('Number of Time Steps', fontsize=fontsize)
axes[0].tick_params(axis='y', labelsize=labelsize)
axes[0].tick_params(axis='x', labelsize=labelsize)

sns.boxplot(x='t', y='lambda_rel', data=data.query('k == 5').query('n == 100'),
            color=color, ax=axes[1])
axes[1].set(yscale='log')
axes[1].set_ylabel(
    'Relative error of Homophily Coefficients', fontsize=fontsize)
axes[1].set_xlabel('Number of Time Steps', fontsize=fontsize)
axes[1].tick_params(axis='y', labelsize=labelsize)
axes[1].tick_params(axis='x', labelsize=labelsize)

sns.boxplot(x='t', y='delta_rel', data=data.query('k == 5').query('n == 100'),
            color=color, ax=axes[2])
axes[2].set(yscale='log')
axes[2].set_ylabel('Relative Error of Sociality Effects', fontsize=fontsize)
axes[2].set_xlabel('Number of Time Steps', fontsize=fontsize)
axes[2].tick_params(axis='y', labelsize=labelsize)
axes[2].tick_params(axis='x', labelsize=labelsize)

sns.boxplot(x='t', y='proba_rel', data=data.query('k == 5').query('n == 100'),
            color=color, ax=axes[3])
axes[3].set(yscale='log')
axes[3].set_ylabel(
    'Relative Error of Connection Probabilities', fontsize=fontsize)
axes[3].set_xlabel('Number of Time Steps', fontsize=fontsize)
axes[3].tick_params(axis='y', labelsize=labelsize)
axes[3].tick_params(axis='x', labelsize=labelsize)

fig.savefig('relative_error_time.{}'.format(ext), dpi=dpi, bbox_inches='tight')


fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

sns.boxplot(
    x='t', y='insample_auc', data=data.query('k == 5').query('n == 100'),
    color=color, ax=axes[0])
axes[0].set_ylabel('In-Sample AUC', fontsize=fontsize)
axes[0].set_xlabel('Number of Time Steps', fontsize=fontsize)

sns.boxplot(
    x='t', y='holdhout_auc', data=data.query('k == 5').query('n == 100'),
    color=color, ax=axes[1])
axes[1].set_ylabel('Held-Out AUC', fontsize=fontsize)
axes[1].set_xlabel('Number of Time Steps', fontsize=fontsize)

fig.savefig('perf_time.{}'.format(ext), dpi=dpi)


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 6), sharey=True)

sns.boxplot(x='n', y='X_rel', data=data.query('k == 5').query('t == 10'),
            color=color, ax=axes[0])
axes[0].set(yscale='log')
axes[0].set_ylabel('Relative Error of Latent Positions', fontsize=fontsize)
axes[0].set_xlabel('Number of Nodes', fontsize=fontsize)
axes[0].tick_params(axis='y', labelsize=labelsize)
axes[0].tick_params(axis='x', labelsize=labelsize)

sns.boxplot(x='n', y='lambda_rel', data=data.query('k == 5').query('t == 10'),
            color=color, ax=axes[1])
axes[1].set(yscale='log')
axes[1].set_ylabel(
    'Relative Error of Homophily Coefficients', fontsize=fontsize)
axes[1].set_xlabel('Number of Nodes', fontsize=fontsize)
axes[1].tick_params(axis='y', labelsize=labelsize)
axes[1].tick_params(axis='x', labelsize=labelsize)

sns.boxplot(x='n', y='delta_rel', data=data.query('k == 5').query('t == 10'),
            color=color, ax=axes[2])
axes[2].set(yscale='log')
axes[2].set_ylabel('Relative Error of Sociality Effects', fontsize=fontsize)
axes[2].set_xlabel('Number of Nodes', fontsize=fontsize)
axes[2].tick_params(axis='y', labelsize=labelsize)
axes[2].tick_params(axis='x', labelsize=labelsize)

sns.boxplot(x='n', y='proba_rel', data=data.query('k == 5').query('t == 10'),
            color=color, ax=axes[3])
axes[3].set(yscale='log')
axes[3].set_ylabel(
    'Relative Error of Connection Probabilities', fontsize=fontsize)
axes[3].set_xlabel('Number of Nodes', fontsize=fontsize)
axes[3].tick_params(axis='y', labelsize=labelsize)
axes[3].tick_params(axis='x', labelsize=labelsize)

fig.savefig('relative_error_nodes.{}'.format(ext), dpi=dpi, bbox_inches='tight')


fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

sns.boxplot(x='n', y='insample_auc', data=data.query('k == 5').query('t == 10'),
            color=color, ax=axes[0])
axes[0].set_ylabel('In-Sample AUC', fontsize=fontsize)
axes[0].set_xlabel('Number of Nodes', fontsize=fontsize)

sns.boxplot(x='n', y='holdhout_auc', data=data.query('k == 5').query('t == 10'),
            color=color, ax=axes[1])
axes[1].set_ylabel('Held-Out AUC', fontsize=fontsize)
axes[1].set_xlabel('Number of Nodes', fontsize=fontsize)

fig.savefig('perf_nodes.{}'.format(ext), dpi=dpi)

# AUC plot
data = data[['insample_auc', 'holdhout_auc', 't', 'n', 'k']]
data = pd.melt(
    data, value_vars=['insample_auc', 'holdhout_auc'], id_vars=['t', 'n', 'k'])

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

sns.boxplot(x='k', y='value', data=data.query('t == 10').query('n == 100'),
            hue='variable', ax=axes[1])
axes[1].set_xlabel('Number of Layers', fontsize=fontsize)
axes[1].set_title('T = 10, n = 100', fontsize=fontsize)
axes[1].set_ylabel('', fontsize=fontsize)
axes[1].get_legend().remove()
axes[1].tick_params(axis='y', labelsize=labelsize)
axes[1].tick_params(axis='x', labelsize=labelsize)

handles, _ = axes[1].get_legend_handles_labels()
axes[1].legend(handles, ["In-Sample", "Holdout"], bbox_to_anchor=(0.5, 1.2),
               loc='upper center', ncol=2, fontsize=fontsize)

sns.boxplot(x='t', y='value', data=data.query('k == 5').query('n == 100'),
            hue='variable', ax=axes[2])
axes[2].set_ylabel('', fontsize=fontsize)
axes[2].set_xlabel('Number of Time Steps', fontsize=fontsize)
axes[2].set_title('K = 5, n = 100', fontsize=fontsize)
axes[2].get_legend().remove()
axes[2].tick_params(axis='y', labelsize=labelsize)
axes[2].tick_params(axis='x', labelsize=labelsize)


sns.boxplot(x='n', y='value', data=data.query('k == 5').query('t == 10'),
            hue='variable', ax=axes[0])
axes[0].set_ylabel('AUC', fontsize=fontsize)
axes[0].set_xlabel('Number of Nodes', fontsize=fontsize)
axes[0].set_title('K = 5, T = 10', fontsize=fontsize)
axes[0].get_legend().remove()
axes[0].tick_params(axis='y', labelsize=labelsize)
axes[0].tick_params(axis='x', labelsize=labelsize)

# reserve 20% top space for legend
plt.subplots_adjust(top=0.8)

fig.savefig('perf_auc.pdf', dpi=dpi, bbox_inches='tight')
