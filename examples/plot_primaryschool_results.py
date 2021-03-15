import os

import joblib
import numpy as np
import matplotlib.pyplot as plt

from multidynet.datasets import load_primaryschool
from multidynet.plots import (
    plot_social_trajectories,
    plot_lambda,
    plot_network,
    plot_node_trajectories,
    plot_pairwise_probabilities,
    plot_network_statistics,
    get_colors)


plt.rc('font', family='serif')
out_dir = 'primaryschool_results'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


Y, grade, gender, layer_labels, time_labels = load_primaryschool()

model = joblib.load(open(os.path.join('models', 'school_model.pkl'), 'rb'))
print('in-sample AUC: {:2f}'.format(model.auc_))


# plot social trajectories
for k in range(Y.shape[0]):
    fig, ax = plot_social_trajectories(
        model, k=k,
        node_list=['Actor 5', 'Actor 195', 'Actor 148'],
        ref_value=None, label_offset=0,
        q_alpha=0.05, alpha=0.05, fill_alpha=0.2,
        line_width=2,
        node_colors=['darkorange', 'tomato', 'steelblue', 'forestgreen'],
        layer_label=layer_labels[k], xlabel='Year',
        node_labels=['Actor {}'.format(i + 1) for i in range(Y.shape[2])],
        figsize=(12, 8), fontsize=18)

    ax.set_xticks([i for i in range(0, 24, 3)])
    ax.set_xticklabels(time_labels[::3], rotation=45, fontsize=16)
    ax.margins(x=0.025)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(axis='x')

    file_name = os.path.join(
        out_dir, 'social_trajectories_{}.pdf'.format(layer_labels[k]))
    fig.savefig(file_name, dpi=100, bbox_inches='tight')


# homophily coefficients
fig, axes = plot_lambda(
    model, q_alpha=0.05, layer_labels=layer_labels, fontsize=16,
    figsize=(13, 6), include_gridlines=False)

file_name = os.path.join(
    out_dir, 'homophily_coefficients.pdf'.format(layer_labels[k]))
fig.savefig(file_name, dpi=100, bbox_inches='tight')


# latent space
t = 5
colors = get_colors(np.arange(10))
colors[1], colors[2] = colors[2], colors[1]
colors[10] = 'black'

fig, ax = plt.subplots(figsize=(22, 10), ncols=2, sharey=True)

k = 0
plot_network(
    Y[k, t], model.Z_[t], X_sigma=model.Z_sigma_[t],
    tau_sq=model.tau_sq_, normalize=False,
    z=grade, colors=colors,
    size=50, with_labels=True, font_size=12,
    edge_width=0.3, figsize=(10, 10), legend_fontsize=16,
    ax=ax[0])
ax[0].set_title('{} from {}'.format(layer_labels[k], time_labels[t]), fontsize=24)

k = 1
plot_network(
    Y[k, t], model.Z_[t], X_sigma=model.Z_sigma_[t],
    tau_sq=model.tau_sq_, normalize=False,
    z=grade, colors=colors,
    size=50, with_labels=True, font_size=12,
    edge_width=0.3, figsize=(10, 10), legend_fontsize=16,
    ax=ax[1])
ax[1].set_title(
    '{} from {}'.format(layer_labels[k], time_labels[t]), fontsize=24)
fig.tight_layout()

file_name = os.path.join(
    out_dir, 'latent_space.pdf'.format(layer_labels[k]))
fig.savefig(file_name, dpi=100, bbox_inches='tight')


# branching factor plots
stat_sim = joblib.load(
    open(os.path.join('models', 'stat_sim.pkl'), 'rb'))
stat_obs = joblib.load(
    open(os.path.join('models', 'stat_obs.pkl'), 'rb'))

fig, ax = plot_network_statistics(
    stat_sim[0], stat_obs[0],
    stat_label='Epidemic Branching Factor',
    time_labels=time_labels, layer_labels=layer_labels,
    time_step=3, figsize=(20, 10), xlabel='Time')
file_name = os.path.join(
    out_dir, 'branching_factor.pdf'.format(layer_labels[k]))
fig.savefig(file_name, dpi=100, bbox_inches='tight')
