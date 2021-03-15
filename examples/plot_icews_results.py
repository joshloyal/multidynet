import os

import joblib
import numpy as np
import matplotlib.pyplot as plt

from multidynet.datasets import load_icews
from multidynet.plots import (
    plot_social_trajectories,
    plot_lambda,
    plot_network,
    plot_node_trajectories,
    plot_pairwise_probabilities)


plt.rc('font', family='serif')
out_dir = 'icews_results'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


Y, countries, layer_labels, time_labels = load_icews(
    dataset='large', country_names='full')


model = joblib.load(open(os.path.join('models', 'icews_model.pkl'), 'rb'))
print('in-sample AUC: {:2f}'.format(model.auc_))


# plot social trajectories
for k in range(Y.shape[0]):
    fig, ax = plot_social_trajectories(
        model, k=k, ref_value=None,
        node_list=['United States', 'Russia', 'Libya', 'Iraq', 'Ukraine',
                   'Syria'],
        node_colors=['darkorange', 'tomato', 'purple', 'forestgreen',
                     'steelblue', 'brown'],
        node_labels=countries, layer_label=layer_labels[k], xlabel='Year',
        line_width=2, label_offset=2, alpha=0.2, fontsize=16,
        figsize=(12, 6))

    ax.set_xticks([i * 12 for i in range(9)])
    ax.set_xticklabels([2009 + i for i in range(9)], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.margins(x=0.025)
    ax.grid(axis='x')
    fig.subplots_adjust(right=0.85)

    file_name = os.path.join(
        out_dir, 'social_trajectories_{}.pdf'.format(layer_labels[k]))
    fig.savefig(file_name, dpi=100, bbox_inches='tight')


# plot homophily coefficients
fig, axes = plot_lambda(model, q_alpha=0.05, layer_labels=layer_labels, fontsize=16,
                        figsize=(14, 6), include_gridlines=False)
fig.subplots_adjust(left=0.2)
file_name = os.path.join(out_dir, 'homophily_coefficients.pdf')
fig.savefig(file_name, dpi=100, bbox_inches='tight')


# latent space for Feb 2012 and Feb 2014

fig, ax= plt.subplots(
    figsize=(28, 15), ncols=2, sharey=True, constrained_layout=True)
ax[0].autoscale(enable=True)

# abbreviate DRC
countries_old = countries.copy()
countries[
    np.where(countries_old == 'Democratic Republic of Congo')[0]] = 'Congo'

node_colors = ['gray' for c in countries]
node_colors[int(np.where(countries == 'Ukraine')[0])] = 'steelblue'
node_colors[int(np.where(countries == 'Russia')[0])] = 'red'
node_colors = np.asarray(node_colors)

calpha = np.asarray(
    [0.5 if n == 'Ukraine' or n == 'Russia' else 0.1 for n in countries])

t = 37
k = 0
plot_network(Y[k, t], model.Z_[t], X_sigma=model.Z_sigma_[t],
            tau_sq=model.tau_sq_, normalize=False,
            with_labels=True, font_size=20,
            node_labels=countries, contour_alpha=calpha,
            node_color=node_colors, size=50, edge_width=0, ax=ax[0])
ax[0].set_title('{}'.format(time_labels[t]), fontsize=32)

t = 61
plot_network(Y[k, t], model.Z_[t], X_sigma=model.Z_sigma_[t],
             tau_sq=model.tau_sq_, normalize=False,
             with_labels=True, font_size=20,
             node_labels=countries, contour_alpha=calpha,
             node_color=node_colors, size=50, edge_width=0, ax=ax[1])
ax[1].set_title('{}'.format(time_labels[t]), fontsize=32)

file_name = os.path.join(out_dir, 'latent_space.pdf')
fig.savefig(file_name, dpi=100, bbox_inches='tight')


# latent trajectories for Ukraine and Russia

fig, ax = plot_node_trajectories(
    model, ['Ukraine', 'Russia'],
    node_colors=['steelblue', 'red', 'goldenrod', 'forestgreen'],
    figsize=(10, 6), fontsize=12,
    node_labels=countries, q_alpha=0.05, linestyle='--')

for p in range(2):
    ax[p].set_xticks([i * 12 for i in range(9)])
    ax[p].set_xticklabels([2009 + i for i in range(9)])
    ax[p].grid(axis='x')
    ax[-1].set_xlabel('Year', fontsize=12)

file_name = os.path.join(out_dir, 'ukraine_russia_trajectories.pdf')
fig.savefig(file_name, dpi=100, bbox_inches='tight')


# connection probabilities of Ukraine and Russia
fig, axes = plt.subplots(figsize=(15,8), nrows=2, sharex=True,
                         gridspec_kw={'height_ratios': [1, 5]})

_, ax = plot_pairwise_probabilities(
    model, 'Ukraine', 'Russia',
    node_labels=countries, horizon=0,
    linestyle='.--', figsize=(10, 6),
    layer_labels=layer_labels,
    ax=axes[1])

for k in range(Y.shape[0]):
    mask = Y[k, :, 60, 8] == 1
    axes[0].plot(np.arange(Y.shape[1])[mask], Y[k, :, 60, 8][mask] - k, '.')
    axes[0].set_yticks([])

    ax.set_xticks([i * 12 for i in range(9)])
    ax.set_xticklabels([2009 + i for i in range(9)], fontsize=16)
    ax.grid(axis='x')
    ax.set_xlabel('Year', fontsize=16)

    axes[0].grid(axis='x')
    [a.set_visible(False) for a in axes[0].spines.values()]

    axes[0].set_ylabel('$Y_{ijt}^k = 1$', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    axes[0].set_xlabel('')

fig.subplots_adjust(right=0.75)
file_name = os.path.join(out_dir, 'ukraine_russia_proba.pdf')
fig.savefig(file_name, dpi=100, bbox_inches='tight')
