import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib.colors import ListedColormap
from scipy.stats import norm
from dynetlsm.plots import get_colors


__all__ = ['plot_network', 'plot_network_communities',
           'plot_sociability', 'plot_lambda', 'plot_node_trajectories']


def plot_network(Y, X, normalize=True, figsize=(8, 6), node_color='orangered',
                 alpha=1.0, size=300, edge_width=0.25, node_labels=None,
                 font_size=12, with_labels=False):
    fig, ax = plt.subplots(figsize=figsize)

    r = np.sqrt((X ** 2).sum(axis=1)).reshape(-1, 1)
    if normalize:
        X = X / r

    cmap = ListedColormap(
        sns.light_palette(node_color, n_colors=np.unique(r).shape[0]))
    G = nx.from_numpy_array(Y)
    if node_labels is not None:
        labels = {node_id : label for node_id, label in enumerate(node_labels)}
    else:
        labels = None

    nx.draw_networkx(G, X, edge_color='gray', width=edge_width,
                     node_color=r.ravel() / r.min(),
                     node_size=size,
                     alpha=alpha,
                     cmap=cmap,
                     labels=labels,
                     font_size=font_size,
                     with_labels=with_labels,
                     ax=ax)
    ax.collections[0].set_edgecolor('white')
    ax.axis('equal')
    ax.axis('off')

    return ax


def plot_network_communities(Y, X, z, normalize=True, figsize=(8, 6),
                             alpha=1.0, size=300, edge_width=0.25,
                             with_labels=False):
    fig, ax = plt.subplots(figsize=figsize)

    r = np.sqrt((X ** 2).sum(axis=1)).reshape(-1, 1)
    if normalize:
        X = X / r

    colors = get_colors(z.ravel())

    G = nx.from_numpy_array(Y)
    nx.draw_networkx(G, X, edge_color='gray', width=edge_width,
                     node_color=colors[z],
                     node_size=size,
                     alpha=alpha,
                     with_labels=with_labels,
                     ax=ax)
    ax.collections[0].set_edgecolor('white')
    ax.axis('equal')
    ax.axis('off')

    return ax


def plot_sociability(model, k=0, node_labels=None, color='gray', ax=None,
                     figsize=(10, 12)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if node_labels is None:
        node_labels = [str(i + 1) for i in range(model.X_.shape[1])]
    node_labels = np.asarray(node_labels)

    order = np.argsort(model.delta_[k])
    log_odds = np.exp(model.delta_[k][order])

    y_pos = np.arange(node_labels.shape[0])
    ax.barh(y_pos, log_odds, align='center', color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(node_labels[order])
    ax.set_xlabel('log-odds [$\exp(delta_k^i)]$')
    ax.set_title('k = {}'.format(k))

    return ax


def plot_node_trajectories(model, node_list, q_alpha=0.8, node_labels=None,
                           nrows=None, ncols=1, alpha=0.2, figsize=(10, 8)):

    if nrows is None:
        nrows = model.X_.shape[2]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    ax = axes.flat

    if node_labels is None:
        node_labels = [i for i in range(model.X_.shape[1])]
    node_labels = np.asarray(node_labels)

    n_time_steps, n_nodes, n_features = model.X_.shape
    z_alpha = norm.ppf(q_alpha)
    ts = np.arange(1, n_time_steps + 1)
    for node_label in node_list:
        node_id = np.where(node_labels == node_label)[0].item()
        x_upp = np.zeros(n_time_steps)
        x_low = np.zeros(n_time_steps)
        for p in range(n_features):
            ax[p].plot(ts, model.X_[:, node_id, p], 'o--',
                       label=node_labels[node_id])
            for t in range(n_time_steps):
                se = z_alpha * np.sqrt(model.X_sigma_[t, node_id, p, p])
                x_upp[t] = model.X_[t, node_id, p] + se
                x_low[t] = model.X_[t, node_id, p] - se
            ax[p].fill_between(ts, x_low, x_upp, alpha=alpha)

    # accomodate legends and title
    ax[0].legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax[-1].set_xlabel('t')
    for p in range(n_features):
        ax[p].set_title('p = {}'.format(p))
        ax[p].hlines(0, 1, n_time_steps, lw=2, linestyles='dotted')
        ax[p].set_ylabel('Latent Position [p = {}]'.format(p))

    plt.subplots_adjust(right=0.7)

    return ax


def plot_lambda(model, palette='muted', height=8, y='latent dimension'):
    n_layers, n_features = model.lambda_.shape

    data = pd.DataFrame(model.lambda_,
        columns=['p = {}'.format(p + 1) for p in range(n_features)])
    data['layer'] = ['k = {}'.format(k + 1) for k in range(n_layers)]

    data = pd.melt(data, id_vars=['layer'], var_name=['latent dimension'],
                   value_name='$\lambda$')

    sns.set_style('whitegrid')
    if y == 'latent dimension':
        g = sns.catplot(x='$\lambda$', y='latent dimension', hue='layer',
                        data=data, kind='bar', palette=palette, height=height)
    else:
        g = sns.catplot(x='$\lambda$', y='layer', hue='latent dimension',
                        data=data, kind='bar', palette=palette, height=height)
    g.despine(left=True)
    sns.set_style('white')

    return g
