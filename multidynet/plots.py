import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from matplotlib.colors import ListedColormap
from dynetlsm.plots import get_colors


__all__ = ['plot_network', 'plot_network_communities']


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


def plot_lambda(lmbda):
    n_layers, n_features = lmbda.shape
