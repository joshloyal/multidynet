import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


__all__ = ['plot_network']


def plot_network(Y, X, normalize=True, figsize=(8, 6), node_color='tomato',
                 alpha=1.0, size=100, edge_width=0.25):
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        r = np.sqrt((X ** 2).sum(axis=1)).reshape(-1, 1)
        X = X / r
        sizes = size * (r / np.min(r))
    else:
        sizes = size


    G = nx.from_numpy_array(Y)
    nx.draw_networkx(G, X, edge_color='gray', width=edge_width,
                     node_color=node_color, node_size=sizes, alpha=alpha,
                     ax=ax)
    ax.collections[0].set_edgecolor('white')
    ax.axis('equal')
    ax.axis('off')

    return ax
