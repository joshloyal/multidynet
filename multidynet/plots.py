import numbers

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.linalg as linalg
import seaborn as sns
import pandas as pd

from matplotlib.colors import ListedColormap, to_hex
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
from scipy.stats import norm
from scipy.special import expit
from sklearn.utils import check_random_state
from dynetlsm.plots import get_colors


__all__ = ['plot_network', 'plot_network_communities',
           'plot_sociability', 'plot_lambda', 'plot_node_trajectories']


def normal_contour(mean, cov, n_std=2, ax=None, **kwargs):
    if cov.shape[0] != 2:
        raise ValueError('Only for bivariate normal densities.')

    eigenvalues, eigenvectors = linalg.eigh(cov)

    # sort the eigenvalues and eigenvectors in descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # determine the angle of rotation
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    if ax is None:
        ax = plt.gca()

    if isinstance(n_std, numbers.Integral):
        # the diameter of the ellipse is twice the square root of the evalues
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          **kwargs)
        ax.add_artist(ellipse)

        return ellipse

    ellipses = []
    for std in n_std:
        width, height = 2 * std * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          **kwargs)

        ax.add_artist(ellipse)
        ellipses.append(ellipse)

    return ellipses


def plot_network(Y, X, tau_sq=None, normalize=True, figsize=(8, 6),
                 node_color='orangered',
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

    # draw center of latent space
    ax.scatter(0, 0, color='k', marker='+', s=200)

    # draw normal contour if available
    if tau_sq is not None:
        normal_contour([0, 0], tau_sq * np.eye(X.shape[1]), n_std=[1],
                       linestyle='--', edgecolor='k',
                       facecolor='none', zorder=1, ax=ax)

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


def plot_sociability(model, k=0, node_labels=None, layer_label=None, ax=None,
                     figsize=(10, 12), color_code=False):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if node_labels is None:
        node_labels = [str(i + 1) for i in range(model.X_.shape[1])]
    node_labels = np.asarray(node_labels)

    order = np.argsort(model.delta_[k])
    odds = np.exp(model.delta_[k][order])
    y_pos = np.arange(node_labels.shape[0])

    if color_code:
        colors = ['steelblue' if odds[i] >= 1. else 'gray' for i in
                  range(len(odds))]
    else:
        colors = 'gray'
    ax.barh(y_pos, odds, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(node_labels[order])
    ax.set_xlabel('odds [$\exp(\delta_k^i)]$')

    if layer_label is not None:
        ax.set_title(layer_label)
    else:
        ax.set_title('k = {}'.format(k))

    return ax


def plot_node_trajectories(model, node_list, q_alpha=0.95, node_labels=None,
                           nrows=None, ncols=1, alpha=0.2, linestyle='o--',
                           figsize=(10, 8)):

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
            ax[p].plot(ts, model.X_[:, node_id, p], linestyle,
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
        ax[p].set_title('p = {}'.format(p + 1))
        ax[p].hlines(0, 1, n_time_steps, lw=2, linestyles='dotted')
        ax[p].set_ylabel('Latent Position [p = {}]'.format(p + 1))

    plt.subplots_adjust(right=0.7)

    return ax


def sample_link_probability(model, k, t, i, j, n_reps=1000, random_state=123):
    rng = check_random_state(random_state)

    deltai = rng.normal(
        loc=model.delta_[k, i], scale=np.sqrt(model.delta_sigma_[k, i]),
        size=n_reps)
    deltaj = rng.normal(
        loc=model.delta_[k, j], scale=np.sqrt(model.delta_sigma_[k, j]),
        size=n_reps)

    Xi = rng.multivariate_normal(model.X_[t, i], model.X_sigma_[t, i],
                                 size=n_reps)
    Xj = rng.multivariate_normal(model.X_[t, j], model.X_sigma_[t, j],
                                 size=n_reps)

    if k == 0:
        lmbdak = np.zeros((n_reps, model.lambda_.shape[1]))
        for p in range(model.lambda_.shape[1]):
            lmbdak[:, p] = (
                2 * rng.binomial(1, model.lambda_proba_[p], size=n_reps) - 1)
    else:
        lmbdak = rng.multivariate_normal(
            model.lambda_[k], model.lambda_sigma_[k], size=n_reps)

    return expit(deltai + deltaj + np.sum(lmbdak * Xi * Xj, axis=1))


def forecast_link_probability(model, k, i, j, horizon=1, n_reps=1000, random_state=123):
    rng = check_random_state(random_state)
    n_features = model.X_.shape[-1]

    deltai = rng.normal(
        loc=model.delta_[k, i], scale=np.sqrt(model.delta_sigma_[k, i]),
        size=n_reps)
    deltaj = rng.normal(
        loc=model.delta_[k, j], scale=np.sqrt(model.delta_sigma_[k, j]),
        size=n_reps)

    if k == 0:
        lmbdak = np.zeros((n_reps, model.lambda_.shape[1]))
        for p in range(model.lambda_.shape[1]):
            lmbdak[:, p] = (
                2 * rng.binomial(1, model.lambda_proba_[p], size=n_reps) - 1)
    else:
        lmbdak = rng.multivariate_normal(
            model.lambda_[k], model.lambda_sigma_[k], size=n_reps)

    Xi = rng.multivariate_normal(model.X_[-1, i], model.X_sigma_[-1, i],
                                 size=n_reps)
    Xj = rng.multivariate_normal(model.X_[-1, j], model.X_sigma_[-1, j],
                                 size=n_reps)

    pis = np.zeros((horizon, n_reps))
    for h in range(horizon):
        Xi = Xi + rng.multivariate_normal(
            np.zeros(n_features), model.sigma_sq_ * np.eye(n_features),
            size=n_reps)
        Xj = Xj + rng.multivariate_normal(
            np.zeros(n_features), model.sigma_sq_ * np.eye(n_features),
            size=n_reps)


        pis[h] = expit(deltai + deltaj + np.sum(lmbdak * Xi * Xj, axis=1))

    return pis


def plot_pairwise_probabilities(model, node_i, node_j, horizon=0,
                                node_labels=None,
                                layer_labels=None, q_alpha=0.975, n_reps=1000,
                                random_state=123, alpha=0.2, linestyle='--',
                                figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)


    if node_labels is None:
        node_labels = [i for i in range(model.X_.shape[1])]
    node_labels = np.asarray(node_labels)

    n_layers, n_time_steps, n_nodes, _ = model.probas_.shape
    ts = np.arange(n_time_steps + horizon)
    i = np.where(node_labels == node_i)[0].item()
    j = np.where(node_labels == node_j)[0].item()
    for k in range(n_layers):
        if layer_labels is None:
            label =  'k = {}'.format(k)
        else:
            label = layer_labels[k]

        if q_alpha is None:
            ax.plot(ts, model.probas_[k, :, i, j], linestyle,
                    label=label)
        else:
            pi_mean = np.zeros(n_time_steps + horizon)
            pi_low = np.zeros(n_time_steps + horizon)
            pi_upp = np.zeros(n_time_steps + horizon)
            for t in range(n_time_steps):
                pis = sample_link_probability(
                    model, k, t, i, j, n_reps=n_reps, random_state=random_state)
                pi_mean[t] = pis.mean()
                pi_low[t] = np.quantile(pis, q=1 - q_alpha)
                pi_upp[t] = np.quantile(pis, q=q_alpha)

            if horizon > 0:
                pis = forecast_link_probability(
                    model, k, i, j, horizon=horizon, n_reps=n_reps,
                    random_state=random_state)

                for h in range(horizon):
                    pi_mean[n_time_steps + h] = pis[h].mean()
                    pi_low[n_time_steps + h] = np.quantile(pis[h], q=1 - q_alpha)
                    pi_upp[n_time_steps + h] = np.quantile(pis[h], q=q_alpha)

            ax.plot(ts, pi_mean, linestyle, label=label)
            ax.fill_between(ts, pi_low, pi_upp, alpha=alpha)

    # accomodate legends and title
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_xlabel('t')
    ax.set_ylabel('Link Probability ({} - {})'.format(node_i, node_j))

    return fig, ax


def plot_lambda(model, q_alpha=0.95, layer_labels=None, height=0.5,
                figsize=(10, 8), include_gridlines=False):
    n_layers, n_features = model.lambda_.shape

    if layer_labels is None:
        layer_labels = ['k = {}'.format(k + 1) for k in range(n_layers)]

    if include_gridlines:
        sns.set_style('whitegrid')

    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    colors = [to_hex(c) for c in sns.color_palette(
              'muted', n_colors=n_layers, desat=0.75)]

    z_alpha = norm.ppf(q_alpha)
    for p, ax in enumerate(axes.flat):
        xerr = z_alpha * np.sqrt(model.lambda_sigma_[:, p, p])

        colors = ['red' if model.lambda_[k, p] > 0 else 'blue' for
                  k in range(n_layers)]
        ax.hlines(np.arange(n_layers), 0, model.lambda_[:, p], lw=1,
                  color=colors, linestyles='--')
        ax.errorbar(model.lambda_[:, p], np.arange(n_layers), fmt='o',
                    xerr=xerr, ecolor='k', capsize=5,
                    color='k', markersize=9, markeredgecolor='w')

        # add text
        for k in range(n_layers):
            align = 'right' if model.lambda_[k, p]  >= 0 else 'left'

            lmbda = model.lambda_[k, p]
            if k == 0:
                txt = '{}'.format(lmbda)
            else:
                txt = '{:.2f} ({:.2f}, {:.2f})'.format(
                    lmbda, lmbda - xerr[k], lmbda + xerr[k])
            ax.text(lmbda, k - 0.1, txt, horizontalalignment=align)

        ax.set_yticks(np.arange(n_layers))
        ax.set_yticklabels(layer_labels)
        ax.invert_yaxis()
        ax.set_title('p = {}'.format(p + 1))

        axes.flat[-1].set_xlabel('Assortativity Parameter ($\lambda_{kp}$)')

    x_max = max([ax.get_xlim()[1] for ax in axes.flat])
    for ax in axes.flat:
        if np.all(model.lambda_ >= 0):
            ax.set_xlim(0, x_max)
        else:
            ax.vlines(0, 0, n_layers - 1, linestyles='dotted')
        sns.despine(ax=ax, bottom=True)

    sns.set_style('white')

    return axes
