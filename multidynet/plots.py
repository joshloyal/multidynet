import numbers
import tempfile

import imageio
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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from dynetlsm.plots import get_colors


__all__ = ['plot_network', 'make_network_animation',
           'plot_sociability', 'plot_lambda', 'plot_node_trajectories',
           'plot_pairwise_probabilities']


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


def plot_network(Y, X, X_sigma=None,
                 z=None, tau_sq=None, normalize=True, figsize=(8, 6),
                 node_color='orangered', color_distance=False,
                 alpha=1.0, contour_alpha=0.25,
                 size=300, edge_width=0.25, node_labels=None,
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

    if z is None:
        if color_distance:
            node_color = r.ravel() / r.min()
        else:
            node_color = np.asarray([node_color] * X.shape[0])
    else:
        encoder = LabelEncoder().fit(z)
        colors = get_colors(z.ravel())
        node_color = colors[encoder.transform(z)]

        # add a legend
        for i in range(encoder.classes_.shape[0]):
            ax.plot([0], [0], 'o', c=colors[i], label=encoder.classes_[i],
                    markeredgecolor='w', zorder=0)
        ax.plot([0], [0], 'o', markeredgecolor='w', c='w', zorder=0)

    # draw latent position credible interval ellipses
    if X_sigma is not None:
        for i in range(X.shape[0]):
            normal_contour(X[i], X_sigma[i], edgecolor='gray',
                           facecolor=node_color[i] if z is not None else 'gray',
                           alpha=contour_alpha, ax=ax, n_std=[2])

    nx.draw_networkx(G, X, edge_color='gray', width=edge_width,
                     node_color=node_color,
                     node_size=size,
                     alpha=alpha,
                     cmap=cmap,
                     labels=labels,
                     font_size=font_size,
                     with_labels=with_labels,
                     ax=ax)

    if X_sigma is not None:
        ax.collections[0].set_edgecolor(None)
    else:
        ax.collections[0].set_edgecolor('white')

    ax.axis('equal')
    ax.axis('off')

    # draw normal contour if available
    if tau_sq is not None:
        # draw center of latent space
        ax.scatter(0, 0, color='k', marker='+', s=200)

        # draw two standard deviation contour
        normal_contour([0, 0], tau_sq * np.eye(X.shape[1]), n_std=[1],
                       linestyle='--', edgecolor='k',
                       facecolor='none', zorder=1, ax=ax)

    if z is not None:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6,
                  fontsize=12)

    return fig, ax


def make_network_animation(filename, Y, X, X_sigma=None,
                           k=0, z=None, tau_sq=None, normalize=True,
                           figsize=(8, 6), node_color='orangered',
                           alpha=1.0, contour_alpha=0.25,
                           size=300, edge_width=0.25,
                           node_labels=None, font_size=12, with_labels=False,
                           layer_labels=None, time_labels=None,
                           title_fmt='{}, {}', border=0.5, duration=1):

    # XXX: hack to shut off plotting within a jupyter notebook...
    plt.ioff()

    n_layers, n_time_steps, _, _ = Y.shape

    if layer_labels is None:
        layer_labels =  ["k = {}".format(k) for k in range(n_layers)]

    if time_labels is None:
        time_labels = ["t = {}".format(t) for t in range(n_time_steps)]

    with tempfile.TemporaryDirectory() as tempdir:
        x_max, y_max = X.max(axis=(0, 1))
        x_min, y_min = X.min(axis=(0, 1))

        pngs = []
        for t in range(Y.shape[1]):
            fig, ax = plot_network(Y[k, t], X[t],
                X_sigma=X_sigma[t] if X_sigma is not None else None,
                z=z, tau_sq=tau_sq,
                normalize=normalize, figsize=figsize, node_color=node_color,
                alpha=alpha, contour_alpha=contour_alpha,
                size=size, edge_width=edge_width,
                node_labels=node_labels, font_size=font_size,
                with_labels=with_labels,)
            ax.set_title(title_fmt.format(layer_labels[k], time_labels[t]))
            ax.set_xlim(x_min - border, x_max + border)
            ax.set_ylim(y_min - border, y_max + border)

            fname = tempfile.TemporaryFile(dir=tempdir, suffix='.png')
            fig.savefig(fname, dpi=100)
            fname.seek(0)
            plt.close(fig)  # necessary to free memory
            pngs.append(fname)

        images = []
        for png in pngs:
            images.append(imageio.imread(png))
        imageio.mimsave(filename, images, duration=duration)

    plt.ion()


def plot_static_sociability(model, k=0, node_labels=None, layer_label=None,
                            ax=None, figsize=(10, 12), color_code=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

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

    return fig, ax


def plot_social_trajectories(
                     model, k=0, q_alpha=0.05, node_list=None, node_colors=None,
                     node_labels=None, layer_label=None, plot_hline=True,
                     xlabel='Time', alpha=0.15, fill_alpha=0.2, line_width=3,
                     ax=None, figsize=(10, 6), label_offset=1, fontsize=12,
                     color_code=False):

    n_layers, n_time_steps, n_nodes = model.delta_.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    if node_labels is None:
        node_labels = [str(i + 1) for i in range(model.delta_.shape[2])]
    node_labels = np.asarray(node_labels)


    for i in range(n_nodes):
        ax.plot(model.delta_[k, :, i].T, 'k-', alpha=alpha)

    if node_list is not None:
        node_list = np.asarray(node_list)
        if node_colors is None:
            node_colors = get_colors(np.arange(len(node_list)))

        for i, node_label in enumerate(node_list):
            node_id = np.where(node_labels == node_label)[0].item()
            ax.plot(model.delta_[k, :, node_id].T, '-',
                    lw=line_width, c=node_colors[i])
            ax.annotate(node_label,
                        xy=(n_time_steps + label_offset,
                            model.delta_[k, -1, node_id]),
                        color=node_colors[i], fontsize=fontsize)

            if q_alpha is not None:
                x_upp = np.zeros(n_time_steps)
                x_low = np.zeros(n_time_steps)
                z_alpha = norm.ppf(1 - q_alpha / 2.)
                ts = np.arange(n_time_steps)
                for t in range(n_time_steps):
                    se = z_alpha * np.sqrt(model.delta_sigma_[k, t, node_id])
                    x_upp[t] = model.delta_[k, t, node_id] + se
                    x_low[t] = model.delta_[k, t, node_id] - se
                ax.fill_between(
                    ts, x_low, x_upp, alpha=fill_alpha, color=node_colors[i])

    if plot_hline:
        ax.hlines(0, 1, n_time_steps, lw=2, linestyles='--', color='k')

    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # axis-labels
    ax.set_ylabel('Sociality', fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)

    if layer_label is not None:
        ax.set_title(layer_label, fontsize=fontsize)
    else:
        ax.set_title('k = {}'.format(k), fontsize=fontsize)


    return fig, ax


def plot_node_trajectories(model, node_list, q_alpha=0.05, node_labels=None,
                           node_colors=None, nrows=None, ncols=1, alpha=0.2,
                           linestyle='o--', fontsize=12,
                           figsize=(10, 8)):

    if nrows is None:
        nrows = model.X_.shape[2]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    ax = axes.flat

    if node_labels is None:
        node_labels = [i for i in range(model.X_.shape[1])]
    node_labels = np.asarray(node_labels)

    if node_colors is None:
        node_colors = get_colors(np.arange(len(node_list)))

    n_time_steps, n_nodes, n_features = model.X_.shape
    z_alpha = norm.ppf(1 - q_alpha / 2.)
    ts = np.arange(n_time_steps)
    for i, node_label in enumerate(node_list):
        node_id = np.where(node_labels == node_label)[0].item()
        x_upp = np.zeros(n_time_steps)
        x_low = np.zeros(n_time_steps)
        for p in range(n_features):
            ax[p].plot(ts, model.X_[:, node_id, p], linestyle,
                       label=node_labels[node_id], c=node_colors[i])
            for t in range(n_time_steps):
                se = z_alpha * np.sqrt(model.X_sigma_[t, node_id, p, p])
                x_upp[t] = model.X_[t, node_id, p] + se
                x_low[t] = model.X_[t, node_id, p] - se
            ax[p].fill_between(
                ts, x_low, x_upp, alpha=alpha, color=node_colors[i])

    # accomodate legends and title
    ax[0].legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=fontsize)
    ax[-1].set_xlabel('t')
    for p in range(n_features):
        #ax[p].set_title('p = {}'.format(p + 1), fontsize=fontsize)
        ax[p].hlines(0, 1, n_time_steps, lw=2, linestyles='dotted', color='k')
        ax[p].set_ylabel('Latent Position [p = {}]'.format(p + 1),
                         fontsize=fontsize)

    plt.subplots_adjust(right=0.7)

    return fig, ax


def sample_link_probability(model, k, t, i, j, n_reps=1000, random_state=123):
    rng = check_random_state(random_state)

    deltai = rng.normal(
        loc=model.delta_[k, t, i], scale=np.sqrt(model.delta_sigma_[k, t, i]),
        size=n_reps)
    deltaj = rng.normal(
        loc=model.delta_[k, t, j], scale=np.sqrt(model.delta_sigma_[k, t, j]),
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


def forecast_link_probability(model, k, i, j, horizon=1, n_reps=1000,
                              random_state=123):
    rng = check_random_state(random_state)
    n_features = model.X_.shape[-1]


    if k == 0:
        lmbdak = np.zeros((n_reps, model.lambda_.shape[1]))
        for p in range(model.lambda_.shape[1]):
            lmbdak[:, p] = (
                2 * rng.binomial(1, model.lambda_proba_[p], size=n_reps) - 1)
    else:
        lmbdak = rng.multivariate_normal(
            model.lambda_[k], model.lambda_sigma_[k], size=n_reps)

    deltai = rng.normal(
        loc=model.delta_[k, -1, i], scale=np.sqrt(model.delta_sigma_[k, -1, i]),
        size=n_reps)
    deltaj = rng.normal(
        loc=model.delta_[k, -1, j], scale=np.sqrt(model.delta_sigma_[k, -1, j]),
        size=n_reps)

    Xi = rng.multivariate_normal(model.X_[-1, i], model.X_sigma_[-1, i],
                                 size=n_reps)
    Xj = rng.multivariate_normal(model.X_[-1, j], model.X_sigma_[-1, j],
                                 size=n_reps)

    pis = np.zeros((horizon, n_reps))
    for h in range(horizon):
        deltai = deltai + rng.normal(
            loc=0, scale=np.sqrt(model.sigma_sq_delta_), size=n_reps)
        deltaj = deltaj + rng.normal(
            loc=0, scale=np.sqrt(model.sigma_sq_delta_), size=n_reps)

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
                                layer_labels=None, q_alpha=0.05, n_reps=1000,
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
                pi_low[t] = np.quantile(pis, q=q_alpha / 2.)
                pi_upp[t] = np.quantile(pis, q=1 - q_alpha / 2.)

            if horizon > 0:
                pis = forecast_link_probability(
                    model, k, i, j, horizon=horizon, n_reps=n_reps,
                    random_state=random_state)

                for h in range(horizon):
                    pi_mean[n_time_steps + h] = pis[h].mean()
                    pi_low[n_time_steps + h] = (
                        np.quantile(pis[h], q=q_alpha / 2.))
                    pi_upp[n_time_steps + h] = (
                        np.quantile(pis[h], q=1 - q_alpha / 2.))

            ax.plot(ts, pi_mean, linestyle, label=label)
            ax.fill_between(ts, pi_low, pi_upp, alpha=alpha)

    # accomodate legends and title
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_xlabel('t')
    ax.set_ylabel('Link Probability ({} - {})'.format(node_i, node_j))

    return fig, ax


def plot_lambda(model, q_alpha=0.05, layer_labels=None, height=0.5,
                fontsize=12,
                figsize=(12, 6), include_gridlines=False):
    n_layers, n_features = model.lambda_.shape

    if layer_labels is None:
        layer_labels = ['k = {}'.format(k + 1) for k in range(n_layers)]

    if include_gridlines:
        sns.set_style('whitegrid')

    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    colors = [to_hex(c) for c in sns.color_palette(
              'muted', n_colors=n_layers, desat=0.75)]

    z_alpha = norm.ppf(1 - q_alpha / 2.)
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
                txt = '{:.3f} ({:.3f}, {:.3f})'.format(
                    lmbda, lmbda - xerr[k], lmbda + xerr[k])
            ax.text(lmbda, k - 0.1, txt, horizontalalignment=align)

        ax.set_yticks(np.arange(n_layers))
        ax.set_yticklabels(layer_labels, fontsize=fontsize)
        ax.invert_yaxis()
        ax.set_title('p = {}'.format(p + 1), fontsize=fontsize)

        axes.flat[-1].set_xlabel('Homophily Parameter ($\lambda_{kp}$)',
                                 fontsize=fontsize)

    x_max = max([ax.get_xlim()[1] for ax in axes.flat])
    for ax in axes.flat:
        if np.all(model.lambda_ >= 0):
            ax.set_xlim(0, x_max)
        else:
            ax.vlines(0, 0, n_layers - 1, linestyles='dotted', color='k')
        sns.despine(ax=ax, bottom=True)

    sns.set_style('white')

    return fig, axes


def plot_network_statistics(stat_sim, stat_obs=None, nrow=1, ncol=None,
                            time_labels=None, stat_label='Statistic',
                            time_step=1,
                            layer_labels=None, figsize=(16, 10),
                            xlabel='Time'):
    n_layers, n_time_steps, _ = stat_sim.shape

    if ncol is None:
        ncol = n_layers
    fig, axes = plt.subplots(nrow, ncol, sharey=True, figsize=figsize)

    if time_labels is None:
        time_labels = np.arange(n_time_steps) + 1

    if layer_labels is None:
        layer_labels = np.arange(n_layers) + 1

    for k, ax in enumerate(axes.flat):
        data = pd.DataFrame()
        for t in range(n_time_steps):
            data[time_labels[t]] = stat_sim[k, t]

        if stat_obs is not None:
            ax.plot(np.arange(n_time_steps), stat_obs[k], 'o--', c='k')
        sns.boxplot(x='variable', y='value', data=pd.melt(data),
                    ax=ax, color='white')

        ax.set_xticklabels(time_labels[::time_step], rotation=45, fontsize=12)
        plt.setp(ax.artists, edgecolor='black')
        plt.setp(ax.lines, color='black')

        ax.set_xticks([i for i in range(0, n_time_steps, time_step)])
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(axis='x')
        ax.set_title(layer_labels[k], fontsize=24)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(stat_label, fontsize=16)

    return fig, axes
