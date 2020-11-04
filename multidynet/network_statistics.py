import numpy as np


def calculate_densities(Y):
    n_layers, n_time_steps, _, _ = Y.shape
    indices = np.tril_indices_from(Y[0, 0])

    density = np.zeros(n_layers)
    for k in range(n_layers):
        num, dem = 0., 0.
        for t in range(n_time_steps):
            y_vec = Y[k, t][indices]
            non_miss = y_vec != -1
            num += y_vec[non_miss].sum()
            dem += np.sum(non_miss)
        density[k] = num / dem

    return density


def calculate_average_degree(Y):
    n_layers, n_time_steps, n_nodes, _ = Y.shape

    degrees = np.zeros((n_layers, n_nodes))
    for k in range(n_layers):
        for t in range(n_time_steps):
            y = Y[k, t].copy()
            y[y == -1] = 0.
            degrees[k] += y.sum(axis=1)
        degrees[k] /= n_time_steps

    return degrees
