import numpy as np

from os.path import dirname, join


__all__ = ['load_travian']


def load_travian():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    n_layers = 3
    n_time_steps = 30
    n_nodes = 2392
    Y = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))

    file_fmt = join('travian', 'travian_{}_{}.npy')
    for k in range(n_layers):
        for t in range(n_time_steps):
            Y[k, t] = np.load(join(file_path, file_fmt.format(k, t)))

    return np.ascontiguousarray(Y)
