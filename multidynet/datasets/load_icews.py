import numpy as np

from os.path import dirname, join


__all__ = ['load_icews']


def load_icews():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    n_nodes = 65
    n_layers = 4
    n_time_steps = 96
    Y = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))

    file_fmt = join('icews', 'icews_{}_{}.npy')
    for k in range(n_layers):
        for t in range(n_time_steps):
            Y[k, t] = np.loadtxt(join(file_path, file_fmt.format(k+1, t+1)))

    countries = np.loadtxt(
        join(file_path, 'icews', 'icews_countries.txt'),
        delimiter='\n', dtype=np.unicode)

    return np.ascontiguousarray(Y), countries
