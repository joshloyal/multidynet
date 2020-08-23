from os.path import join

import joblib
import numpy as np

n_nodes = 242
n_layers = 2
n_time_steps = 16

Y = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))

file_fmt = join('primaryschool_{}_{}.npy')
for k in range(n_layers):
    for t in range(n_time_steps):
        Y[k, t] = np.loadtxt(join('.', file_fmt.format(k+1, t+1)))


joblib.dump(Y, join('.', 'numpy_data', 'primaryschool30.gz'), compress=('gzip', 3))
