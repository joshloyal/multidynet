import os
import numpy as np
import pandas as pd

from os.path import join
from joblib import Parallel, delayed

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import correlated_dynamic_multilayer_network


def ic_selection(Y, n_features, random_state=42):
    model = DynamicMultilayerNetworkLSM(
            max_iter=1000,
            n_features=n_features,
            lambda_var_prior=4,
            lambda_odds_prior=1,
            tol=1e-2, init_type='both',
            n_init=4, n_jobs=-1,
            random_state=123 * (random_state + 1)).fit(Y, n_samples=500)

    return model.aic_, model.bic_, model.dic(Y)[0], model.waic(Y)[0]


def sim_single(seed, sigma):

    Y, X, lmbda, delta, probas, dists, z = correlated_dynamic_multilayer_network(
        n_nodes=100, n_layers=5, n_time_steps=10, n_features=2,
        tau=1.0, center=[0.75, 0.75], rho=0.5, rho_t=0.4, sigma=sigma,
        random_state=seed)

    res = []
    for d in range(1, 6):
        res.append(ic_selection(Y, d))
    res = np.asarray(res)

    out_file = f'results_{seed}.csv'
    dir_base = 'output_dimension_selection'
    dir_name = join(dir_base, f'sigma_{sigma}')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data = pd.DataFrame(np.argmin(res, axis=0) + 1).T
    data.columns = ['aic', 'bic', 'dic', 'waic']
    data.to_csv(join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50
for sigma in [0.01, 0.05, 0.1, 0.2, 0.3]:
    for i in range(n_reps):
        sim_single(seed=i, sigma=sigma)
