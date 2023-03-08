import os
import numpy as np
import pandas as pd

from os.path import join
from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import correlated_dynamic_multilayer_network
from multidynet.model_selection import train_test_split
from multidynet.callbacks import TestMetricsCallback


def sim_single(seed, sigma, rho):

    Y, X, lmbda, delta, probas, dists, z = correlated_dynamic_multilayer_network(
        n_nodes=100, n_layers=5, n_time_steps=30,
        n_features=2, tau=1.0,
        center=[0.75, 0.75], rho=0.5, rho_t=rho, sigma=sigma,
        random_state=seed)

    n_layers, n_time_steps, n_nodes, _ = Y.shape
    n_dyads = 0.5 * n_layers * n_time_steps * n_nodes * (n_nodes - 1)
    Y_train, test_indices = train_test_split(Y, test_size=0.2, random_state=seed)

    results_curves = {}
    results_metrics = {}

    model_mf = DynamicMultilayerNetworkLSM(
                    max_iter=100, n_features=2,
                    lambda_var_prior=4,
                    init_covariance_type='full',
                    approx_type='mean_field',
                    tol=1e-2, init_type='svt',
                    stopping_criteria=None,
                    random_state=123 * (seed + 1))

    callback = TestMetricsCallback(Y=Y, probas=probas, test_indices=test_indices)
    model_mf.fit(Y_train, callback=callback)

    results_metrics['mf_sigma_sq'] = model_mf.sigma_sq_
    results_metrics['mf_bias'] = model_mf.sigma_sq_ - sigma ** 2

    results_curves['mf_niter'] = np.arange(len(model_mf.callback_.times_))
    results_curves['mf_time'] =  model_mf.callback_.times_
    results_curves['mf_cor'] = model_mf.callback_.correlations_

    model_svi = DynamicMultilayerNetworkLSM(
                    max_iter=100, n_features=2,
                    lambda_var_prior=4,
                    init_covariance_type='full',
                    approx_type='structured',
                    tol=1e-2, init_type='svt',
                    stopping_criteria=None,
                    random_state=123 * (seed + 1))

    callback = TestMetricsCallback(Y=Y, probas=probas, test_indices=test_indices)
    model_svi.fit(Y_train, callback=callback)

    results_metrics['svi_sigma_sq'] = model_svi.sigma_sq_
    results_metrics['svi_bias'] = model_svi.sigma_sq_ - sigma ** 2

    results_curves['svi_niter'] = np.arange(len(model_svi.callback_.times_))
    results_curves['svi_time'] = model_svi.callback_.times_
    results_curves['svi_cor'] = model_svi.callback_.correlations_

    # make directory
    dir_base = 'output_smf_mf_comparison'
    dir_name = join(dir_base, f'sigma_{sigma}_rho_{rho}')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    out_file = f'results_metrics_{seed}.csv'
    data = pd.DataFrame(results_metrics, index=[0])
    data.to_csv(join(dir_name, out_file), index=False)

    out_file = f'results_curves_{seed}.csv'
    data = pd.DataFrame(results_curves)
    data.to_csv(join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50
for sigma in [0.01, 0.05, 0.1, 0.2, 0.3]:
    for rho in [0.0, 0.4, 0.8]:
        for i in range(n_reps):
            sim_single(seed=i, sigma=sigma, rho=rho)
