import os
import numpy as np
import pandas as pd
import plac

from os.path import join
from multidynet import DynamicMultilayerNetworkLSM
from multidynet.multinet.multinet import MultilayerNetworkLSM
from multidynet.datasets import correlated_dynamic_multilayer_network
from multidynet.model_selection import train_test_split
from multidynet.metrics import calculate_correlation, calculate_auc


max_iter = 1000


def sim_single(seed, sigma, rho):

    Y, X, lmbda, delta, probas, dists, z = correlated_dynamic_multilayer_network(
        n_nodes=100, n_layers=5, n_time_steps=10,
        n_features=2, tau=1.0,
        center=[0.75, 0.75],
        rho=0.5, rho_t=rho, sigma=sigma,
        random_state=seed)

    n_layers, n_time_steps, n_nodes, _ = Y.shape
    n_dyads = 0.5 * n_layers * n_time_steps * n_nodes * (n_nodes - 1)

    Y_train, test_indices = train_test_split(Y, test_size=0.2, random_state=seed)

    probas_pred = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))
    for t in range(n_time_steps):
        model = MultilayerNetworkLSM(
                    max_iter=max_iter, n_features=2,
                    lambda_var_prior=4,
                    init_covariance_type='full',
                    tol=1e-2, init_type='both',
                    n_init=4, n_jobs=-1,
                    stopping_criteria='loglik',
                    random_state=123 * (seed + 1))

        model.fit(Y_train[:, t, ...])
        probas_pred[:, t, ...] = model.probas_


    results = {}
    results['sep_cor'] = calculate_correlation(
        probas, probas_pred, test_indices=test_indices)
    results['sep_auc'] = calculate_auc(
        Y, probas_pred, test_indices=test_indices)

    model_joint = DynamicMultilayerNetworkLSM(
            lambda_var_prior=4,
            max_iter=max_iter, n_features=2,
            init_covariance_type='full',
            approx_type='structured',
            tol=1e-2, init_type='both',
            n_init=4, n_jobs=-1,
            stopping_criteria='loglik',
            random_state=123 * (seed + 1))
    model_joint.fit(Y_train)

    results['joint_cor'] = calculate_correlation(
        probas, model_joint.probas_, test_indices=test_indices)
    results['joint_auc'] = calculate_auc(
        Y, model_joint.probas_, test_indices=test_indices)

    out_file = f'results_{seed}.csv'
    dir_base = 'output_joint_separate_estimation'
    dir_name = join(dir_base, f'sigma_{sigma}_rho_{rho}')
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data = pd.DataFrame(results, index=[0])
    data.to_csv(join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50
for sigma in [0.01, 0.05, 0.1, 0.2, 0.3]:
    for rho in [0.0, 0.4, 0.8]:
        for i in range(n_reps):
            sim_single(seed=i, sigma=sigma, rho=rho)
