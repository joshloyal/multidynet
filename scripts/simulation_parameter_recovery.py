import os
import numpy as np
import pandas as pd

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import correlated_dynamic_multilayer_network
from multidynet.metrics import (
    calculate_auc, calculate_correlation,
    score_latent_space_perml, score_homophily_matrix,
    score_social_trajectories)


max_iter = 1000

def sim_single(seed, n_nodes=100, n_layers=10, n_time_steps=10):

    Y, X, lmbda, delta, probas, dists, z = correlated_dynamic_multilayer_network(
        n_nodes=n_nodes, n_layers=n_layers, n_time_steps=n_time_steps, n_features=2,
        center=[0.75, 0.75], tau=1.0, include_delta=True,
        rho=0.5, rho_t=0.4, sigma=0.05, random_state=seed)

    # fit 10 models in parallel
    model = DynamicMultilayerNetworkLSM(max_iter=max_iter, n_features=2,
                                        lambda_var_prior=4,
                                        lambda_odds_prior=1,
                                        tol=1e-2, init_type='both',
                                        n_init=4, n_jobs=-1,
                                        random_state=123 * (seed + 1))
    model.fit(Y)

    # in-sample and out-of-sample AUC estimates
    insample_auc = model.auc_
    insample_cor = calculate_correlation(probas, model.probas_)

    # compare connection probabilities
    proba_rel = np.sum((probas - model.probas_) ** 2) / np.sum(probas ** 2)

    # parameter estimates
    X_rel = score_latent_space_perml(X, model.Z_)
    dist_rel = np.sum((dists - model.dist_) ** 2) / np.sum(dists ** 2)
    lambda_rel = score_homophily_matrix(lmbda, model.lambda_)
    delta_rel = np.sum((delta - model.gamma_) ** 2) / np.sum(delta ** 2)

    # save results
    data = pd.DataFrame({
            'insample_auc' : insample_auc,
            'insample_cor' : insample_cor,
            'proba_rel': proba_rel,
            'dist_rel': dist_rel,
            'X_rel' : X_rel,
            'lambda_rel': lambda_rel,
            'delta_rel': delta_rel,
            'sigma_sq': model.sigma_sq_,
            'converged': model.converged_
    }, index=[0])

    dir_base = 'output_parameter_recovery'
    out_dir = os.path.join(dir_base, 'sim_k{}_t{}_n{}')
    out_dir = out_dir.format(n_layers, n_time_steps, n_nodes)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = 'results{}.csv'.format(seed)
    data.to_csv(os.path.join(out_dir, out_file), index=False)



# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

# scenario 1
for n_nodes in [50, 100, 200, 500, 1000]:
    for i in range(n_reps):
        sim_single(seed=i, n_nodes=n_nodes, n_layers=5, n_time_steps=10)


# scenario 2
for n_layers in [5, 10, 20]:
    for i in range(n_reps):
        sim_single(seed=i, n_nodes=100, n_layers=n_layers, n_time_steps=10)


# scenario 3
for n_time_steps in [10, 50, 100]:
    for i in range(n_reps):
        sim_single(seed=i, n_nodes=100, n_layers=5, n_time_steps=n_time_steps)
