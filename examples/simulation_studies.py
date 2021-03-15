import os
import numpy as np
import pandas as pd
import plac

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import dynamic_multilayer_network
from multidynet.model_selection import train_test_split
from multidynet.metrics import (
    calculate_auc, score_latent_space, score_homophily_matrix,
    score_social_trajectories)


def sim_single(seed, n_nodes=100, n_layers=10, n_time_steps=10):
    Y, X, lmbda, delta, probas, dists = dynamic_multilayer_network(
        n_nodes=n_nodes, n_layers=n_layers, n_time_steps=n_time_steps,
        tau_sq=4.0, sigma_sq=0.05,
        random_state=seed)

    # 80% - 20% split
    Y_train, test_indices = train_test_split(Y, test_size=0.2, random_state=42)

    # fit 10 models in parallel
    model = DynamicMultilayerNetworkLSM(max_iter=1000, n_features=2,
                                        lambda_var_prior=10,
                                        lambda_odds_prior=1,
                                        tol=1e-2, n_init=10,
                                        n_jobs=-1,
                                        random_state=123 * (seed + 1))
    model.fit(Y_train)

    # in-sample and out-of-sample AUC estimates
    insample_auc = model.auc_
    holdout_auc = calculate_auc(Y, model.probas_, test_indices)

    # compare connection probabilities
    proba_rel = np.sum((probas - model.probas_) ** 2) / np.sum(probas ** 2)

    # parameter estimates
    X_rel = score_latent_space(X, model.Z_)
    dist_rel = np.sum((dists - model.dist_) ** 2) / np.sum(dists ** 2)
    lambda_rel = score_homophily_matrix(lmbda, model.lambda_)
    delta_rel = np.sum((delta - model.gamma_) ** 2) / np.sum(delta ** 2)

    # save results
    data = pd.DataFrame({
            'insample_auc' : insample_auc,
            'holdhout_auc' : holdout_auc,
            'proba_rel': proba_rel,
            'dist_rel': dist_rel,
            'X_rel' : X_rel,
            'lambda_rel': lambda_rel,
            'delta_rel': delta_rel,
            'tau_sq': model.tau_sq_,
            'sigma_sq': model.sigma_sq_,
            'converged': model.converged_
    }, index=[0])

    out_dir = os.path.join('results', 'sim_k{}_t{}_n{}')
    out_dir = out_dir.format(n_layers, n_time_steps, n_nodes)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = 'results{}.csv'.format(seed)
    data.to_csv(os.path.join(out_dir, out_file), index=False)



# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 30

# simulation 1
for n_nodes in [50, 100, 200, 500, 1000]:
    for i in range(n_reps):
        sim_single(seed=i, n_nodes=n_nodes, n_layers=5, n_time_steps=10)
        print(i)


# simulation 2
for n_layers in [5, 10, 20]:
    for i in range(n_reps):
        sim_single(seed=i, n_nodes=100, n_layers=n_layers, n_time_steps=10)


# simulation 3
for n_time_steps in [10, 50, 100]:
    for i in range(n_reps):
        sim_single(seed=i, n_nodes=100, n_layers=5, n_time_steps=n_time_steps)
