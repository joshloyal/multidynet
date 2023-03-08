import os
import numpy as np
import pandas as pd

from os.path import join
from joblib import Parallel, delayed

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import correlated_dynamic_multilayer_network
from multidynet.callbacks import TimeitCallback


times_smf = []
times_mf = []
n_layers = 5 
max_iter = 100

def run_benchmark():
    time_points = [10, 20, 30, 40]
    for t in time_points:
        Y, X, lmbda, delta, probas, dists, z = correlated_dynamic_multilayer_network(
            n_nodes=n_nodes, n_layers=n_layers, n_time_steps=t,
            n_features=2, tau=1.0, center=[0.75, 0.75], 
            rho=0.5, rho_t=0.4, sigma=0.05, 
            random_state=0)
        
        model = DynamicMultilayerNetworkLSM(
                        max_iter=max_iter, n_features=2,
                        init_covariance_type='full',
                        lambda_var_prior=4,
                        approx_type='mean_field',
                        init_type='svt',
                        stopping_criteria=None,
                        random_state=123)

        callback = TimeitCallback()
        model.fit(Y, callback=callback, verbose=False)
        med_time = np.median(np.diff(np.asarray(model.callback_.times_)))
        times_mf.append({'n_time_steps': t, 'algo': 'MF', 'time': med_time})

        model = DynamicMultilayerNetworkLSM(
                        max_iter=25, n_features=2,
                        init_covariance_type='full',
                        lambda_var_prior=4,
                        approx_type='structured',
                        init_type='svt',
                        stopping_criteria=None,
                        random_state=123)

        callback = TimeitCallback()
        model.fit(Y, callback=callback, verbose=False)
        med_time = np.median(np.diff(np.asarray(model.callback_.times_)))
        times_smf.append({'n_time_steps': t, 'algo': 'SMF', 'time': med_time})
        
    data_mf = pd.DataFrame(times_mf)
    data_smf = pd.DataFrame(times_smf)
    data = pd.concat([data_mf, data_smf])

    return data


def benchmark_single(n_nodes):
    data = pd.concat(Parallel(n_jobs=-1)(delayed(run_benchmark)() for i in range(10)))

    dir_name = 'output_run_time'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(join(dir_name, f'results_{n_nodes}.csv'), index=False)


for n_nodes in [25, 50, 100, 200]:
    benchmark_single(n_nodes)
