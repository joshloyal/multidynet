import os
import plac

import joblib
import numpy as np

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import load_icews


max_iter = 10000

def run_icews(n_features):
    Y, countries, layer_labels, time_labels = load_icews(dataset='large')

    model = DynamicMultilayerNetworkLSM(max_iter=max_iter, n_features=n_features,
                                        lambda_var_prior=4.,
                                        lambda_odds_prior=1,
                                        tol=1e-2, init_type='both',
                                        n_init=4, n_jobs=-1,
                                        init_covariance_type='full',
                                        random_state=82590)
    model.fit(Y)


    out_dir = 'output_icews'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'icews_model_p{}.pkl'.format(n_features))

    joblib.dump(model, out_file, compress=('zlib', 6))


for d in range(1, 7):
    run_icews(n_features)
