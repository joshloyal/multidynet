import os

import joblib
import numpy as np

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import load_icews
from multidynet.model_selection import train_test_split
from multidynet.metrics import calculate_auc


Y, countries, layer_labels, time_labels = load_icews(dataset='large')


model = DynamicMultilayerNetworkLSM(max_iter=50, n_features=4,
                                    lambda_var_prior=10.,
                                    lambda_odds_prior=1,
                                    tol=1e-2, init_type='svt',
                                    init_covariance_type='full',
                                    random_state=82590)
model.fit(Y)


#out_dir = 'icews_results'
#if not os.path.exists(out_dir):
#    os.makedirs(out_dir)
#out_file = os.path.join(out_dir, 'icews_model.pkl')
#
#joblib.dump(model, out_file, compress=('zlib', 6))
