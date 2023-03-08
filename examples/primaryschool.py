import os

import joblib

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import load_primaryschool
from multidynet.model_selection import train_test_split
from multidynet.metrics import calculate_auc


Y, _, _, _, _ = load_primaryschool()

model = DynamicMultilayerNetworkLSM(max_iter=50, n_features=2,
                                    lambda_var_prior=10.,
                                    lambda_odds_prior=1,
                                    tol=1e-2, init_type='svt',
                                    random_state=82590)
model.fit(Y)

#out_dir = 'school_results'
#if not os.path.exists(out_dir):
#    os.makedirs(out_dir)
#out_file = os.path.join(out_dir, 'school_model.pkl')
#
#joblib.dump(model, out_file, compress=('zlib', 6))
