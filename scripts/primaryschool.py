import os
import plac
import joblib

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import load_primaryschool


max_iter = 10000


def run_school(n_features):
    Y, _, _, _, _ = load_primaryschool(reference_layer='Friday')

    model = DynamicMultilayerNetworkLSM(max_iter=n_iter, n_features=n_features,
                                        lambda_var_prior=4.,
                                        lambda_odds_prior=1,
                                        tol=1e-2, init_type='both',
                                        n_init=4, n_jobs=-1,
                                        random_state=8251990)
    model.fit(Y)

    out_dir = 'output_primaryschool'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, 'school_model_p{}.pkl'.format(n_features, n_iter))

    joblib.dump(model, out_file, compress=('zlib', 6))


for d in range(1, 7):
    run_school(n_features)
