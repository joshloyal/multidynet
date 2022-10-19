import numpy as np

from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

from multidynet.model_selection import kfold
from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import dynamic_multilayer_network
from multidynet.datasets import correlated_dynamic_multilayer_network
from multidynet.metrics import calculate_auc


def kfold_selection(Y, n_features, n_folds=4, random_state=42):
    n_layers, n_time_steps, n_nodes, _ = Y.shape
    loglik = 0.
    auc = 0.
    folds = kfold(Y, n_splits=n_folds, random_state=random_state)
    for Y_train, test_indices in folds:
        # fit model
        model = DynamicMultilayerNetworkLSM(
                max_iter=50,
                n_features=n_features,
                lambda_var_prior=10,
                lambda_odds_prior=1,
                tol=1e-2, n_init=1,
                n_jobs=1,
                random_state=123)

        model.fit(Y_train)
        print(model.lambda_)
        loglik += model.loglikelihood(Y, test_indices=test_indices)

        y_true = []
        y_pred = []
        y_proba = model.probas_
        subdiag = np.tril_indices(n_nodes, k=-1)
        for k in range(n_layers):
            for t in range(n_time_steps):
                y_true.extend(Y[k, t][subdiag][test_indices[k][t]])
                y_pred.extend(y_proba[k, t][subdiag][test_indices[k][t]])
        auc += roc_auc_score(y_true, y_pred)

    return n_features, loglik / n_folds, auc / n_folds

def ic_selection(Y, n_features, random_state=42):
    model = DynamicMultilayerNetworkLSM(
            max_iter=50,
            n_features=n_features,
            lambda_var_prior=10,
            lambda_odds_prior=1,
            tol=1e-2, n_init=1,
            n_jobs=1,
            random_state=123).fit(Y, n_samples=1000)

    return model.aic_, model.bic_, model.dic(Y)[0], model.waic(Y)[0]

Y, X, lmbda, delta, probas, dists, z = correlated_dynamic_multilayer_network(
    n_nodes=100, n_layers=4, n_time_steps=10, n_features=2,
    tau=np.array([0.5, 0.75]),
    rho=0., rho_t=0.8, sigma=0.01,
    random_state=3)

#res = Parallel(n_jobs=-1)(delayed(kfold_selection)(Y, d) for d in range(1, 5))
#res = np.asarray(res)

res = Parallel(n_jobs=-1)(delayed(ic_selection)(Y, d) for d in range(1, 5))
res = np.asarray(res)
