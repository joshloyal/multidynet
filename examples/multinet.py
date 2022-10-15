import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr

import numpy as np

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.multinet.multinet import MultilayerNetworkLSM
from multidynet.datasets import dynamic_multilayer_network, correlated_dynamic_multilayer_network
from multidynet.model_selection import train_test_split
from multidynet.callbacks import TestMetricsCallback
from sklearn.metrics import roc_auc_score
from multidynet.metrics import calculate_correlation, calculate_auc


Y, X, lmbda, delta, probas, dists, z = correlated_dynamic_multilayer_network(
    n_nodes=100, n_layers=4, n_time_steps=10, n_features=2,
    center=0.75, tau=[0.25, 0.5], include_delta=True,
    rho=0., rho_t=0.8, sigma=0.01,
    random_state=1)
print(lmbda)

plt.scatter(X[0, :, 0], X[0, :, 1], c=z)
plt.axis('equal')
plt.show()

n_layers, n_time_steps, n_nodes, _ = Y.shape
n_dyads = 0.5 * n_layers * n_time_steps * n_nodes * (n_nodes - 1)

print(Y.mean(axis=(2, 3)))

Y_train, test_indices = train_test_split(Y, test_size=0.2, random_state=23)

probas_pred = np.zeros((n_layers, n_time_steps, n_nodes, n_nodes))
for t in range(n_time_steps):
    print('t = {}'.format(t))
    model = MultilayerNetworkLSM(
                max_iter=50, n_features=2,
                init_covariance_type='full',
                tol=1e-2, n_init=10,
                stopping_criteria='loglik',
                n_jobs=-1,
                random_state=123)

    #callback = TestMetricsCallback(Y=Y[:, :-1], probas=probas[:, :-1], test_indices=test_indices)

    model.fit(Y_train[:, t, ...])
    probas_pred[:, t, ...] = model.probas_

#test_indices =  [test_indices[k][t] for k in range(Yk.shape[0])]
#print(calculate_correlation(probas[:, t, ...], model.probas_, test_indices=test_indices))
#print(calculate_auc(Y[:, t, :, :], model.probas_, test_indices=test_indices))

print(calculate_correlation(probas, probas_pred, test_indices=test_indices))
print(calculate_auc(Y, probas_pred, test_indices=test_indices))

model_joint = DynamicMultilayerNetworkLSM(
        max_iter=50, n_features=2,
        init_covariance_type='full',
        approx_type='structured',
        tol=1e-2, n_init=10,
        stopping_criteria='loglik',
        n_jobs=-1,
        random_state=123)
model_joint.fit(Y_train)

print(calculate_correlation(probas, model_joint.probas_, test_indices=test_indices))
print(calculate_auc(Y, model_joint.probas_, test_indices=test_indices))

model_mf = DynamicMultilayerNetworkLSM(
        max_iter=50, n_features=2,
        init_covariance_type='full',
        approx_type='mean_field',
        tol=1e-2, n_init=10,
        stopping_criteria='loglik',
        n_jobs=-1,
        random_state=123)
model_mf.fit(Y_train)

print(calculate_correlation(probas, model_mf.probas_, test_indices=test_indices))
print(calculate_auc(Y, model_mf.probas_, test_indices=test_indices))
