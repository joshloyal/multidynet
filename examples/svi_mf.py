import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr

import numpy as np

from multidynet.mcmc import DynamicMultilayerEigenmodelHMC
from multidynet.advi import DynamicMultilayerEigenmodelADVI
from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import dynamic_multilayer_network, correlated_dynamic_multilayer_network
from multidynet.model_selection import train_test_split
from multidynet.callbacks import TestMetricsCallback
from sklearn.metrics import roc_auc_score



sigma = 0.05

Y, X, lmbda, delta, probas, dists, z = correlated_dynamic_multilayer_network(
    n_nodes=50, n_layers=5, n_time_steps=20,
    n_features=2, tau=np.array([0.5, 0.25]),
    center=0.5, rho_t=0.4, sigma=0.05,
    random_state=1)

n_layers, n_time_steps, n_nodes, _ = Y.shape
n_dyads = 0.5 * n_layers * n_time_steps * n_nodes * (n_nodes - 1)

print(Y.mean(axis=(2, 3)))

Y_train, test_indices = train_test_split(Y, test_size=0.2, random_state=23)

model_mf = DynamicMultilayerNetworkLSM(
                max_iter=50, n_features=2,
                init_covariance_type='full',
                approx_type='mean_field',
                tol=1e-2, n_init=10,
                stopping_criteria=None,
                n_jobs=-1,
                random_state=123)

callback = TestMetricsCallback(Y=Y, probas=probas, test_indices=test_indices)
model_mf.fit(Y_train, callback=callback)
#print('MF Bias(sigma_sq): ', np.abs(model_mf.sigma_sq_ - sigma**2))
#
## forecast accuracy
#y_true = []
#y_true_probas = []
#y_pred = []
#y_proba = model_mf.forecast_probas(random_state=42)
#subdiag = np.tril_indices(n_nodes, k=-1)
#for k in range(n_layers):
#    y_true.extend(Y[k, -1][subdiag])
#    y_true_probas.extend(probas[k, -1][subdiag])
#    y_pred.extend(y_proba[k][subdiag])
#print('forecast AUC:', roc_auc_score(y_true, y_pred))
#print('forecast correlation:', pearsonr(y_true_probas, y_pred)[0])
#
plt.plot(model_mf.callback_.times_, model_mf.callback_.correlations_,
         linewidth=2, linestyle='dashed', label='mean_field')
#for i in range(10):
#    plt.plot(model_mf._callbacks[i].times_, model_mf._callbacks[i].correlations_,
#             alpha=0.25, c='black', linestyle='dashed')

model_svi = DynamicMultilayerNetworkLSM(
                max_iter=50, n_features=2,
                init_covariance_type='full',
                approx_type='structured',
                tol=1e-2, n_init=10,
                stopping_criteria=None,
                n_jobs=-1,
                random_state=123)

callback = TestMetricsCallback(Y=Y, probas=probas, test_indices=test_indices)
model_svi.fit(Y_train, callback=callback)
#print('SVI Bias(sigma_sq): ', np.abs(model_svi.sigma_sq_ - sigma**2))
#
plt.plot(model_svi.callback_.times_, model_svi.callback_.correlations_, linewidth=2, label='structured')
#for i in range(10):
#    plt.plot(model_svi._callbacks[i].times_, model_svi._callbacks[i].correlations_,
#             alpha=0.25, c='black')

#
#plt.legend()
plt.xlabel('Wallclock Time [s]')
plt.ylabel('Pearson Correlation')
plt.show()


plt.plot(model_mf.callback_.correlations_, label='Structured VI')
plt.plot(model_svi.callback_.correlations_, label='Mean Field')
plt.xlabel('# of Iterations')
plt.ylabel('Pearson Correlation')
plt.show()

## forecast accuracy
#y_true = []
#y_true_probas = []
#y_pred = []
#y_proba = model_svi.forecast_probas(random_state=42)
#subdiag = np.tril_indices(n_nodes, k=-1)
#for k in range(n_layers):
#    y_true.extend(Y[k, -1][subdiag])
#    y_true_probas.extend(probas[k, -1][subdiag])
#    y_pred.extend(y_proba[k][subdiag])
#print('forecast AUC:', roc_auc_score(y_true, y_pred))
#print('forecast correlation:', pearsonr(y_true_probas, y_pred)[0])
