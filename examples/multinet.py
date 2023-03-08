import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr
import numpy as np

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.multinet.multinet import MultilayerNetworkLSM
from multidynet.datasets import dynamic_multilayer_network, correlated_dynamic_multilayer_network
from multidynet.model_selection import train_test_split
from multidynet.callbacks import TestMetricsCallback
from multidynet.plots import plot_network
from sklearn.metrics import roc_auc_score
from multidynet.metrics import calculate_correlation, calculate_auc, score_homophily_matrix 
from multidynet.metrics import score_latent_space, score_latent_space_perml, score_social_trajectories


seed = 0
n_features = 2
Y, X, lmbda, delta, probas, dists, z = correlated_dynamic_multilayer_network(
    n_nodes=100, n_layers=5, n_time_steps=6, n_features=n_features,
    #center=0.75, tau=0.5, include_delta=True,
    #center=[0.5, 0.5], tau=0.25, include_delta=True,
    center=[0.75, 0.75], tau=1.0, include_delta=True,
    rho=0.5, rho_t=0.8, sigma=0.2, random_state=seed)
print(np.cov(X[0].T))
print(lmbda)
print(Y.mean(axis=(2,3)))
#seed = 0
#n_features = 2
#Y, X, lmbda, delta, probas, dists = dynamic_multilayer_network(
#    n_nodes=100, n_layers=10, n_time_steps=10, n_features=n_features,
#    sigma_sq=(0.05**2), sigma_sq_delta=(0.05**2),
#    random_state=seed)
#print(lmbda)

#t = -1
#plt.scatter(X[t, :, 0], X[t, :, 1], c=z)
#plt.axis('equal')
#plt.show()
#
#plt.imshow(Y[0,0], cmap='Greys', interpolation='none')
#plt.show()
#
#plot_network(Y[0, 0], X[0], tau_sq=0.5**2)
#plt.show()
#
#plt.hist(probas[1,0].ravel(), bins=30, edgecolor='w')
#plt.show()
#
#for i in range(delta.shape[2]):
#    plt.plot(delta[0, :, i], 'k-', alpha=0.5)
#plt.show()

n_layers, n_time_steps, n_nodes, _ = Y.shape
n_dyads = 0.5 * n_layers * n_time_steps * n_nodes * (n_nodes - 1)
Y_train, test_indices = train_test_split(Y[:, :-1], test_size=0.2, random_state=seed)

probas_pred = np.zeros((n_layers, n_time_steps-1, n_nodes, n_nodes))
U = np.zeros((n_time_steps, n_nodes, n_features))
for t in range(n_time_steps-1):
    print('t = {}'.format(t))
    model = MultilayerNetworkLSM(
                max_iter=500, n_features=n_features,
                init_covariance_type='full',
                tol=1e-2, init_type='svt',
                stopping_criteria='loglik',
                n_init=4, n_jobs=-1,
                random_state=123 * (seed + 1))

    #callback = TestMetricsCallback(Y=Y[:, :-1], probas=probas[:, :-1], test_indices=test_indices)

    model.fit(Y_train[:, t, ...])
    probas_pred[:, t, ...] = model.probas_
    U[t] = model.X_ - np.mean(model.X_, axis=0)
    #print(model.lambda_)
    #indices =  [test_indices[k][t] for k in range(Y.shape[0])]
    #print(calculate_correlation(probas[:, t, ...], model.probas_, test_indices=indices))
    #print(calculate_auc(Y[:, t, :, :], model.probas_, test_indices=indices))

print(score_latent_space(X, U))
print(score_latent_space_perml(X, U))
print(calculate_correlation(probas[:, :-1], probas_pred, test_indices=test_indices))
print(calculate_auc(Y[:, :-1], probas_pred, test_indices=test_indices))
print(calculate_correlation(probas[:, -1], probas_pred[:, -1]))

model_joint = DynamicMultilayerNetworkLSM(
        max_iter=500, n_features=n_features,
        lambda_var_prior=4,
        init_covariance_type='full',
        approx_type='structured',
        tol=1e-2, init_type='svt',
        n_init=4, n_jobs=-1,
        random_state=123 * (seed + 1))
model_joint.fit(Y_train)
print(model_joint.lambda_)
#print(score_homophily_matrix(lmbda, model_joint.lambda_))
#print(score_latent_space(X, model_joint.Z_))
print(score_latent_space(X[:-1], model_joint.Z_))
print(score_latent_space_perml(X[:-1], model_joint.Z_))
#print(np.sum((delta - model_joint.gamma_) ** 2) / np.sum(delta ** 2))
#
print(calculate_correlation(probas[:, :-1], model_joint.probas_, test_indices=test_indices))
print(calculate_auc(Y[:, :-1], model_joint.probas_, test_indices=test_indices))

probas_forecast = model_joint.forecast_probas(random_state=1)
print(calculate_correlation(probas[:, -1], probas_forecast))
