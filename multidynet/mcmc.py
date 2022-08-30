import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import itertools

from jax import jit, random, vmap, value_and_grad
from sklearn.utils import check_random_state
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs

from .model_selection import dynamic_multilayer_adjacency_to_vec


def find_permutation(x_ref, x):
    best_perm = None
    best_rel = np.inf
    n_features = x.shape[-1]
    for perm in itertools.permutations(np.arange(n_features)):
        rel = np.sum((x_ref - x[:, perm]) ** 2)
        if rel < best_rel:
            best_perm = perm
    return best_perm



def dynamic_multilayer_eigenmodel(
        Y, n_layers, n_time_steps, n_nodes, n_features=2):

    # reference layer
    lmbda0 = numpyro.sample("lambda0",
            dist.Bernoulli(probs=jnp.repeat(0.5, n_features)))
    lmbda0 = 2 * lmbda0 - 1

    # homophily coefficients
    lmbdak = jnp.sqrt(10) * numpyro.sample("lmbdak",
            dist.Normal(0, 1).expand([n_layers - 1, n_features]))

    lmbda = jnp.concatenate((lmbda0.reshape(1, -1), lmbdak), axis=0)
    numpyro.deterministic("lambda", lmbda)

    # delta
    a_tau_sq = 2.05
    b_tau_sq = 1.05 * 10
    tau_delta = numpyro.sample("tau_delta",
            dist.InverseGamma(concentration=a_tau_sq, rate=b_tau_sq))

    c_sigma_sq = 1
    d_sigma_sq = 1
    sigma_delta = numpyro.sample("sigma_delta",
                dist.InverseGamma(concentration=c_sigma_sq, rate=d_sigma_sq))

    X0_delta = jnp.sqrt(tau_delta) * numpyro.sample("X0_delta",
            dist.Normal(0, 1).expand([n_layers, n_nodes]))

    Z_delta = numpyro.sample("Z_delta",
        dist.GaussianRandomWalk(
                scale=jnp.sqrt(sigma_delta) * np.ones(n_layers*n_nodes),
            num_steps=n_time_steps-1)).reshape(-1, n_layers, n_nodes)

    delta = X0_delta + jnp.concatenate(
        (jnp.zeros((1, n_layers, n_nodes)), Z_delta), axis=0)

    delta = numpyro.deterministic("delta", delta.transpose((1, 0, 2)))

    # latent space
    a_tau_sq = 2.05
    b_tau_sq = 1.05 * 10
    tau = numpyro.sample("tau",
            dist.InverseGamma(concentration=a_tau_sq, rate=b_tau_sq))

    c_sigma_sq = 1
    d_sigma_sq = 1
    sigma = numpyro.sample("sigma",
            dist.InverseGamma(concentration=c_sigma_sq, rate=d_sigma_sq))

    X0 = jnp.sqrt(tau) * numpyro.sample("X0",
            dist.Normal(0, 1).expand([n_nodes, n_features]))
    Z = numpyro.sample("Z",
        dist.GaussianRandomWalk(
                scale=jnp.sqrt(sigma) * np.ones(n_nodes * n_features),
            num_steps=n_time_steps-1)).reshape(-1, n_nodes, n_features)
    X = X0 + jnp.concatenate(
        (jnp.zeros((1, n_nodes, n_features)), Z), axis=0)

    numpyro.deterministic("X", X)

    # calculate likelihood
    n_dyads = int(0.5 * n_nodes * (n_nodes-1))
    subdiag = jnp.tril_indices(n_nodes, k=-1)
    eta = jnp.zeros((n_layers, n_time_steps, n_dyads))
    for k in range(n_layers):
        for t in range(n_time_steps):
            d = delta[k, t].reshape(-1, 1)
            eta = eta.at[k, t].set(
                    (d + d.T + ((X[t] * lmbda[k]) @ X[t].T))[subdiag])

    with numpyro.handlers.condition(data={"Y": Y}):
        numpyro.sample("Y", dist.Bernoulli(logits=eta))


class DynamicMultilayerEigenmodelHMC(object):
    def __init__(self, n_features=2, random_state=42):
        self.n_features = n_features
        self.random_state = random_state

    def sample(self, Y, n_warmup=1000, n_samples=1000, adapt_delta=0.8):
        numpyro.enable_x64()

        n_layers, n_time_steps, n_nodes, _ = Y.shape
        y = dynamic_multilayer_adjacency_to_vec(Y)

        model_args = (y, n_layers, n_time_steps, n_nodes, self.n_features)
        model_kwargs = {}

        kernel = NUTS(
            dynamic_multilayer_eigenmodel, target_accept_prob=adapt_delta)
        kernel = DiscreteHMCGibbs(kernel, modified=False)
        self.mcmc_ = MCMC(
            kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=1)

        rng_key = random.PRNGKey(self.random_state)
        self.mcmc_.run(rng_key, *model_args,  **model_kwargs)
        self.samples_ = self.mcmc_.get_samples()
        self.samples_ = jax.tree_map(lambda x : np.array(x), self.samples_)

        # permute samples
        n_samples = self.samples_['lambda'].shape[0]
        best_lambda = self.samples_['lambda'][0]
        for idx in range(n_samples):
            perm = find_permutation(best_lambda, self.samples_['lambda'][idx])
            self.samples_['lambda'][idx] = self.samples_['lambda'][idx]

        return self
