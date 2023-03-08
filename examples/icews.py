import matplotlib.pyplot as plt

from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import load_icews
from multidynet.plots import plot_latent_space

Y, countries, layer_labels, time_labels = load_icews(dataset='small')


model = DynamicMultilayerNetworkLSM(
    max_iter=500, n_features=2, init_type='svt')

model.fit(Y)

print(model.lambda_)

fig, ax = plt.subplots(figsize=(20, 6), ncols=2)


k = 0
t = 0
plot_latent_space(Y[k, t], model.Z_[t], X_sigma=model.Z_sigma_[t],
                   with_labels=True, font_size=10,
                   contour_alpha=0.1,
                   size=0, edge_width=0,
                   node_labels=countries,
                   ax=ax[0])
ax[0].axhline(0, linestyle='--', c='k')
ax[0].axvline(0, linestyle='--', c='k')
ax[0].set_xlabel('Dimension 1')
ax[0].set_ylabel('Dimension 2')
ax[0].set_title(time_labels[t])

k = 0
t = 11
plot_latent_space(Y[k, t], model.Z_[t], X_sigma=model.Z_sigma_[t],
                   with_labels=True, font_size=10,
                   contour_alpha=0.1,
                   size=0, edge_width=0,
                   node_labels=countries,
                   ax=ax[1])
ax[1].axhline(0, linestyle='--', c='k')
ax[1].axvline(0, linestyle='--', c='k')
ax[1].set_xlabel('Dimension 1')
ax[1].set_ylabel('Dimension 2')
ax[1].set_title(time_labels[t])

fig.savefig('icews_small_ls.png', dpi=300, bbox_inches='tight')
