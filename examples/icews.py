from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import load_icews


Y, countries, layer_labels, time_labels = load_icews(dataset='small')


model = DynamicMultilayerNetworkLSM(
    max_iter=500, n_features=2, init_type='svt')

model.fit(Y)
