from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import load_icews


Y, countries, layer_labels, time_labels = load_icews(dataset='large')


model = DynamicMultilayerNetworkLSM(max_iter=50, n_features=2, random_state=42)
model.fit(Y)
