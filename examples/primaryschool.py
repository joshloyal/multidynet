from multidynet import DynamicMultilayerNetworkLSM
from multidynet.datasets import load_primaryschool


Y, _, _, _, _ = load_primaryschool()

model = DynamicMultilayerNetworkLSM(max_iter=50, n_features=2, random_state=42)
model.fit(Y)
