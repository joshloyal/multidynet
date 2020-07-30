import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import LabelEncoder


layer_type = ['trades', 'messages', 'attacks']
file_fmt = '{}/{}-timestamped-2009-12-{}.csv'

n_layers = 3
n_time_steps = 30

def get_edgelists():
    df = pd.DataFrame(columns=['node_a', 'node_b', 'k', 't'])
    for k in range(n_layers):
        for t in range(n_time_steps):
            file_name = file_fmt.format(layer_type[k], layer_type[k], t + 1)
            data = pd.read_csv(
                file_name, header=None, names=['timestamp', 'node_a', 'node_b'])
            data = data[['node_a', 'node_b']]
            data['k'] = k
            data['t'] = t
            df = pd.concat((df, data))

    return df


# load edgelist
#data = get_edgelists()
#data.to_csv('travian_edgelists.csv', index=False)

data = pd.read_csv('travian_edgelists.csv')

# filter nodes that appear across all layers
node_ids = np.unique(data[['node_a', 'node_b']].values.ravel())
for k in range(n_layers):
    data_k = data.query('k == {}'.format(k))
    nodes_k = np.unique(data_k[['node_a', 'node_b']].values.ravel())
    node_ids = [i for i in node_ids if i in nodes_k]

node_ids = np.asarray(node_ids)
data = data[data['node_a'].isin(node_ids)]
data = data[data['node_b'].isin(node_ids)]
n_nodes = node_ids.shape[0]

# re-label nodes
encoder = LabelEncoder().fit(data[['node_a', 'node_b']].values.ravel())
data['node_a'] = encoder.transform(data['node_a'].values)
data['node_b'] = encoder.transform(data['node_b'].values)

# create adjacency matrices
for k in range(n_layers):
    for t in range(n_time_steps):
        data_kt = (data.query('k == {}'.format(k))
                       .query('t == {}'.format(t)))
        edgelist = data_kt[['node_a', 'node_b']].values
        edge_data = np.ones(edgelist.shape[0])
        Y_tmp = sp.coo_matrix((edge_data, (edgelist[:, 0], edgelist[:, 1])),
                              shape=(n_nodes, n_nodes), dtype=np.int).toarray()
        Y_tmp += Y_tmp.T
        Y = (Y_tmp > 0).astype(int)
        Y[np.diag_indices_from(Y)] = 0.
        np.save('travian_{}_{}.npy'.format(k, t), Y)
