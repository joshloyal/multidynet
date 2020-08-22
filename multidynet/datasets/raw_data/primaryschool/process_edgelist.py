import pandas as pd

from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('primaryschool.csv', sep='\t',
                 header=None,
                 names=['timestamp', 'node_a', 'node_b', 'class_a', 'class_b'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

df['day'] = df['timestamp'].apply(lambda x : x.day)
df['hour'] = df['timestamp'].apply(lambda x : x.hour)

# remove first day since it is not measured over all periods
#df = df.query('day > 2')

# re-label node ids to be contiguous
encoder = LabelEncoder().fit(df[['node_a', 'node_b']].values.ravel())
df['node_a'] = encoder.transform(df['node_a']) + 1  # make one indexed
df['node_b'] = encoder.transform(df['node_b']) + 1

df.to_csv('edgelist.csv', index=False)
