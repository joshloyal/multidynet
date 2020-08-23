import pandas as pd

from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('primaryschool.csv', sep='\t',
                 header=None,
                 names=['timestamp', 'node_a', 'node_b', 'class_a', 'class_b'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

df['day'] = df['timestamp'].apply(lambda x : x.day)
df['hour'] = df['timestamp'].apply(lambda x : x.hour)
df['minute'] = df['timestamp'].apply(lambda x: x.minute)

def time_block(x):
    if x < 30:
        return 1
    else:
        return 2
df['time_block'] = df['minute'].apply(time_block)


# combine hours 8 and 9, since data collection started at 8:45 (day 1), 8:30 (day 2)
df.loc[df['hour'] == 8, 'time_block'] = 1
df.loc[df['hour'] == 8, 'minute'] = 0
df.loc[df['hour'] == 8, 'hour'] = 9

# combine hours 16 and 17, since the school day ends at 17:20 (day 1), 17:05 (day 2)
df.loc[df['hour'] == 17, 'time_block'] = 2
df.loc[df['hour'] == 17, 'minute'] = 59
df.loc[df['hour'] == 17, 'hour'] = 16

# re-label node ids to be contiguous
encoder = LabelEncoder().fit(df[['node_a', 'node_b']].values.ravel())
df['node_a'] = encoder.transform(df['node_a']) + 1  # make one indexed
df['node_b'] = encoder.transform(df['node_b']) + 1

df.to_csv('edgelist.csv', index=False)

# meta data
df = pd.read_csv('metadata_primaryschool.txt', sep="\t", header=None,
                 names=['node_id', 'class', 'gender'])
df['node_id'] = encoder.transform(df['node_id'])
df.to_csv('covariates.csv', index=False)
