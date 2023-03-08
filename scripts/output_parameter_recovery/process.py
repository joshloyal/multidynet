import pandas as pd
import os
import glob

for dir_name in glob.glob('sim_k*'):
    data = None
    for file_name in glob.glob(os.path.join(dir_name, '*.csv')):
        if data is None:
            data = pd.read_csv(file_name)
        else:
            data = pd.concat((data, pd.read_csv(file_name)))

    data.to_csv(os.path.join(dir_name, dir_name, '.csv'), index=False)
