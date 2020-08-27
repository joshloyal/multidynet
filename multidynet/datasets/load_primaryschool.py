import joblib
import numpy as np
import pandas as pd

from os.path import dirname, join


__all__ = ['load_primaryschool']


def load_primaryschool(minute_window=30):
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    # adjacency matrices
    if minute_window == 30:
        file_name = join(file_path, 'primaryschool', 'numpy_data',
                         'primaryschool30.gz')
        Y = joblib.load(open(file_name, 'rb'))
    else:
        file_name = join(file_path, 'primaryschool', 'numpy_data',
                         'primaryschool20.gz')
        Y = joblib.load(open(file_name, 'rb'))

    # covariates
    file_name = join(file_path, 'primaryschool', 'numpy_data',
                     'covariates.csv')
    X = pd.read_csv(file_name).values

    return np.ascontiguousarray(Y), X[:, 1], X[:, 2]
