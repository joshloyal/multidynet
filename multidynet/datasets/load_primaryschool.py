import joblib
import numpy as np
import pandas as pd

from os.path import dirname, join


__all__ = ['load_primaryschool']


def load_primaryschool():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    # adjacency matrices
    #if minute_window == 30:
    #    file_name = join(file_path, 'primaryschool', 'numpy_data',
    #                     'primaryschool30.gz')
    #    Y = joblib.load(open(file_name, 'rb'))
    #else:
    #    file_name = join(file_path, 'primaryschool', 'numpy_data',
    #                     'primaryschool20.gz')
    #    Y = joblib.load(open(file_name, 'rb'))

    file_name = join(file_path, 'primaryschool', 'numpy_data',
                     'primaryschool20.gz')
    Y = joblib.load(open(file_name, 'rb'))

    # covariates
    file_name = join(file_path, 'primaryschool', 'numpy_data',
                     'covariates.csv')
    X = pd.read_csv(file_name).values

    # labels
    layer_labels = ['Thursday', 'Friday']

    time_labels = [
        "8:30 to 9:20", "9:20 to 9:40", "9:40 to 10:00", "10:00 to 10:20",
        "10:20 to 10:40", "10:40 to 11:00", "11:00 to 11:20", "11:20 to 11:40",
        "11:40 to 12:00", "12:00 to 12:20", "12:20 to 12:40", "12:40 to 1:00",
        "1:00 to 1:20", "1:20 to 1:40", "1:40 to 2:00", "2:00 to 2:20",
        "2:20 to 2:40", "2:40 to 3:00", "3:00 to 3:20", "3:20 to 3:40",
        "3:40 to 4:00", "4:00 to 4:20", "4:20 to 4:40", "4:40 to 5:30"
    ]

    return np.ascontiguousarray(Y), X[:, 1], X[:, 2], layer_labels, time_labels
