import numpy as np
import joblib

from os.path import dirname, join


__all__ = ['load_primaryschool']


def load_primaryschool():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    file_name = join(file_path, 'primaryschool', 'numpy_data',
                     'primaryschool.gz')
    Y = joblib.load(open(file_name, 'rb'))

    return np.ascontiguousarray(Y)
