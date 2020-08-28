import numpy as np
import pandas as pd
import joblib

from os.path import dirname, join


__all__ = ['load_icews']


def load_icews():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    file_name = join(file_path, 'icews', 'numpy_data', 'icews_networks.gz')
    Y = joblib.load(open(file_name, 'rb'))

    countries = np.loadtxt(
        join(file_path, 'icews', 'numpy_data', 'icews_countries.txt'),
        delimiter='\n', dtype=np.unicode)


    layer_labels = ['Verbal Cooperation', 'Material Cooperation',
                    'Verbal Conflict', 'Material Conflict']

    time_labels = pd.date_range(
        start='January 01 2009', end='December 31 2016', freq='M')
    time_labels = time_labels.strftime('%b %Y')

    return np.ascontiguousarray(Y), countries, layer_labels, time_labels
