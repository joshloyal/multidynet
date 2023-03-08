import numpy as np
import pandas as pd
import joblib

from os.path import dirname, join


__all__ = ['load_icews']


def load_icews(dataset='small', country_names='full', year_range=None):
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')

    dir_name = 'icews_small' if dataset == 'small' else 'icews_large'
    file_name = join(file_path, dir_name, 'numpy_data', 'icews_networks.gz')
    Y = joblib.load(open(file_name, 'rb'))

    if country_names == 'full':
        countries = pd.read_csv(
            join(file_path, dir_name, 'numpy_data', 'icews_countries.csv')).values.ravel()

    else:
        countries = np.loadtxt(
            join(file_path, dir_name, 'numpy_data', 'icews_countries_iso.txt'),
            delimiter='\n', dtype=np.unicode)


    layer_labels = ['Verbal Cooperation', 'Material Cooperation',
                    'Verbal Conflict', 'Material Conflict']

    time_labels = pd.date_range(
        start='January 01 2009', end='December 31 2016', freq='M')
    time_labels = time_labels.strftime('%B %Y')

    if year_range is not None:
        start = 'Jan {}'.format(year_range[0])
        end = 'Jan {}'.format(year_range[1] + 1)
        start_id = np.where(start == time_labels)[0][0]
        end_id = np.where(end == time_labels)[0][0]
        Y = Y[:, start_id:end_id, :, :]
        time_labels = time_labels[start_id:end_id]
    
    if dataset == 'small':
        return np.ascontiguousarray(Y)[:, -12:, ...], countries, layer_labels, time_labels[-12:]
    else:
        return np.ascontiguousarray(Y), countries, layer_labels, time_labels
