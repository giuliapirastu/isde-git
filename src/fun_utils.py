from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    n_samples = x.shape[0]
    n_tr = int(n_samples * tr_fraction)

    # Genera una permutazione casuale degli indici dei campioni
    idx = np.random.permutation(n_samples)

    # Separa gli indici per training e test
    tr_idx = idx[:n_tr]
    ts_idx = idx[n_tr:]

    # Estrae i dati e le relative etichette
    tr_x, tr_y = x[tr_idx], y[tr_idx]
    ts_x, ts_y = x[ts_idx], y[ts_idx]

    return tr_x, tr_y, ts_x, ts_y

