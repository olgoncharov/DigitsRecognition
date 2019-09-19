import numpy as np


def feature_normalize(data):
    mu = np.mean(data, axis=0)
    data_norm = data - mu
    sigma = np.std(data_norm, axis=0)
    sigma[sigma == 0] = 1   # чтобы избежать деления на ноль
    data_norm = data_norm / sigma

    return data_norm, mu, sigma
