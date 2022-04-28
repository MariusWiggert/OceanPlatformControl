import numpy as np


def euclidean_distance(state, target):
    return np.sqrt((state.lat - target.lat)**2 + (state.lon - target.lon)**2)