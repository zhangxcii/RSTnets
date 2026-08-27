import numpy as np


def relative_l2(ref, pred):
    return np.linalg.norm(ref - pred) / np.linalg.norm(ref) * 100
