"""
Functions for computing prediction sets
"""

import numpy as np
import math
import cvxpy as cp
from tqdm import tqdm
from conditionalconformal import CondConf

def compute_sets_split(scores_cal, scores_test, scores_test_all, alpha):

    qhat = np.quantile(scores_cal, np.ceil((len(scores_cal) + 1) * (1 - alpha))/ len(scores_cal), method='higher')
    coverages = scores_test <= qhat
    prediction_sets = scores_test_all <= qhat
    set_sizes = prediction_sets.sum(axis=1)

    return coverages, prediction_sets, set_sizes

def compute_sets_cond(phi_cal, scores_cal, phi_test, scores_test, scores_test_all, alpha, rand=False):

    cond_conf = CondConf(None, None)
    cond_conf.setup_problem_precomputed(None, phi_cal, scores_cal)

    coverages = np.zeros(len(phi_test))
    set_sizes = np.zeros(len(phi_test))
    prediction_sets = np.zeros(scores_test_all.shape, dtype=bool)

    for i in tqdm(range(len(phi_test))):
        coverages[i], prediction_sets[i] = cond_conf.predict_classification(1 - alpha, None, phi_test[i, :], scores_test[i], scores_test_all[i], exact=True, randomize=rand)
        set_sizes[i] = prediction_sets[i].sum()

    return coverages, prediction_sets, set_sizes
