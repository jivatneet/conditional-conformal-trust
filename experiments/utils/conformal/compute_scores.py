# Code is taken from https://github.com/jjcherian/conditional-conformal. The code is available for use under the MIT license.
# @article{gibbs2023conformal,
#     title={Conformal Prediction with Conditional Guarantees},
#     author={Isaac Gibbs, John J. Cherian, Emmanuel J. Cand\`es},
#     publisher = {arXiv},
#     year = {2023},
#     note = {arXiv:2305.12616 [stat.ME]},
#     url = {https://arxiv.org/abs/2305.12616},
# }

"""
Functions for computing conformity scores
"""

import numpy as np
from temperature_scaling import torch_ts

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def class_conf_score(probs,y):
    return sum(probs[probs > probs[int(y)]])

def compute_conformity_score(XTrain, XTest, yTrain, yTest, temp_scaling=True):
    scaleFactor = 1.0

    if temp_scaling:
        T = torch_ts(XTrain,yTrain)
        scaleFactor = T.detach().numpy()
    
    normXtrain = np.apply_along_axis(softmax, 1, XTrain/scaleFactor)
    normXtest = np.apply_along_axis(softmax, 1, XTest/scaleFactor)

    scoresTrain = np.zeros(len(yTrain))
    for i in range(len(yTrain)):
        scoresTrain[i] = class_conf_score(normXtrain[i,:],yTrain[i])

    scoresTest = np.zeros(len(yTest))
    for i in range(len(yTest)):
        scoresTest[i] = class_conf_score(normXtest[i,:],yTest[i])
    return scoresTrain, scoresTest

def compute_conformity_score_softmax(calib_logits, test_logits, calib_labels, test_labels, rand=False, temp_scaling=True):
    scaleFactor = 1.0

    if temp_scaling:
        T = torch_ts(calib_logits, calib_labels)
        scaleFactor = T.detach().numpy()
    
    cal_smx = np.apply_along_axis(softmax, 1, calib_logits/scaleFactor)
    test_smx = np.apply_along_axis(softmax, 1, test_logits/scaleFactor)

    n = cal_smx.shape[0]
    cal_scores = 1 - cal_smx[np.arange(n), calib_labels]

    n_test = test_smx.shape[0]
    test_scores = 1 - test_smx[np.arange(n_test), test_labels]
    test_scores_all = 1 - test_smx

    return cal_scores, test_scores, test_scores_all

def compute_conformity_score_aps(calib_logits, test_logits, calib_labels, test_labels, rand=False, temp_scaling=True):
    """
    Code is taken from https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-raps.ipynb
    and modified. The code has the MIT License.
    """
    scaleFactor = 1.0

    if temp_scaling:
        T = torch_ts(calib_logits, calib_labels)
        scaleFactor = T.detach().numpy()
    
    cal_smx = np.apply_along_axis(softmax, 1, calib_logits/scaleFactor)
    test_smx = np.apply_along_axis(softmax, 1, test_logits/scaleFactor)

    n = cal_smx.shape[0]
    cal_pi = cal_smx.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt 
    cal_L = np.where(cal_pi == calib_labels[:, None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n), cal_L] if rand else (cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L]) # - cal_srt_reg[np.arange(n), cal_L])

    n_test = test_smx.shape[0]
    test_pi = test_smx.argsort(1)[:,::-1]
    test_srt = np.take_along_axis(test_smx, test_pi, axis=1)
    test_srt_reg = test_srt 
    test_scores_cumsum_intermed = (test_srt_reg.cumsum(axis=1) - np.random.rand(n_test, 1)*test_srt_reg) if rand else (test_srt_reg.cumsum(axis=1)) # - test_srt_reg) 
    test_L = np.where(test_pi == test_labels[:, None])[1]
    test_scores = test_scores_cumsum_intermed[np.arange(n_test), test_L] 
    
    test_scores_cumsum_ordered = np.take_along_axis(test_scores_cumsum_intermed, test_pi.argsort(axis=1), axis=1)

    return cal_scores, test_scores, test_scores_cumsum_ordered

def compute_conformity_score_raps(calib_logits, test_logits, calib_labels, test_labels, rand=False, temp_scaling=True, lam_reg=0.01, k_reg=5):
    """
    Code is taken from https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-raps.ipynb
    and modified. The code has the MIT License.
    """
    scaleFactor = 1.0

    if temp_scaling:
        T = torch_ts(calib_logits, calib_labels)
        scaleFactor = T.detach().numpy()
    
    cal_smx = np.apply_along_axis(softmax, 1, calib_logits/scaleFactor)
    test_smx = np.apply_along_axis(softmax, 1, test_logits/scaleFactor)

    reg_vec = np.array(k_reg*[0,] + (cal_smx.shape[1] - k_reg) * [lam_reg,])[None,:]

    n = cal_smx.shape[0]
    cal_pi = cal_smx.argsort(1)[:,::-1]; 
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == calib_labels[:, None])[1]
    cal_scores = cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L] - np.random.rand(n)*cal_srt_reg[np.arange(n), cal_L] if rand else (cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L] - cal_srt_reg[np.arange(n), cal_L])

    n_test = test_smx.shape[0]
    test_pi = test_smx.argsort(1)[:,::-1]
    test_srt = np.take_along_axis(test_smx, test_pi, axis=1)
    test_srt_reg = test_srt + reg_vec
    test_scores_cumsum_intermed = (test_srt_reg.cumsum(axis=1) - np.random.rand(n_test, 1)*test_srt_reg) if rand else (test_srt_reg.cumsum(axis=1) - test_srt_reg) 
    test_L = np.where(test_pi == test_labels[:, None])[1]
    test_scores = test_scores_cumsum_intermed[np.arange(n_test), test_L] 
    
    test_scores_cumsum_ordered = np.take_along_axis(test_scores_cumsum_intermed, test_pi.argsort(axis=1), axis=1)

    return cal_scores, test_scores, test_scores_cumsum_ordered
