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
Functions for computing coverages
"""

import numpy as np
import math
import cvxpy as cp
from tqdm import tqdm
from conditionalconformal.condconf import setup_cvx_problem_calib

def compute_coverages(XCal, scoresCal, XTest, scoresTest, alpha):
    qSplit = np.quantile(scoresCal,[math.ceil((1-alpha) * (len(scoresCal) + 1)) / len(scoresCal)])
    coveragesSplit = scoresTest <= qSplit

    coveragesCond = np.zeros(len(XTest))
    for i in tqdm(range(len(XTest))):
        prob = setup_cvx_problem_calib(1-alpha,None,
                                            np.concatenate((scoresCal,np.array([scoresTest[i]]))), np.vstack((XCal,XTest[i,:])),{})
        if "MOSEK" in cp.installed_solvers():
            # prob.solve(solver="MOSEK", verbose=False)
            try:
                prob.solve(solver="MOSEK", verbose=False)
            except cp.error.SolverError:
                prob.solve(solver="ECOS")
        else:
            prob.solve(verbose=True)
        coveragesCond[i] = scoresTest[i] <= XTest[i,:]@prob.constraints[2].dual_value

    return coveragesSplit, coveragesCond