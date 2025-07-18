import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

import argparse
import numpy as np
import pandas as pd
import math, random
import torch
from random import sample
from scipy.special import softmax
from sklearn.preprocessing import PolynomialFeatures

from conformal import compute_conformity_score, compute_conformity_score_softmax, compute_conformity_score_aps, compute_conformity_score_raps, compute_sets_split, compute_sets_cond
from utils.model import get_image_classifier, split_test
from utils.data import get_image_dataset
from utils.binning import get_bin_edges

os.environ["DATA_DIR"] = 'data'
os.environ["CACHE_DIR"] = '~/.cache' # path where pretrained models are downloaded
os.environ["TRUST_SCORES_DIR"] = 'trust_scores'

def main(args):

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print("Using device:", device)

    # load dataset and precomputed features
    model, preprocess = get_image_classifier(args.model_name, device=device)

    # load precomputed features and trust scores
    calib_trust_scores, test_trust_scores = None, None
    if args.dataset_name in ["imagenet", "places", "fitzpatrick17k"]:
        train_dataset, test_dataset = get_image_dataset(args.dataset_name, preprocess=preprocess)
        test_features, test_logits, test_labels = model.run_and_cache_outputs(test_dataset, args.batch_size, args.features_dir)
        test_labels, test_features, test_logits, _, calib_labels, calib_features, calib_logits, _ = split_test(test_labels, test_features, test_logits, split=0.5, seed=args.seed)  # splitting the test set into calibration and evaluation sets

        # load trust scores
        scores_dir = os.environ.get("TRUST_SCORES_DIR", None)
        calib_trust_scores = np.load(os.path.join(scores_dir, f"val_{args.dataset_name}_trust_seed{args.seed}.npy"), allow_pickle=True)
        test_trust_scores = np.load(os.path.join(scores_dir, f"test_{args.dataset_name}_trust_seed{args.seed}.npy"), allow_pickle=True)
        calib_trust_scores, test_trust_scores = np.expand_dims(calib_trust_scores, axis=-1), np.expand_dims(test_trust_scores, axis=-1)  # (n, 1), (n, 1)
    
    else:
        train_dataset, val_dataset, test_dataset = get_image_dataset(args.dataset_name, preprocess=preprocess)
        calib_features, calib_logits, calib_labels = model.run_and_cache_outputs(val_dataset, args.batch_size, args.features_dir)
        test_features, test_logits, test_labels = model.run_and_cache_outputs(test_dataset, args.batch_size, args.features_dir)

        # load trust scores
        scores_dir = os.environ.get("TRUST_SCORES_DIR", None)
        calib_trust_scores = np.load(os.path.join(scores_dir, f"val_{args.dataset_name}_trust.npy"), allow_pickle=True)
        test_trust_scores = np.load(os.path.join(scores_dir, f"test_{args.dataset_name}_trust.npy"), allow_pickle=True)
        calib_trust_scores, test_trust_scores = np.expand_dims(calib_trust_scores, axis=-1), np.expand_dims(test_trust_scores, axis=-1)  # (n, 1), (n, 1)
        
    # setup output folder
    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = os.path.join(args.output_dir, f"{args.dataset_name}_{args.model_name}")
    os.makedirs(results_dir, exist_ok=True)

    # compute conformity scores
    res_fname = f"alpha_{args.alpha}_score_fn_{args.score_fn}_scores_randomize_{args.scores_randomize}_temp_scale_{args.temp_scaling}_use_binning_{args.use_binning}_bins_{args.num_bins}_strategy_{args.bin_strategy}_seed_{args.seed}" if args.use_binning \
                                    else f"alpha_{args.alpha}_score_fn_{args.score_fn}_scores_randomize_{args.scores_randomize}_temp_scale_{args.temp_scaling}_use_binning_{args.use_binning}_seed_{args.seed}"
    calib_scores, test_scores, test_scores_all = None, None, None
    calib_scores, test_scores = compute_conformity_score(calib_logits, test_logits, calib_labels, test_labels, temp_scaling=args.temp_scaling)
    if args.score_fn == "softmax":
        calib_scores, test_scores, test_scores_all = compute_conformity_score_softmax(calib_logits, test_logits, calib_labels, test_labels, temp_scaling=args.temp_scaling)
    elif args.score_fn == "aps":
        calib_scores, test_scores, test_scores_all = compute_conformity_score_aps(calib_logits, test_logits, calib_labels, test_labels, rand=args.scores_randomize, temp_scaling=args.temp_scaling)
    elif args.score_fn == "raps":
        calib_scores, test_scores, test_scores_all = compute_conformity_score_raps(calib_logits, test_logits, calib_labels, test_labels, rand=args.scores_randomize, temp_scaling=args.temp_scaling)
    np.save(os.path.join(results_dir, f"{res_fname}_calib_conf_score_{args.score_fn}.npy"), calib_scores)
    np.save(os.path.join(results_dir, f"{res_fname}_test_conf_score_{args.score_fn}.npy"), test_scores)
    np.save(os.path.join(results_dir, f"{res_fname}_test_conf_score_all_{args.score_fn}.npy"), test_scores_all)

    # compute coverages and prediction sets for split conformal
    coverages_split, prediction_sets_split, set_sizes_split = compute_sets_split(calib_scores, test_scores, test_scores_all, args.alpha)
    np.save(os.path.join(results_dir, f"{res_fname}_coverages_split.npy"), coverages_split)
    np.save(os.path.join(results_dir, f"{res_fname}_prediction_sets_split.npy"), prediction_sets_split)
    np.save(os.path.join(results_dir, f"{res_fname}_set_sizes_split.npy"), set_sizes_split)

    # compute coverages and prediction sets for conditional conformal
    if args.use_binning:
        # construct phi
        calib_phi, test_phi = None, None
        calib_confidence, test_confidence = softmax(calib_logits, axis=1), softmax(test_logits, axis=1)

        print('performing binning. binning strategy:', args.bin_strategy, 'num_bins:', args.num_bins)
        calib_conf_bins, calib_trust_bins, calib_final_conf_scores, calib_final_trust_scores = get_bin_edges(calib_confidence, calib_trust_scores, args.num_bins, args.bin_strategy)
        test_conf_bins, test_trust_bins, test_final_conf_scores, test_final_trust_scores = get_bin_edges(test_confidence, test_trust_scores, args.num_bins, args.bin_strategy)

        # compute corresponding confidence and trust score bins
        calib_conf_phi, calib_trust_phi = np.zeros((calib_final_conf_scores.shape[0], args.num_bins)), np.zeros((calib_final_trust_scores.shape[0], args.num_bins))
        for bin_ind in range(args.num_bins):
            conf_mask = (calib_final_conf_scores <= calib_conf_bins[bin_ind+1]) & (calib_final_conf_scores > calib_conf_bins[bin_ind])
            trust_mask = (calib_final_trust_scores <= calib_trust_bins[bin_ind+1]) & (calib_final_trust_scores > calib_trust_bins[bin_ind])
            calib_conf_phi[:, bin_ind] = np.where(conf_mask, 1, calib_conf_phi[:, bin_ind])
            calib_trust_phi[:, bin_ind] = np.where(trust_mask, 1, calib_trust_phi[:, bin_ind])

        calib_phi = np.concatenate([calib_conf_phi, calib_trust_phi], axis=1)

        test_conf_phi, test_trust_phi = np.zeros((test_final_conf_scores.shape[0], args.num_bins)), np.zeros((test_final_trust_scores.shape[0], args.num_bins))
        for bin_ind in range(args.num_bins):
            conf_mask = (test_final_conf_scores <= test_conf_bins[bin_ind+1]) & (test_final_conf_scores > test_conf_bins[bin_ind])
            trust_mask = (test_final_trust_scores <= test_trust_bins[bin_ind+1]) & (test_final_trust_scores > test_trust_bins[bin_ind])
            test_conf_phi[:, bin_ind] = np.where(conf_mask, 1, test_conf_phi[:, bin_ind])
            test_trust_phi[:, bin_ind] = np.where(trust_mask, 1, test_trust_phi[:, bin_ind])

        test_phi = np.concatenate([test_conf_phi, test_trust_phi], axis=1)

        # save subgrouping
        np.save(os.path.join(results_dir, f"{res_fname}_calib_phi.npy"), calib_phi)
        np.save(os.path.join(results_dir, f"{res_fname}_test_phi.npy"), test_phi)
        
        coverages_cond, prediction_sets_cond, set_sizes_cond = compute_sets_cond(calib_phi, calib_scores, test_phi, test_scores, test_scores_all, args.alpha, rand=args.scores_randomize)
        np.save(os.path.join(results_dir, f"{res_fname}_coverages_cond.npy"), coverages_cond)
        np.save(os.path.join(results_dir, f"{res_fname}_prediction_sets_cond.npy"), prediction_sets_cond)
        np.save(os.path.join(results_dir, f"{res_fname}_set_sizes_cond.npy"), set_sizes_cond)

    else:
        for degree in args.degree:

            res_fname_degree = f"alpha_{args.alpha}_score_fn_{args.score_fn}_scores_randomize_{args.scores_randomize}_temp_scale_{args.temp_scaling}_use_binning_{args.use_binning}_seed_{args.seed}_degree_{degree}"
        
            # construct phi
            calib_phi, test_phi = None, None
            calib_confidence, test_confidence = softmax(calib_logits, axis=1), softmax(test_logits, axis=1)

            conf_phi = np.max(calib_confidence, axis=1)
            calib_phi = np.concatenate([conf_phi.reshape(-1, 1), calib_trust_scores], axis=1)
            poly = PolynomialFeatures(degree)
            calib_phi = poly.fit_transform(calib_phi)

            conf_phi = np.max(test_confidence, axis=1)
            test_phi = np.concatenate([conf_phi.reshape(-1, 1), test_trust_scores], axis=1)
            poly = PolynomialFeatures(degree)
            test_phi = poly.fit_transform(test_phi)

            # save subgrouping
            np.save(os.path.join(results_dir, f"{res_fname_degree}_calib_phi.npy"), calib_phi)
            np.save(os.path.join(results_dir, f"{res_fname_degree}_test_phi.npy"), test_phi)

            coverages_cond, prediction_sets_cond, set_sizes_cond = compute_sets_cond(calib_phi, calib_scores, test_phi, test_scores, test_scores_all, args.alpha, rand=args.scores_randomize)
            np.save(os.path.join(results_dir, f"{res_fname_degree}_coverages_cond.npy"), coverages_cond)
            np.save(os.path.join(results_dir, f"{res_fname_degree}_prediction_sets_cond.npy"), prediction_sets_cond)
            np.save(os.path.join(results_dir, f"{res_fname_degree}_set_sizes_cond.npy"), set_sizes_cond)

    print('\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run conditional conformal calibration')
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--features_dir", type=str, default="./features") 
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1, help="random seed")

    # conformity score
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--score_fn", type=str, default="aps", choices=['softmax', 'aps', 'raps'])
    parser.add_argument("--scores_randomize", action="store_true")
    parser.add_argument("--temp_scaling", action="store_true")

    # trust score
    parser.add_argument("--use_binning", action="store_true", help="whether to use discrete bins or continuous score features")
    parser.add_argument("--num_bins", type=int, default=5, help="number of bins for calibration")
    parser.add_argument("--bin_strategy", type=str, default="quantile", choices=['quantile', 'uniform'])

    # polynomial features
    parser.add_argument("--degree", metavar='N', type=int, nargs='*', default=[5], help='maximal degree of the polynomial features')

    # cvxpy
    parser.add_argument("--solver", type=str, default="mosek", help="solver for cvxpy")

    args = parser.parse_args()
    main(args)

   

