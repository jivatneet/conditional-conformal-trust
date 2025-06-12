"""
Functions for aggregating results and computing evaluation metrics
"""

import os
import pandas as pd
import numpy as np
import faiss
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import scoreatpercentile
from scipy.stats import pearsonr
from utils.model import split_test


def get_ranks_array(logits, labels):

    def rank_of_true_label(logits, label):
        # sort logits in descending order
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        
        # find index of true label in sorted array
        true_label_index = np.where(sorted_indices == label)[0][0]
        
        # calculate rank (higher rank means further from top-1 accuracy)
        rank = true_label_index + 1
        
        return rank
    
    ranks = []
    for i in range(len(logits)):
        rank = rank_of_true_label(logits[i], labels[i])
        ranks.append(rank)

    return np.array(ranks)
    
def create_bin_mapping(num_bins_axis1, num_bins_axis2):
    """
    compute bin index mappings  
    """
    assignments_2d_to_1d, rev_assignments_1d_to_2d = {}, {}
    count = 0
    for i in range(num_bins_axis1):
        for j in range(num_bins_axis2):
            assignment = (i, j)
            assignments_2d_to_1d[assignment] = count
            rev_assignments_1d_to_2d[count] = assignment
            count += 1

    return assignments_2d_to_1d, rev_assignments_1d_to_2d

def get_bin_edges(scores_array, num_bins, bin_strategy):
    bin_edges = None

    if bin_strategy == "quantile":
        # code is taken from https://github.com/mertyg/beyond-confidence-atypicality. The code available for use under the MIT license.
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        bin_edges = np.quantile(scores_array, q=quantiles)

    elif bin_strategy == "uniform":
        score_min, score_max = scores_array.min(), scores_array.max()
        bin_edges = np.linspace(score_min, score_max, num_bins + 1)

    return bin_edges

def get_bin_indices(confidence_scores, ood_scores, conf_bin_edges, ood_bin_edges, assignments_2d_to_1d):
    bin_indices_subgrouping = np.zeros(confidence_scores.shape[0], dtype=int)

    for i in range(confidence_scores.shape[0]):
        for conf_ind in range(len(conf_bin_edges)-1):
            if (confidence_scores[i] <= conf_bin_edges[conf_ind+1]) & (confidence_scores[i] > conf_bin_edges[conf_ind]):
                for ood_ind in range(len(ood_bin_edges)-1):
                    if (ood_scores[i] <= ood_bin_edges[ood_ind+1]) & (ood_scores[i] > ood_bin_edges[ood_ind]):
                        bin_indices_subgrouping[i] = assignments_2d_to_1d[(conf_ind, ood_ind)]

    return bin_indices_subgrouping

def get_bin_indices_single_axis(scores, bin_edges):
    bin_indices = np.zeros(scores.shape[0], dtype=int)

    for i in range(scores.shape[0]):
        for ind in range(len(bin_edges)-1):
            if (scores[i] <= bin_edges[ind+1]) & (scores[i] > bin_edges[ind]):
                bin_indices[i] = ind

    return bin_indices


# load precomputed features, ood scores, and results
def load_precomputed_features(features_dir, dataset_name, model_name, split='test'):
    feats = np.load(os.path.join(features_dir, f'{model_name}_{dataset_name}_{split}_acts.npy'))
    labels = np.load(os.path.join(features_dir, f'{model_name}_{dataset_name}_{split}_lbls.npy'))
    logits = np.load(os.path.join(features_dir, f'{model_name}_{dataset_name}_{split}_logits.npy'))
    return feats, labels, logits

def load_ood_scores(ood_scores_dir, dataset_name, seed=None):
    calib_ood_scores, test_ood_scores = None, None
    if dataset_name in ["imagenet", "places", "fitzpatrick17k"]:
        calib_ood_scores = np.load(os.path.join(ood_scores_dir, f"val_{dataset_name}_trust_seed{seed}.npy"), allow_pickle=True)
        test_ood_scores = np.load(os.path.join(ood_scores_dir, f"test_{dataset_name}_trust_seed{seed}.npy"), allow_pickle=True)
    else:
        calib_ood_scores = np.load(os.path.join(ood_scores_dir, f"val_{dataset_name}_trust.npy"), allow_pickle=True)
        test_ood_scores = np.load(os.path.join(ood_scores_dir, f"test_{dataset_name}_trust.npy"), allow_pickle=True)
    return calib_ood_scores, test_ood_scores

# compute metrics
def compute_acc(logits, labels):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == labels)

def compute_acc_array_topk(logits, labels, k=5):
    topk_preds = np.argsort(logits, axis=1)[:, -k:]
    return [label in topk_preds[i] for i, label in enumerate(labels)]

def compute_size_acc_auc(set_sizes, accs):
    return roc_auc_score(accs, 1 - (set_sizes / set_sizes.max()))

def compute_coverage_gap(coverages_array, subgrouping, alpha=0.1):
    cov_gap = 0
    for i in np.unique(subgrouping):
        cov_gap += np.abs((1 - alpha) - np.mean(coverages_array[subgrouping == i]))

    return 100 * (cov_gap / len(np.unique(subgrouping)))

def compute_class_cond_coverage_gap(coverages_array, labels, alpha=0.1):
    cov_gap = 0
    for label in range(labels.max() + 1):
        temp_df = (labels == label)
        if temp_df.sum() == 0:
            continue
        cov_gap += np.abs((1 - alpha) - np.mean(coverages_array[temp_df])) 
    
    return 100 * (cov_gap / (labels.max() + 1))

def compute_coverage_gap_regions(coverages_array, euclidean_balls, alpha=0.1):
    cov_gap = 0
    for i in range(len(euclidean_balls)):
        cov_gap += np.abs((1 - alpha) - np.mean(coverages_array[euclidean_balls[i]]))

    return 100 * (cov_gap / len(euclidean_balls))

def compute_2d_sample_size_array(subgrouping, num_bins_conf, num_bins_ood, rev_assignments):
    size_arr = np.zeros((num_bins_conf, num_bins_ood))
    for i in np.unique(subgrouping):
        size_arr[rev_assignments[i][0], rev_assignments[i][1]] = sum(subgrouping == i)

    return size_arr

def compute_2d_coverage_array(coverages_array, subgrouping, num_bins_conf, num_bins_ood, rev_assignments):
    cov_arr = np.zeros((num_bins_conf, num_bins_ood))
    for i in np.unique(subgrouping):
        cov_arr[rev_assignments[i][0], rev_assignments[i][1]] = np.mean(coverages_array[subgrouping == i])

    return cov_arr

def compute_2d_set_size_array(set_sizes_array, subgrouping, num_bins_conf, num_bins_ood, rev_assignments):
    size_arr = np.zeros((num_bins_conf, num_bins_ood))
    for i in np.unique(subgrouping):
        size_arr[rev_assignments[i][0], rev_assignments[i][1]] = np.mean(set_sizes_array[subgrouping == i])

    return size_arr

def compute_worst_coverage(coverages_array, subgrouping):
    worst_cov = 1.
    for i in np.unique(subgrouping):
        # print('group index: ', i, 'num samples = ', np.sum(subgrouping == i), 'cov = ', np.mean(coverages_array[subgrouping == i]))
        worst_cov = min(worst_cov, np.mean(coverages_array[subgrouping == i]))

    return worst_cov

def compute_worst_coverage_regions(coverages_array, euclidean_balls):
    worst_cov = 1.
    for i in range(len(euclidean_balls)):
        worst_cov = min(worst_cov, np.mean(coverages_array[euclidean_balls[i]]))

    return worst_cov

def compute_worst_coverage_gap_high_label(coverages_array, subgrouping, labels_high):
    worst_cov_gap = 0.
    for i in np.unique(subgrouping):
        cov_gap = 0
        cov_labels_high = []
        for high_label in range(2):
            temp_df = (subgrouping == i) & (labels_high == high_label)
            cov_gap += np.abs(1 - np.mean(coverages_array[temp_df]))
            cov_labels_high.append(np.mean(coverages_array[temp_df]))
        worst_cov_gap = max(worst_cov_gap, np.abs(cov_labels_high[0] - cov_labels_high[1]))

    return worst_cov_gap

def compute_avg_set_size(sizes_array):
    return np.mean(sizes_array)

def compute_rank_stratified_set_size(sizes_array, ranks_subgrouping):
    avg_set_sizes = []
    for i in np.unique(ranks_subgrouping):
        avg_set_sizes.append(np.mean(sizes_array[ranks_subgrouping == i]))

    return avg_set_sizes

def fit_binwise_quantile(scores_cal, cal_subgrouping, alpha):
    qhat_dict = {}
    for i in np.unique(cal_subgrouping):
        qhat = np.quantile(scores_cal[cal_subgrouping == i], 1 - alpha, method='higher')
        qhat_dict[i] = qhat

    return qhat_dict

def compute_naive_sets(scores_test, scores_test_all, test_subgrouping, binwise_qhat):
    coverages = np.zeros(scores_test.shape[0])
    set_sizes = np.zeros(scores_test.shape[0])
    prediction_sets = np.zeros(scores_test_all.shape, dtype=bool)

    for i in np.unique(test_subgrouping):
        if i not in binwise_qhat:
            continue
        coverages[test_subgrouping == i] = scores_test[test_subgrouping == i] <= binwise_qhat[i]
        prediction_sets[test_subgrouping == i] = scores_test_all[test_subgrouping == i] <= binwise_qhat[i]
        set_sizes[test_subgrouping == i] = prediction_sets[test_subgrouping == i].sum(axis=1)

    return coverages, set_sizes

def initialize_metrics_dict(methods, num_rank_bins=4):
    metrics = {}
    for method in methods:
        metrics[method] = {
            'conf_trust_cov_gap': [],
            'conf_rank_cov_gap': [],
            'class_cov_gap': [],
            'avg_set_size': [],
            'marginal_cov': [],
            'set_size_top1_acc_auc': [],
            'set_size_top2_acc_auc': [],
            'set_size_top3_acc_auc': [],
            'set_size_top4_acc_auc': [],
            'set_size_top5_acc_auc': [],
        } 
        
    return metrics

def initialize_metrics_dict_fitzpatrick(methods):
    metrics = {}
    for method in methods:
        metrics[method] = {
            'fitzpatrick_scale_cov_gap': [],
            'worst_group_cov': [],
            'worst_high_label_cov_gap': [],
            'class_cov_gap': [],
            'avg_set_size': [],
            'marginal_cov': [],
        } 
        
    return metrics

def aggregate_results_over_seeds(dataset_name, model_name, features_dir, results_dir, res_fname, ood_scores_dir, methods, score_fn, degree, num_bins_conf=10, num_bins_ood=4, num_seeds=10, alpha=0.1, log_trust=False): 
    
    print("dataset: ", dataset_name, ", score_fn: ", score_fn, ", degree: ", degree)

    # load precomputed features
    test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
    calib_features, calib_labels, calib_logits = None, None, None
    calib_ood_scores, test_ood_scores = None, None

    # initialize results dict
    metrics_dict = initialize_metrics_dict(methods)
    assignments_2d_to_1d, rev_assignments_1d_to_2d = create_bin_mapping(num_bins_conf, num_bins_ood)

    for method in methods:
        for seed in range(1, num_seeds + 1):
            res_fname_seed = res_fname + f'_seed_{seed}'
            if dataset_name in ["imagenet", "places", "fitzpatrick17k"]:
                test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
                test_labels, test_features, test_logits, _, calib_labels, calib_features, calib_logits, _ = split_test(test_labels, test_features, test_logits, split=0.5, seed=seed)
                calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name, seed=seed)
                if log_trust:
                    calib_ood_scores, test_ood_scores = np.log(1 + calib_ood_scores.astype('float64')), np.log(1 + test_ood_scores.astype('float64'))   
            else:
                calib_features, calib_labels, calib_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='val')
                calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name)
                if log_trust:
                    calib_ood_scores, test_ood_scores = np.log(1 + calib_ood_scores.astype('float64')), np.log(1 + test_ood_scores.astype('float64'))   

            calib_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_calib_conf_score_{score_fn}.npy"))
            test_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_{score_fn}.npy"))
            test_scores_cumsum_ordered = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_all_{score_fn}.npy"))

            # get ranks array and bin edges
            ranks_array = get_ranks_array(test_logits, test_labels)
            rank_bin_edges = []
            if dataset_name == "imagenet":
                rank_bin_edges = [0., 1., 2., 4., 1000.]
            else:
                rank_bin_edges = get_bin_edges(ranks_array, num_bins_ood, bin_strategy='quantile')
                rank_bin_edges[0] = 0.
            test_confidence = softmax(test_logits, axis=1)
            test_conf_scores = np.max(test_confidence, axis=1)
            conf_bin_edges = get_bin_edges(test_conf_scores, num_bins_conf, bin_strategy='uniform')
            ood_bin_edges = get_bin_edges(test_ood_scores, num_bins_ood, bin_strategy='quantile')
            conf_rank_subgrouping = get_bin_indices(test_conf_scores, ranks_array, conf_bin_edges, rank_bin_edges, assignments_2d_to_1d)
            conf_ood_subgrouping = get_bin_indices(test_conf_scores, test_ood_scores, conf_bin_edges, ood_bin_edges, assignments_2d_to_1d)  
            rank_single_axis_subgrouping = get_bin_indices_single_axis(ranks_array, rank_bin_edges)

            if method == 'split':
                coverages = np.load(os.path.join(results_dir, f"{res_fname_seed}_coverages_split.npy"))
                set_sizes = np.load(os.path.join(results_dir, f"{res_fname_seed}_set_sizes_split.npy"))

                metrics_dict[method]['conf_trust_cov_gap'].append(compute_coverage_gap(coverages, conf_ood_subgrouping))
                metrics_dict[method]['conf_rank_cov_gap'].append(compute_coverage_gap(coverages, conf_rank_subgrouping))
                metrics_dict[method]['class_cov_gap'].append(compute_class_cond_coverage_gap(coverages, test_labels))
                metrics_dict[method]['avg_set_size'].append(compute_avg_set_size(set_sizes))
                metrics_dict[method]['marginal_cov'].append(np.mean(coverages))

                metrics_dict[method]['set_size_top1_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=1)))
                metrics_dict[method]['set_size_top2_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=5)))
                metrics_dict[method]['set_size_top3_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=10)))
                metrics_dict[method]['set_size_top4_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=15)))
                metrics_dict[method]['set_size_top5_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=20)))

            elif method == 'conditional': 
                res_fname_degree = res_fname_seed + f'_degree_{degree}'
                coverages = np.load(os.path.join(results_dir, f"{res_fname_degree}_coverages_cond.npy"))
                set_sizes = np.load(os.path.join(results_dir, f"{res_fname_degree}_set_sizes_cond.npy"))

                metrics_dict[method]['conf_trust_cov_gap'].append(compute_coverage_gap(coverages, conf_ood_subgrouping))
                metrics_dict[method]['conf_rank_cov_gap'].append(compute_coverage_gap(coverages, conf_rank_subgrouping))
                metrics_dict[method]['class_cov_gap'].append(compute_class_cond_coverage_gap(coverages, test_labels))
                metrics_dict[method]['avg_set_size'].append(compute_avg_set_size(set_sizes))
                metrics_dict[method]['marginal_cov'].append(np.mean(coverages))

                metrics_dict[method]['set_size_top1_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=1)))
                metrics_dict[method]['set_size_top2_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=5)))
                metrics_dict[method]['set_size_top3_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=10)))
                metrics_dict[method]['set_size_top4_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=15)))
                metrics_dict[method]['set_size_top5_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=20)))

            elif method == 'naive':
                calib_confidence = softmax(calib_logits, axis=1)
                calib_conf_scores = np.max(calib_confidence, axis=1)
                calib_conf_bin_edges = get_bin_edges(calib_conf_scores, num_bins_conf, bin_strategy='uniform')
                calib_ood_bin_edges = get_bin_edges(calib_ood_scores, num_bins_ood, bin_strategy='quantile')
                calib_conf_ood_subgrouping = get_bin_indices(calib_conf_scores, calib_ood_scores, calib_conf_bin_edges, calib_ood_bin_edges, assignments_2d_to_1d)  
                binwise_qhat = fit_binwise_quantile(calib_scores, calib_conf_ood_subgrouping, alpha)
                coverages, set_sizes = compute_naive_sets(test_scores, test_scores_cumsum_ordered, conf_ood_subgrouping, binwise_qhat)

                metrics_dict[method]['conf_trust_cov_gap'].append(compute_coverage_gap(coverages, conf_ood_subgrouping))
                metrics_dict[method]['conf_rank_cov_gap'].append(compute_coverage_gap(coverages, conf_rank_subgrouping))
                metrics_dict[method]['class_cov_gap'].append(compute_class_cond_coverage_gap(coverages, test_labels))
                metrics_dict[method]['avg_set_size'].append(compute_avg_set_size(set_sizes))
                metrics_dict[method]['marginal_cov'].append(np.mean(coverages))

                metrics_dict[method]['set_size_top1_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=1)))
                metrics_dict[method]['set_size_top2_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=5)))
                metrics_dict[method]['set_size_top3_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=10)))
                metrics_dict[method]['set_size_top4_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=15)))
                metrics_dict[method]['set_size_top5_acc_auc'].append(compute_size_acc_auc(set_sizes, compute_acc_array_topk(test_logits, test_labels, k=20)))
            
    # mean and std over seeds
    conf_trust_cov_gap_means = []
    conf_trust_cov_gap_ses = []
    conf_rank_cov_gap_means = []
    conf_rank_cov_gap_ses = []
    class_cov_gap_means = []
    class_cov_gap_ses = []
    avg_size_means = []
    avg_size_ses = []
    marginal_cov_means = []
    marginal_cov_ses = []

    for method in methods:
        conf_trust_cov_gap_means.append(np.mean(metrics_dict[method]['conf_trust_cov_gap']))
        conf_trust_cov_gap_ses.append(np.std(metrics_dict[method]['conf_trust_cov_gap']) / np.sqrt(num_seeds))
        conf_rank_cov_gap_means.append(np.mean(metrics_dict[method]['conf_rank_cov_gap']))
        conf_rank_cov_gap_ses.append(np.std(metrics_dict[method]['conf_rank_cov_gap']) / np.sqrt(num_seeds))
        class_cov_gap_means.append(np.mean(metrics_dict[method]['class_cov_gap']))
        class_cov_gap_ses.append(np.std(metrics_dict[method]['class_cov_gap']) / np.sqrt(num_seeds))
        avg_size_means.append(np.mean(metrics_dict[method]['avg_set_size']))
        avg_size_ses.append(np.std(metrics_dict[method]['avg_set_size']) / np.sqrt(num_seeds))
        marginal_cov_means.append(np.mean(metrics_dict[method]['marginal_cov']))
        marginal_cov_ses.append(np.std(metrics_dict[method]['marginal_cov']) / np.sqrt(num_seeds))

    df = pd.DataFrame({
        'method': methods,
        'conf_trust_cov_gap_mean': conf_trust_cov_gap_means,
        'conf_trust_cov_gap_se': conf_trust_cov_gap_ses,
        'conf_rank_cov_gap_mean': conf_rank_cov_gap_means,
        'conf_rank_cov_gap_se': conf_rank_cov_gap_ses,
        'class_cov_gap_mean': class_cov_gap_means,
        'class_cov_gap_se': class_cov_gap_ses,
        'avg_size_mean': avg_size_means,
        'avg_size_se': avg_size_ses,
        'marginal_cov_mean': marginal_cov_means,
        'marginal_cov_se': marginal_cov_ses,
    })

    return df

def aggregate_results_over_seeds_fitzpatrick(dataset_name, model_name, features_dir, results_dir, res_fname, ood_scores_dir, methods, score_fn, degree, num_bins_conf=1, num_bins_ood=7, num_seeds=10, alpha=0.1): 
    
    print("dataset: ", dataset_name, ", score_fn: ", score_fn, ", degree: ", degree)

    # load precomputed features
    test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
    calib_features, calib_labels, calib_logits = None, None, None
    calib_ood_scores, test_ood_scores = None, None

    # initialize results dict
    metrics_dict = initialize_metrics_dict_fitzpatrick(methods)
    assignments_2d_to_1d, rev_assignments_1d_to_2d = create_bin_mapping(num_bins_conf, num_bins_ood)

    for method in methods:
        for seed in range(1, num_seeds + 1):
            res_fname_seed = res_fname + f'_seed_{seed}'
            test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
            test_fitzpatrick = np.load(os.path.join(features_dir, f'{model_name}_{dataset_name}_test_fitzpatrick.npy'))
            test_labels_high = np.load(os.path.join(features_dir, f'{model_name}_{dataset_name}_test_lbls_high.npy'))
            test_labels_high, _ = train_test_split(test_labels_high, test_size=0.5, random_state=seed, stratify=test_labels)    
            test_labels, test_fitzpatrick, test_logits, _, calib_labels, calib_fitzpatrick, calib_logits, _ = split_test(test_labels, test_fitzpatrick, test_logits, split=0.5, seed=seed)
            calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name, seed=seed)

            calib_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_calib_conf_score_{score_fn}.npy"))
            test_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_{score_fn}.npy"))
            test_scores_cumsum_ordered = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_all_{score_fn}.npy"))

            # get bin edges for fitzpatrick scale
            test_confidence = softmax(test_logits, axis=1)
            test_conf_scores = np.max(test_confidence, axis=1)
            conf_bin_edges = get_bin_edges(test_conf_scores, num_bins_conf, bin_strategy='uniform')
            fitzpatrick_bin_edges = [-2, -1, 1, 2, 3, 4, 5, 6]
            fitzpatrick_single_axis_subgrouping = get_bin_indices_single_axis(test_fitzpatrick, fitzpatrick_bin_edges) 

            if method == 'split':
                coverages = np.load(os.path.join(results_dir, f"{res_fname_seed}_coverages_split.npy"))
                set_sizes = np.load(os.path.join(results_dir, f"{res_fname_seed}_set_sizes_split.npy"))

                metrics_dict[method]['fitzpatrick_scale_cov_gap'].append(compute_coverage_gap(coverages, fitzpatrick_single_axis_subgrouping))
                metrics_dict[method]['worst_group_cov'].append(compute_worst_coverage(coverages, fitzpatrick_single_axis_subgrouping))
                metrics_dict[method]['worst_high_label_cov_gap'].append(compute_worst_coverage_gap_high_label(coverages, fitzpatrick_single_axis_subgrouping, test_labels_high))
                metrics_dict[method]['class_cov_gap'].append(compute_class_cond_coverage_gap(coverages, test_labels))
                metrics_dict[method]['avg_set_size'].append(compute_avg_set_size(set_sizes))
                metrics_dict[method]['marginal_cov'].append(np.mean(coverages))

            elif method == 'conditional': 
                res_fname_degree = res_fname_seed + f'_degree_{degree}'
                coverages = np.load(os.path.join(results_dir, f"{res_fname_degree}_coverages_cond.npy"))
                set_sizes = np.load(os.path.join(results_dir, f"{res_fname_degree}_set_sizes_cond.npy"))

                metrics_dict[method]['fitzpatrick_scale_cov_gap'].append(compute_coverage_gap(coverages, fitzpatrick_single_axis_subgrouping))
                metrics_dict[method]['worst_group_cov'].append(compute_worst_coverage(coverages, fitzpatrick_single_axis_subgrouping))
                metrics_dict[method]['worst_high_label_cov_gap'].append(compute_worst_coverage_gap_high_label(coverages, fitzpatrick_single_axis_subgrouping, test_labels_high))
                metrics_dict[method]['class_cov_gap'].append(compute_class_cond_coverage_gap(coverages, test_labels))
                metrics_dict[method]['avg_set_size'].append(compute_avg_set_size(set_sizes))
                metrics_dict[method]['marginal_cov'].append(np.mean(coverages))

            elif method == 'naive':
                calib_confidence = softmax(calib_logits, axis=1)
                calib_conf_scores = np.max(calib_confidence, axis=1)
                calib_fitzpatrick_single_axis_subgrouping = get_bin_indices_single_axis(calib_fitzpatrick, fitzpatrick_bin_edges)
                binwise_qhat = fit_binwise_quantile(calib_scores, calib_fitzpatrick_single_axis_subgrouping, alpha)
                coverages, set_sizes = compute_naive_sets(test_scores, test_scores_cumsum_ordered, fitzpatrick_single_axis_subgrouping, binwise_qhat)

                metrics_dict[method]['fitzpatrick_scale_cov_gap'].append(compute_coverage_gap(coverages, fitzpatrick_single_axis_subgrouping))
                metrics_dict[method]['worst_group_cov'].append(compute_worst_coverage(coverages, fitzpatrick_single_axis_subgrouping))
                metrics_dict[method]['worst_high_label_cov_gap'].append(compute_worst_coverage_gap_high_label(coverages, fitzpatrick_single_axis_subgrouping, test_labels_high))
                metrics_dict[method]['class_cov_gap'].append(compute_class_cond_coverage_gap(coverages, test_labels))
                metrics_dict[method]['avg_set_size'].append(compute_avg_set_size(set_sizes))
                metrics_dict[method]['marginal_cov'].append(np.mean(coverages))
                
    # mean and std over seeds
    fitzpatrick_scale_cov_gap_means = []
    fitzpatrick_scale_cov_gap_ses = []
    worst_group_cov_means = []
    worst_group_cov_ses = []
    worst_high_label_cov_gap_means = []
    worst_high_label_cov_gap_ses = []
    class_cov_gap_means = []
    class_cov_gap_ses = []
    avg_size_means = []
    avg_size_ses = []
    marginal_cov_means = []
    marginal_cov_ses = []

    for method in methods:
        fitzpatrick_scale_cov_gap_means.append(np.mean(metrics_dict[method]['fitzpatrick_scale_cov_gap']))
        fitzpatrick_scale_cov_gap_ses.append(np.std(metrics_dict[method]['fitzpatrick_scale_cov_gap']) / np.sqrt(num_seeds))
        worst_group_cov_means.append(np.mean(metrics_dict[method]['worst_group_cov']))
        worst_group_cov_ses.append(np.std(metrics_dict[method]['worst_group_cov']) / np.sqrt(num_seeds))
        worst_high_label_cov_gap_means.append(np.mean(metrics_dict[method]['worst_high_label_cov_gap']))
        worst_high_label_cov_gap_ses.append(np.std(metrics_dict[method]['worst_high_label_cov_gap']) / np.sqrt(num_seeds))
        class_cov_gap_means.append(np.mean(metrics_dict[method]['class_cov_gap']))
        class_cov_gap_ses.append(np.std(metrics_dict[method]['class_cov_gap']) / np.sqrt(num_seeds))
        avg_size_means.append(np.mean(metrics_dict[method]['avg_set_size']))
        avg_size_ses.append(np.std(metrics_dict[method]['avg_set_size']) / np.sqrt(num_seeds))
        marginal_cov_means.append(np.mean(metrics_dict[method]['marginal_cov']))
        marginal_cov_ses.append(np.std(metrics_dict[method]['marginal_cov']) / np.sqrt(num_seeds))

    df = pd.DataFrame({
        'method': methods,
        'fitzpatrick_scale_cov_gap_mean': fitzpatrick_scale_cov_gap_means,
        'fitzpatrick_scale_cov_gap_se': fitzpatrick_scale_cov_gap_ses,
        'worst_group_cov_mean': worst_group_cov_means,
        'worst_group_cov_se': worst_group_cov_ses,
        'worst_high_label_cov_gap_mean': worst_high_label_cov_gap_means,
        'worst_high_label_cov_gap_se': worst_high_label_cov_gap_ses,
        'class_cov_gap_mean': class_cov_gap_means,
        'class_cov_gap_se': class_cov_gap_ses,
        'avg_size_mean': avg_size_means,
        'avg_size_se': avg_size_ses,
        'marginal_cov_mean': marginal_cov_means,
        'marginal_cov_se': marginal_cov_ses
    })

    return df

def return_2d_array_metric(dataset_name, model_name, x_axis, features_dir, results_dir, res_fname, ood_scores_dir, method, score_fn, degree, metric='coverage', num_bins_conf=10, num_bins_ood=4, seed=1, alpha=0.1, reciprocal=False):
    print("dataset: ", dataset_name, ", x axis: ", x_axis, ", score_fn: ", score_fn, ", degree: ", degree)

    # load precomputed features
    test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
    calib_features, calib_labels, calib_logits = None, None, None
    calib_ood_scores, test_ood_scores = None, None

    # initialize 2d array and assignments
    arr_2d = None
    assignments_2d_to_1d, rev_assignments_1d_to_2d = create_bin_mapping(num_bins_conf, num_bins_ood)

    res_fname_seed = res_fname + f'_seed_{seed}'
    if dataset_name in ["imagenet", "places", "fitzpatrick17k"]:
        test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
        test_labels, test_features, test_logits, _, calib_labels, calib_features, calib_logits, _ = split_test(test_labels, test_features, test_logits, split=0.5, seed=seed)
        calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name, seed=seed)
        if reciprocal:
            calib_ood_scores, test_ood_scores = 1 / (1 + calib_ood_scores.astype('float64')), 1 / (1 + test_ood_scores.astype('float64'))
    else:
        calib_features, calib_labels, calib_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='val')
        calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name)
        if reciprocal:
            calib_ood_scores, test_ood_scores = 1 / (1 + calib_ood_scores.astype('float64')), 1 / (1 + test_ood_scores.astype('float64'))

    calib_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_calib_conf_score_{score_fn}.npy"))
    test_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_{score_fn}.npy"))
    test_scores_cumsum_ordered = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_all_{score_fn}.npy"))

    # get confidence bin edges
    test_confidence = softmax(test_logits, axis=1)
    test_conf_scores = np.max(test_confidence, axis=1)
    conf_bin_edges = get_bin_edges(test_conf_scores, num_bins_conf, bin_strategy='uniform')

    conf_ood_subgrouping = None
    if x_axis == "rank":
        # get ranks array and bin edges
        ranks_array = get_ranks_array(test_logits, test_labels)
        rank_bin_edges = []
        if dataset_name == "imagenet":
            rank_bin_edges = [0., 1., 2., 4., 1000.]
        else:
            rank_bin_edges = get_bin_edges(ranks_array, num_bins_ood, bin_strategy='quantile')
            rank_bin_edges[0] = 0.
        conf_ood_subgrouping = get_bin_indices(test_conf_scores, ranks_array, conf_bin_edges, rank_bin_edges, assignments_2d_to_1d)

    elif x_axis == "trust":
        ood_bin_edges = get_bin_edges(test_ood_scores, num_bins_ood, bin_strategy='quantile')
        conf_ood_subgrouping = get_bin_indices(test_conf_scores, test_ood_scores, conf_bin_edges, ood_bin_edges, assignments_2d_to_1d)  

    coverages, set_sizes = None, None
    if method == 'split':
        coverages = np.load(os.path.join(results_dir, f"{res_fname_seed}_coverages_split.npy"))
        set_sizes = np.load(os.path.join(results_dir, f"{res_fname_seed}_set_sizes_split.npy"))

    elif method == 'conditional': 
        res_fname_degree = res_fname_seed + f'_degree_{degree}'
        coverages = np.load(os.path.join(results_dir, f"{res_fname_degree}_coverages_cond.npy"))
        set_sizes = np.load(os.path.join(results_dir, f"{res_fname_degree}_set_sizes_cond.npy"))

    elif method == 'naive':
        calib_confidence = softmax(calib_logits, axis=1)
        calib_conf_scores = np.max(calib_confidence, axis=1)
        calib_conf_bin_edges = get_bin_edges(calib_conf_scores, num_bins_conf, bin_strategy='uniform')
        calib_ood_bin_edges = get_bin_edges(calib_ood_scores, num_bins_ood, bin_strategy='quantile')
        calib_conf_ood_subgrouping = get_bin_indices(calib_conf_scores, calib_ood_scores, calib_conf_bin_edges, calib_ood_bin_edges, assignments_2d_to_1d)  
        binwise_qhat = fit_binwise_quantile(calib_scores, calib_conf_ood_subgrouping, alpha)
        coverages, set_sizes = compute_naive_sets(test_scores, test_scores_cumsum_ordered, conf_ood_subgrouping, binwise_qhat)

    # compute 2d array
    if metric == "coverage":
        arr_2d = compute_2d_coverage_array(coverages, conf_ood_subgrouping, num_bins_conf, num_bins_ood, rev_assignments_1d_to_2d)
    elif metric == "sample_size":
        arr_2d = compute_2d_sample_size_array(conf_ood_subgrouping, num_bins_conf, num_bins_ood, rev_assignments_1d_to_2d)
    elif metric == "set_size":
        arr_2d = compute_2d_set_size_array(set_sizes, conf_ood_subgrouping, num_bins_conf, num_bins_ood, rev_assignments_1d_to_2d)

    return arr_2d

def find_neighbor_regions(dataset_name, model_name, features_dir, results_dir, res_fname, ood_scores_dir, methods, score_fn, degree, num_bins_conf=10, num_bins_ood=4, seed=1, num_trials=100, alpha=0.1):
    
    print("dataset: ", dataset_name, ", score_fn: ", score_fn, ", degree: ", degree)

    # initialize results dict
    assignments_2d_to_1d, rev_assignments_1d_to_2d = create_bin_mapping(num_bins_conf, num_bins_ood)
    metrics_dict = {}
    for method in methods:
        metrics_dict[method] = {
            'region_cov_gap': [],
            'worst_region_cov': [],
        } 

    # load precomputed features
    test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
    calib_features, calib_labels, calib_logits = None, None, None
    calib_ood_scores, test_ood_scores = None, None

    res_fname_seed = res_fname + f'_seed_{seed}'
    if dataset_name in ["imagenet", "places", "fitzpatrick17k"]:
        test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
        test_labels, test_features, test_logits, _, calib_labels, calib_features, calib_logits, _ = split_test(test_labels, test_features, test_logits, split=0.5, seed=seed)
        calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name, seed=seed)
    else:
        calib_features, calib_labels, calib_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='val')
        calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name)

    calib_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_calib_conf_score_{score_fn}.npy"))
    test_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_{score_fn}.npy"))
    test_scores_cumsum_ordered = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_all_{score_fn}.npy"))

    test_confidence = softmax(test_logits, axis=1)
    test_conf_scores = np.max(test_confidence, axis=1)
    conf_bin_edges = get_bin_edges(test_conf_scores, num_bins_conf, bin_strategy='uniform')
    ood_bin_edges = get_bin_edges(test_ood_scores, num_bins_ood, bin_strategy='quantile')
    conf_ood_subgrouping = get_bin_indices(test_conf_scores, test_ood_scores, conf_bin_edges, ood_bin_edges, assignments_2d_to_1d)  
    
    # build FAISS index for test features
    n_points = test_features.shape[0]
    dim = test_features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(test_features)
    print("FAISS index built")

    # find neighboring points within a radius that have at least `min_points` neighbors
    def find_euclidean_ball(index, data, query_point, min_points=500):
        radius = 25.
        while True:
            distances, indices = index.search(np.array([query_point]), len(data))
            
            # filter by radius and remove the query point itself
            within_radius = distances[0] <= radius ** 2
            neighbors = indices[0][within_radius]
            neighbors = neighbors[neighbors != np.argmin(distances[0])]
            print(f"{len(neighbors)} neighbors found within radius {radius}")

            if len(neighbors) >= min_points:
                return neighbors
            radius *= 1.10  

    # compute cov gap and worst covergage across regions
    for trial in range(num_trials):
        print(f"Trial {trial+1}")
        # randomly sample 10 points from test points
        sampled_points_indices = np.random.choice(n_points, 20, replace=False)

        # find euclidean balls with minimum density
        euclidean_balls = []
        for idx in sampled_points_indices:
            query_point = test_features[idx]
            ball = find_euclidean_ball(index, test_features, query_point)
            euclidean_balls.append(ball)

        for i, ball in enumerate(euclidean_balls):
            print(f"Ball {i+1}: {len(ball)} points")

        for method in methods:
            if method == 'split':
                coverages = np.load(os.path.join(results_dir, f"{res_fname_seed}_coverages_split.npy"))
                set_sizes = np.load(os.path.join(results_dir, f"{res_fname_seed}_set_sizes_split.npy"))

                metrics_dict[method]['region_cov_gap'].append(compute_coverage_gap_regions(coverages, euclidean_balls))
                metrics_dict[method]['worst_region_cov'].append(compute_worst_coverage_regions(coverages, euclidean_balls))

            elif method == 'conditional': 
                res_fname_degree = res_fname_seed + f'_degree_{degree}'
                coverages = np.load(os.path.join(results_dir, f"{res_fname_degree}_coverages_cond.npy"))
                set_sizes = np.load(os.path.join(results_dir, f"{res_fname_degree}_set_sizes_cond.npy"))

                metrics_dict[method]['region_cov_gap'].append(compute_coverage_gap_regions(coverages, euclidean_balls))
                metrics_dict[method]['worst_region_cov'].append(compute_worst_coverage_regions(coverages, euclidean_balls))

            elif method == 'naive':
                calib_confidence = softmax(calib_logits, axis=1)
                calib_conf_scores = np.max(calib_confidence, axis=1)
                calib_conf_bin_edges = get_bin_edges(calib_conf_scores, num_bins_conf, bin_strategy='uniform')
                calib_ood_bin_edges = get_bin_edges(calib_ood_scores, num_bins_ood, bin_strategy='quantile')
                calib_conf_ood_subgrouping = get_bin_indices(calib_conf_scores, calib_ood_scores, calib_conf_bin_edges, calib_ood_bin_edges, assignments_2d_to_1d)  
                binwise_qhat = fit_binwise_quantile(calib_scores, calib_conf_ood_subgrouping, alpha)
                coverages, set_sizes = compute_naive_sets(test_scores, test_scores_cumsum_ordered, conf_ood_subgrouping, binwise_qhat)

                metrics_dict[method]['region_cov_gap'].append(compute_coverage_gap_regions(coverages, euclidean_balls))
                metrics_dict[method]['worst_region_cov'].append(compute_worst_coverage_regions(coverages, euclidean_balls))

    # mean and std over seeds
    region_cov_gap_means = []
    region_cov_gap_ses = []
    worst_region_cov_means = []
    worst_region_cov_ses = []

    for method in methods:
        region_cov_gap_means.append(np.mean(metrics_dict[method]['region_cov_gap']))
        region_cov_gap_ses.append(np.std(metrics_dict[method]['region_cov_gap']) / np.sqrt(num_trials))
        worst_region_cov_means.append(np.mean(metrics_dict[method]['worst_region_cov']))
        worst_region_cov_ses.append(np.std(metrics_dict[method]['worst_region_cov']) / np.sqrt(num_trials))

    df = pd.DataFrame({
        'method': methods,
        'region_cov_gap_mean': region_cov_gap_means,
        'region_cov_gap_se': region_cov_gap_ses,
        'worst_region_cov_mean': worst_region_cov_means,
        'worst_region_cov_se': worst_region_cov_ses
    })

    return df

def find_neighbor_regions_radii(dataset_name, model_name, features_dir, results_dir, res_fname, ood_scores_dir, methods, score_fn, degree, radius=10.0, num_bins_conf=10, num_bins_ood=4, seed=1, num_trials=100, alpha=0.1):
    
    print("dataset: ", dataset_name, ", score_fn: ", score_fn, ", degree: ", degree)

    # initialize results dict
    assignments_2d_to_1d, rev_assignments_1d_to_2d = create_bin_mapping(num_bins_conf, num_bins_ood)
    metrics_dict = {}
    for method in methods:
        metrics_dict[method] = {
            'region_cov_gap': [],
            'worst_region_cov': [],
        } 

    # load precomputed features
    test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
    calib_features, calib_labels, calib_logits = None, None, None
    calib_ood_scores, test_ood_scores = None, None

    res_fname_seed = res_fname + f'_seed_{seed}'
    if dataset_name in ["imagenet", "places", "fitzpatrick17k"]:
        test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
        test_labels, test_features, test_logits, _, calib_labels, calib_features, calib_logits, _ = split_test(test_labels, test_features, test_logits, split=0.5, seed=seed)
        calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name, seed=seed)
    else:
        calib_features, calib_labels, calib_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='val')
        calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name)

    calib_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_calib_conf_score_{score_fn}.npy"))
    test_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_{score_fn}.npy"))
    test_scores_cumsum_ordered = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_all_{score_fn}.npy"))

    test_confidence = softmax(test_logits, axis=1)
    test_conf_scores = np.max(test_confidence, axis=1)
    conf_bin_edges = get_bin_edges(test_conf_scores, num_bins_conf, bin_strategy='uniform')
    ood_bin_edges = get_bin_edges(test_ood_scores, num_bins_ood, bin_strategy='quantile')
    conf_ood_subgrouping = get_bin_indices(test_conf_scores, test_ood_scores, conf_bin_edges, ood_bin_edges, assignments_2d_to_1d)  
    
    # build FAISS index for test features
    n_points = test_features.shape[0]
    dim = test_features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(test_features)
    print("FAISS index built")

    # find neighboring points within a radius that have at least `min_points` neighbors
    radius_init = radius
    def find_euclidean_ball(index, data, query_point, radius_ball, min_points=10):
        while True:
            distances, indices = index.search(np.array([query_point]), len(data))
            
            # filter by radius and remove the query point itself
            within_radius = distances[0] <= radius_ball ** 2
            neighbors = indices[0][within_radius]
            neighbors = neighbors[neighbors != indices[0][np.argmin(distances[0])]]
            # print(f"{len(neighbors)} neighbors found within radius {radius_ball}")

            if len(neighbors) >= min_points:
                # print(f"{len(neighbors)} neighbors found within radius {radius_ball}")
                return neighbors
            radius_ball *= 1.10

    # compute cov gap and worst covergage across regions
    for trial in range(num_trials):
        print(f"Trial {trial+1}")
        # randomly sample 100 points from test points
        sampled_points_indices = np.random.choice(n_points, 100, replace=False)

        # find euclidean balls with minimum density
        euclidean_balls = []
        for idx in sampled_points_indices:
            query_point = test_features[idx]
            ball = find_euclidean_ball(index, test_features, query_point, radius_init)
            euclidean_balls.append(ball)

        for i, ball in enumerate(euclidean_balls):
            print(f"Ball {i+1}: {len(ball)} points")

        for method in methods:
            if method == 'split':
                coverages = np.load(os.path.join(results_dir, f"{res_fname_seed}_coverages_split.npy"))
                set_sizes = np.load(os.path.join(results_dir, f"{res_fname_seed}_set_sizes_split.npy"))

                metrics_dict[method]['region_cov_gap'].append(compute_coverage_gap_regions(coverages, euclidean_balls))
                metrics_dict[method]['worst_region_cov'].append(compute_worst_coverage_regions(coverages, euclidean_balls))

            elif method == 'conditional': 
                res_fname_degree = res_fname_seed + f'_degree_{degree}'
                coverages = np.load(os.path.join(results_dir, f"{res_fname_degree}_coverages_cond.npy"))
                set_sizes = np.load(os.path.join(results_dir, f"{res_fname_degree}_set_sizes_cond.npy"))

                metrics_dict[method]['region_cov_gap'].append(compute_coverage_gap_regions(coverages, euclidean_balls))
                metrics_dict[method]['worst_region_cov'].append(compute_worst_coverage_regions(coverages, euclidean_balls))

            elif method == 'naive':
                calib_confidence = softmax(calib_logits, axis=1)
                calib_conf_scores = np.max(calib_confidence, axis=1)
                calib_conf_bin_edges = get_bin_edges(calib_conf_scores, num_bins_conf, bin_strategy='uniform')
                calib_ood_bin_edges = get_bin_edges(calib_ood_scores, num_bins_ood, bin_strategy='quantile')
                calib_conf_ood_subgrouping = get_bin_indices(calib_conf_scores, calib_ood_scores, calib_conf_bin_edges, calib_ood_bin_edges, assignments_2d_to_1d)  
                binwise_qhat = fit_binwise_quantile(calib_scores, calib_conf_ood_subgrouping, alpha)
                coverages, set_sizes = compute_naive_sets(test_scores, test_scores_cumsum_ordered, conf_ood_subgrouping, binwise_qhat)

                metrics_dict[method]['region_cov_gap'].append(compute_coverage_gap_regions(coverages, euclidean_balls))
                metrics_dict[method]['worst_region_cov'].append(compute_worst_coverage_regions(coverages, euclidean_balls))

    # mean and std over seeds
    region_cov_gap_means = []
    region_cov_gap_ses = []
    worst_region_cov_means = []
    worst_region_cov_ses = []

    for method in methods:
        region_cov_gap_means.append(np.mean(metrics_dict[method]['region_cov_gap']))
        region_cov_gap_ses.append(np.std(metrics_dict[method]['region_cov_gap']) / np.sqrt(num_trials))
        worst_region_cov_means.append(np.mean(metrics_dict[method]['worst_region_cov']))
        worst_region_cov_ses.append(np.std(metrics_dict[method]['worst_region_cov']) / np.sqrt(num_trials))

    df = pd.DataFrame({
        'method': methods,
        'region_cov_gap_mean': region_cov_gap_means,
        'region_cov_gap_se': region_cov_gap_ses,
        'worst_region_cov_mean': worst_region_cov_means,
        'worst_region_cov_se': worst_region_cov_ses
    })

    return df

def return_correlation_plotting_metrics(dataset_name, model_name, features_dir, results_dir, res_fname, ood_scores_dir, score_fn, degree, method='conditional', seed=1, alpha=0.1, reciprocal=False):
    print("dataset: ", dataset_name, "score_fn: ", score_fn, ", degree: ", degree)

    # load precomputed features
    test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
    calib_features, calib_labels, calib_logits = None, None, None
    calib_ood_scores, test_ood_scores = None, None

    res_fname_seed = res_fname + f'_seed_{seed}'
    if dataset_name in ["imagenet", "places", "fitzpatrick17k"]:
        test_features, test_labels, test_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='test')
        test_labels, test_features, test_logits, _, calib_labels, calib_features, calib_logits, _ = split_test(test_labels, test_features, test_logits, split=0.5, seed=seed)
        calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name, seed=seed)
        if reciprocal:
            calib_ood_scores, test_ood_scores = 1 / (1 + calib_ood_scores.astype('float64')), 1 / (1 + test_ood_scores.astype('float64'))
    else:
        calib_features, calib_labels, calib_logits = load_precomputed_features(features_dir, dataset_name, model_name, split='val')
        calib_ood_scores, test_ood_scores = load_ood_scores(ood_scores_dir, dataset_name)
        if reciprocal:
            calib_ood_scores, test_ood_scores = 1 / (1 + calib_ood_scores.astype('float64')), 1 / (1 + test_ood_scores.astype('float64'))

    calib_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_calib_conf_score_{score_fn}.npy"))
    test_scores = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_{score_fn}.npy"))
    test_scores_cumsum_ordered = np.load(os.path.join(results_dir, f"{res_fname_seed}_test_conf_score_all_{score_fn}.npy"))

    # get confidence bin edges
    test_confidence = softmax(test_logits, axis=1)
    test_conf_scores = np.max(test_confidence, axis=1)

    ranks_array = get_ranks_array(test_logits, test_labels)
    trust_array = test_ood_scores

    coverages, set_sizes = None, None
    if method == 'split':
        coverages = np.load(os.path.join(results_dir, f"{res_fname_seed}_coverages_split.npy"))
        set_sizes = np.load(os.path.join(results_dir, f"{res_fname_seed}_set_sizes_split.npy"))

    elif method == 'conditional': 
        res_fname_degree = res_fname_seed + f'_degree_{degree}'
        coverages = np.load(os.path.join(results_dir, f"{res_fname_degree}_coverages_cond.npy"))
        set_sizes = np.load(os.path.join(results_dir, f"{res_fname_degree}_set_sizes_cond.npy"))

    elif method == 'naive':
        calib_confidence = softmax(calib_logits, axis=1)
        calib_conf_scores = np.max(calib_confidence, axis=1)
        calib_conf_bin_edges = get_bin_edges(calib_conf_scores, num_bins_conf, bin_strategy='uniform')
        calib_ood_bin_edges = get_bin_edges(calib_ood_scores, num_bins_ood, bin_strategy='quantile')
        calib_conf_ood_subgrouping = get_bin_indices(calib_conf_scores, calib_ood_scores, calib_conf_bin_edges, calib_ood_bin_edges, assignments_2d_to_1d)  
        binwise_qhat = fit_binwise_quantile(calib_scores, calib_conf_ood_subgrouping, alpha)
        coverages, set_sizes = compute_naive_sets(test_scores, test_scores_cumsum_ordered, conf_ood_subgrouping, binwise_qhat)
    
    return ranks_array, trust_array, set_sizes
