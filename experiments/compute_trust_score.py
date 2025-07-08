"""
This code is used to compute trust scores for each model.
"""

import os
import os.path as osp
import numpy as np
import faiss
import torch
from argparse import ArgumentParser
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from scipy.special import softmax

parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="imagenet")
parser.add_argument("--features_dir", type=str, default="/path/to/features")
parser.add_argument("--normalize", type=bool, default=True)
parser.add_argument("--num_neighbors", type=int, default=10)
parser.add_argument("--distance_measure", type=str, default="L2")
parser.add_argument('--model_name', default='resnet50')

args = parser.parse_args()

def split_test(lbls, acts, logits, split=0.2, seed=1):
    """Split data into calibration and test sets."""
    indices = np.arange(len(lbls))
    lbls_1, lbls_2, acts_1, acts_2, logits_1, logits_2, train_idx, test_idx = train_test_split(
        lbls, acts, logits, indices, test_size=1-split, random_state=seed, stratify=lbls
    )    
    return lbls_1, acts_1, logits_1, train_idx, lbls_2, acts_2, logits_2, test_idx

def compute_trust_scores(train_features, train_labels, val_features, val_labels, val_logits, test_features, test_labels, test_logits):
    """Compute trust scores for validation and test sets."""
    # Initialize FAISS indices for each class
    dim = train_features.shape[1]
    n_labels = np.max(train_labels) + 1
    index_arr = [None] * n_labels
    for label in range(n_labels):
        X_to_use = train_features[np.where(train_labels == label)[0]]
        index = faiss.IndexFlatL2(dim) 
        index.add(X_to_use)
        index_arr[label] = index

    # Compute predictions
    val_preds = np.argmax(val_logits, axis=-1)
    test_preds = np.argmax(test_logits, axis=-1)

    # Compute distances to each class
    val_d = np.tile(None, (val_features.shape[0], n_labels))
    test_d = np.tile(None, (test_features.shape[0], n_labels))
    for label_idx in range(n_labels):
        val_d[:, label_idx] = index_arr[label_idx].search(val_features, 2)[0][:, -1]
        test_d[:, label_idx] = index_arr[label_idx].search(test_features, 2)[0][:, -1]

    # Compute trust scores for validation set
    sorted_d = np.sort(val_d, axis=1)
    d_to_pred = val_d[range(val_d.shape[0]), val_preds]
    d_to_closest_not_pred = np.where(
        sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1]
    )
    min_dist = 1e-12
    val_trust = d_to_closest_not_pred / (d_to_pred + min_dist)

    # Compute trust scores for test set
    sorted_d = np.sort(test_d, axis=1)
    d_to_pred = test_d[range(test_d.shape[0]), test_preds]
    d_to_closest_not_pred = np.where(
        sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1]
    )
    test_trust = d_to_closest_not_pred / (d_to_pred + min_dist)

    return val_trust, test_trust

# Create output directory
output_dir = '../trust_scores'
os.makedirs(output_dir, exist_ok=True)

if args.dataset_name in ["imagenet", "places", "fitzpatrick17k"]:
    # Load test data
    acts_file = os.path.join(args.features_dir, f"{args.model_name}_{args.dataset_name}_test_acts.npy")
    lbls_file = os.path.join(args.features_dir, f"{args.model_name}_{args.dataset_name}_test_lbls.npy")
    logits_file = os.path.join(args.features_dir, f"{args.model_name}_{args.dataset_name}_test_logits.npy")
    test_labels_orig = np.load(lbls_file)
    test_features_orig = np.load(acts_file)
    test_logits_orig = np.load(logits_file)

    # Load training data
    acts_file = os.path.join(args.features_dir, f"{args.model_name}_{args.dataset_name}_train_acts.npy")
    lbls_file = os.path.join(args.features_dir, f"{args.model_name}_{args.dataset_name}_train_lbls.npy")
    logits_file = os.path.join(args.features_dir, f"{args.model_name}_{args.dataset_name}_train_logits.npy")
    train_labels = np.load(lbls_file)
    train_features = np.load(acts_file)
    train_logits = np.load(logits_file)

    if args.normalize:
        test_features_orig = test_features_orig / np.linalg.norm(test_features_orig, axis=1, keepdims=True)
        train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
    print("Features shape:", test_features_orig.shape)

    # Compute trust scores for each seed
    for seed in range(1, 11):
        print('seed:', seed)
        test_labels, test_features, test_logits, _, calib_labels, calib_features, calib_logits, _ = split_test(
            test_labels_orig, test_features_orig, test_logits_orig, split=0.5, seed=seed
        )

        val_trust, test_trust = compute_trust_scores(
            train_features, train_labels,
            calib_features, calib_labels, calib_logits,
            test_features, test_labels, test_logits
        )

        # Save trust scores
        np.save(os.path.join(output_dir, f"val_{args.dataset_name}_trust_seed{seed}.npy"), val_trust)
        np.save(os.path.join(output_dir, f"test_{args.dataset_name}_trust_seed{seed}.npy"), test_trust)
        print(f"Saved trust scores for seed {seed}")

else:  # imagenet_lt and places_lt
    # Load train/val/test data
    splits = ['train', 'val', 'test']
    data = {}
    for split in splits:
        acts_file = os.path.join(args.features_dir, f"{args.model_name}_{args.dataset_name}_{split}_acts.npy")
        lbls_file = os.path.join(args.features_dir, f"{args.model_name}_{args.dataset_name}_{split}_lbls.npy")
        logits_file = os.path.join(args.features_dir, f"{args.model_name}_{args.dataset_name}_{split}_logits.npy")
        data[split] = {
            'labels': np.load(lbls_file),
            'features': np.load(acts_file),
            'logits': np.load(logits_file)
        }

    if args.normalize:
        for split in splits:
            data[split]['features'] = data[split]['features'] / np.linalg.norm(data[split]['features'], axis=1, keepdims=True)
    print("Features shape:", data['test']['features'].shape)

    # Compute trust scores
    val_trust, test_trust = compute_trust_scores(
        data['train']['features'], data['train']['labels'],
        data['val']['features'], data['val']['labels'], data['val']['logits'],
        data['test']['features'], data['test']['labels'], data['test']['logits']
    )

    # Save trust scores
    np.save(os.path.join(output_dir, f"val_{args.dataset_name}_trust.npy"), val_trust)
    np.save(os.path.join(output_dir, f"test_{args.dataset_name}_trust.npy"), test_trust)
    print("Saved trust scores")



