# Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores

This is the official code repository for [Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores](https://arxiv.org/abs/2501.10139). The repository includes code for reproducing our experiments to construct prediction sets that improve conditional coverage with respect to model confidence and the trust score. We also provide a notebook that allows you to easily perform the conditional conformal procedure for any custom function class you define (see details in [Custom Function Class](#custom-function-class)).

## Setup

We provide an `environment.yaml` file with all dependecies. You can set up a Python 3.10 conda environment and install dependencies by running:
```
conda env create -f environment.yaml
conda activate conformal-trust
```

### Downloading Data

Please download the datasets from original sources and store in `data/` folder. This should point to the `train/` and `val/` folders for ImageNet and `places365/` for the Places dataset. When you run the code (`run_condconf.py`) for the first time, pre-trained models are downloaded and loaded, and features are extracted for the training and test sets.

### Computing Trust scores

We compute `trust_score` using the code provided by the original paper (https://github.com/google/TrustScore). For faster computation, we use faiss library. You can download precomputed trust scores by running
```
bash download_trust_scores.sh
```
This will create a `trust_scores` folder and download the trust scores for all datasets. To generate the trust scores from scratch, you can run  `utils/compute_trust_scores.sh`.

## Reproducing Results

We provide `run_condconf.sh` which is the script for running all experiments. The `dataset_name` can be set accordingly from {imagenet, imagenet_lt, places, places_lt, fitzpatrick17k}. Similarly, the `model_name` can be set. See `scripts/` for the default script for all datasets.

### Evaluation

We provide `evaluation.ipynb` notebook in `notebooks/` that provides the code to aggregate results over all seeds and methods.

## Custom Function Class 

We provide `conditional_custom_function_class.ipynb` in `notebooks/` to easily use the conditional conformal prediction algorithm for any custom function class you define. Refer to the notebook for simple instructions on how to define your function class and run the algorithm.

### Citing this work

```bibtex
@article{kaur2025conformal,
  title={Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores}, 
  author={Kaur, Jivat Neet and Jordan, Michael I. and Alaa, Ahmed}, 
  journal={arXiv preprint arXiv:2501.10139},
  year={2025}
}
```


### Acknowledgements

We adapt the `conditionalconformal` package from [Conformal Prediction with Conditional Guarantees](https://github.com/jjcherian/conditional-conformal) for our conditional conformal prediction procedure. For processing the data and model embeddings, we adapt the code from [Beyond Confidence: Reliable Models Should Also Consider Atypicality](https://github.com/mertyg/beyond-confidence-atypicality/tree/main).


