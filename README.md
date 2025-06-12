# Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores

This is the official code repository of [Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores](https://arxiv.org/abs/2501.10139). 

## Setup

We modify the `conditionalconformal` package (https://github.com/jjcherian/conditional-conformal) for our conditional conformal prediction procedure. The modified package can be set up as:

```bash
$ pip install -e .
```

To install the rest of the requirements, we recommend using a conda environment or virtual environment.
```
pip install -r requirements.txt
```


## Reproducing Results

We provide `run_condconf.sh` which is the script for running all experiments. The `dataset_name` can be set accordingly from [imagenet, imagenet_lt, places, places_lt, fitzpatrick17k]. Similarly, the `model_name` can be set. See `/scripts` for the script for all datasets.

### Computing Trust scores

We compute `trust_score` using the code provided by the original paper (https://github.com/google/TrustScore). For faster computation, we use faiss library.

### Evaluation

We provide `evaluation.ipynb` notebook that provides the code to aggregate results over all seeds and methods.

### Citing this work

```
@article{kaur2025conformal,
      title={Conformal Prediction Sets with Improved Conditional Coverage using Trust Scores}, 
      author={Kaur, Jivat Neet and Jordan, Michael I. and Alaa, Ahmed}, 
      journal={arXiv preprint arXiv:2501.10139},
      year={2025}
}
```


### Acknowledgements

