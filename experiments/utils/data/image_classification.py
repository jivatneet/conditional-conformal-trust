# Code is taken from https://github.com/mertyg/beyond-confidence-atypicality and modified. The code available for use under the MIT license.
# @article{yuksekgonul2023beyond,
#   title={Beyond Confidence: Reliable Models Should Also Consider Atypicality},
#   author={Yuksekgonul, Mert and Zhang, Linjun and Zou, James and Guestrin, Carlos},
#   journal={arXiv preprint arXiv:2305.18262},
#   year={2023}
# }

import os
import subprocess
import pandas as pd
import skimage
from skimage import io
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                impath = line.split()[0]
                self.img_path.append(os.path.join(root, impath))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    
def get_image_dataset(dataset_name, preprocess=None):
    """Careful: This function returns 
    a) train_dataset, test_dataset for balanced classification
    b) train_dataset, val_dataset, test_dataset for imbalanced classification (since earlier papers already proposed used splits for val/test.)
    """
    if dataset_name == "imagenet":
        imagenet_dir = os.environ.get("DATA_DIR", None)
        if imagenet_dir is None:
            raise ValueError("DATA_DIR not set. Please set your environment variable to the path of the ImageNet dataset, or modify here.")
        train_dataset = datasets.ImageNet(root=imagenet_dir, split = 'train', transform=preprocess)
        train_dataset.dataset_name = "imagenet_train"
        test_dataset = datasets.ImageNet(root=imagenet_dir, split = 'val', transform=preprocess)
        test_dataset.dataset_name = "imagenet_test"
        return train_dataset, test_dataset
    
    elif dataset_name == "imagenet_lt":
        imagenet_dir = os.environ.get("DATA_DIR", None)
        if imagenet_dir is None:
            raise ValueError("DATA_DIR not set. Please set your environment variable to the path of the ImageNet dataset, or modify here.")

        cache_dir = os.environ.get("CACHE_DIR", "~/.cache")
        urls = {
            "train": "https://github.com/facebookresearch/classifier-balancing/raw/main/data/ImageNet_LT/ImageNet_LT_train.txt",
            "val": "https://github.com/facebookresearch/classifier-balancing/raw/main/data/ImageNet_LT/ImageNet_LT_val.txt",
            "test": "https://github.com/facebookresearch/classifier-balancing/raw/main/data/ImageNet_LT/ImageNet_LT_test.txt" 
        }
        
        train_txt = os.path.join(cache_dir, "imagenet_lt", "train.txt")
        val_txt = os.path.join(cache_dir, "imagenet_lt", "val.txt")
        test_txt = os.path.join(cache_dir, "imagenet_lt", "test.txt")
        if not os.path.exists(val_txt):
            os.makedirs(os.path.dirname(test_txt), exist_ok=True)
            subprocess.call(["wget", "-O", test_txt, urls["test"]])
            subprocess.call(["wget", "-O", val_txt, urls["val"]])
            subprocess.call(["wget", "-O", train_txt, urls["train"]])
            
        # Needs pointing to the right place, i.e. path with ImageNet `train` and `val` folders
        train_dataset = LT_Dataset(root=imagenet_dir, txt=train_txt, transform=preprocess)
        val_dataset = LT_Dataset(root=imagenet_dir, txt=val_txt, transform=preprocess)
        test_dataset = LT_Dataset(root=imagenet_dir, txt=test_txt, transform=preprocess)

        train_dataset.dataset_name = "imagenet_lt_train"
        val_dataset.dataset_name = "imagenet_lt_val"
        test_dataset.dataset_name = "imagenet_lt_test"
        return train_dataset, val_dataset, test_dataset
    
    elif dataset_name == "places":
        places_dir = os.environ.get("DATA_DIR", None)
        if places_dir is None:
            raise ValueError("DATA_DIR not set. Please set your environment variable to the path of the Places dataset, or modify here.")
        
        # Needs pointing to the right place, i.e. path with Places `train` and `val` folders
        places_dir = os.path.join(places_dir, "places_lt", "places365_standard")
        train_dataset = datasets.ImageFolder(os.path.join(places_dir, "train"), transform=preprocess)
        train_dataset.dataset_name = "places_train"
        test_dataset = datasets.ImageFolder(os.path.join(places_dir, "val"), transform=preprocess)
        test_dataset.dataset_name = "places_test"
        return train_dataset, test_dataset

    elif dataset_name == "places_lt":
        places_dir = os.environ.get("DATA_DIR", None)
        if places_dir is None:
            raise ValueError("DATA_DIR not set. Please set your environment variable to the path of the Places dataset, or modify here.")
        
        cache_dir = os.environ.get("CACHE_DIR", "~/.cache")
        urls = {
            "train": "https://github.com/facebookresearch/classifier-balancing/raw/main/data/Places_LT/Places_LT_train.txt",
            "val": "https://github.com/facebookresearch/classifier-balancing/raw/main/data/Places_LT/Places_LT_val.txt",
            "test": "https://github.com/facebookresearch/classifier-balancing/raw/main/data/Places_LT/Places_LT_test.txt"
        }
        
        train_txt = os.path.join(cache_dir, "places_lt", "train.txt")
        val_txt = os.path.join(cache_dir, "places_lt", "val.txt")
        test_txt = os.path.join(cache_dir, "places_lt", "test.txt")
        if not os.path.exists(test_txt):
            os.makedirs(os.path.dirname(test_txt), exist_ok=True)
            subprocess.call(["wget", "-O", test_txt, urls["test"]])
            subprocess.call(["wget", "-O", val_txt, urls["val"]])
            subprocess.call(["wget", "-O", train_txt, urls["train"]])

        # Needs pointing to the right place, i.e. path with Places `train` and `val` folders
        places_dir = os.path.join(places_dir, "places_lt", "places365_standard")
        train_dataset = LT_Dataset(root=places_dir, txt=train_txt, transform=preprocess)
        val_dataset = LT_Dataset(root=places_dir, txt=val_txt, transform=preprocess)
        test_dataset = LT_Dataset(root=places_dir, txt=test_txt, transform=preprocess)

        train_dataset.dataset_name = "places_lt_train"
        val_dataset.dataset_name = "places_lt_val"
        test_dataset.dataset_name = "places_lt_test"
        print('train', len(train_dataset), 'val', len(val_dataset))
        return train_dataset, val_dataset, test_dataset
    
    elif dataset_name == "fitzpatrick17k":
        data_dir = os.environ.get("DATA_DIR", None)
        if data_dir is None:
            raise ValueError("DATA_DIR not set. Please set your environment variable to the path of the Fitzpatrick 17k dataset, or modify here.")

        # Needs pointing to the right place, i.e. train and test csv files and data/finalfitz17k
        fitzpatrick_dir = os.path.join(data_dir, "fitzpatrick")
        data_dir = os.path.join(fitzpatrick_dir, "data", "finalfitz17k")
        train_csv = os.path.join(fitzpatrick_dir, "fitzpatrick17k", "train.csv")
        test_csv = os.path.join(fitzpatrick_dir, "fitzpatrick17k", "test.csv")
        train_dataset = SkinDataset(train_csv, data_dir, transform=preprocess)
        train_dataset.dataset_name = "fitzpatrick17k_train"
        test_dataset = SkinDataset(test_csv, data_dir, transform=preprocess)
        test_dataset.dataset_name = "fitzpatrick17k_test"
        return train_dataset, test_dataset

    else:
        raise NotImplementedError(f"Unknown dataset {dataset_name}")