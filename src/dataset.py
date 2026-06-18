import os

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUFFIXES = (".jpg", ".jpeg", ".png")


class ISICDataset(Dataset):
    def __init__(self, frame, images_dir, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def _resolve(self, name):
        for suffix in SUFFIXES:
            path = os.path.join(self.images_dir, name + suffix)
            if os.path.isfile(path):
                return path
        return os.path.join(self.images_dir, name)

    def image_path(self, idx):
        return self._resolve(str(self.frame.iloc[idx]["image"]))

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(self._resolve(str(row["image"]))).convert("RGB")
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


train_transform = eval_transform


def load_table(csv_path):
    return pd.read_csv(csv_path)


def split_frame(frame, group_by="lesion_id", val_size=0.2, random_state=42):
    if group_by in frame.columns:
        groups = frame[group_by].fillna(frame["image"]).astype(str)
    else:
        groups = frame["image"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(frame, groups=groups))
    return frame.iloc[train_idx].copy(), frame.iloc[val_idx].copy()


def get_splits(cfg):
    frame = load_table(cfg["paths"]["ground_truth"])
    s = cfg["split"]
    return split_frame(frame, s["group_by"], s["val_size"], s["random_state"])


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
