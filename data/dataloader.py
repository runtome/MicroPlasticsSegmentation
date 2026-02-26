"""
DataLoader factory for MicroPlastics dataset.
"""
import json
from typing import Optional, Callable

from torch.utils.data import DataLoader

from .dataset import MicroPlasticsDataset, collate_fn
from .transforms import get_train_transforms, get_val_transforms


def build_dataloader(
    split: str,
    images_dir: str,
    annotation_path: str,
    splits_file: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: int = 640,
    fold: Optional[int] = None,
    transforms: Optional[Callable] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Build a DataLoader for a given split.

    Args:
        split: 'train', 'val', or 'test'
        fold: If not None, use this fold index for 5-fold CV (0-indexed)
    """
    with open(splits_file) as f:
        splits = json.load(f)

    if fold is not None:
        fold_data = splits["folds"][fold]
        if split == "train":
            file_names = fold_data["train"]
        elif split == "val":
            file_names = fold_data["val"]
        else:
            file_names = splits["test"]
    else:
        file_names = splits[split]

    if transforms is None:
        if split == "train":
            transforms = get_train_transforms(image_size)
        else:
            transforms = get_val_transforms(image_size)

    dataset = MicroPlasticsDataset(
        images_dir=images_dir,
        annotation_path=annotation_path,
        file_names=file_names,
        transforms=transforms,
        image_size=image_size,
    )

    shuffle = split == "train"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=shuffle,
    )
    return loader
