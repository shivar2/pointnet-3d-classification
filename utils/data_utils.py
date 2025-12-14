from torch.utils.data import Subset
import numpy as np

def split_data(dataset, val_ratio=0.15, seed=42):
    np.random.seed(seed)
    targets = np.array([label for _, label in dataset])
    num_classes = len(np.unique(targets))

    train_indices = []
    val_indices = []

    for c in range(num_classes):
        class_idx = np.where(targets == c)[0]
        np.random.shuffle(class_idx)
        n_val = int(len(class_idx) * val_ratio)
        val_indices.extend(class_idx[:n_val])
        train_indices.extend(class_idx[n_val:])


    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset
