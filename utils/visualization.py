import os
import numpy as np
import matplotlib.pyplot as plt

def plot_classes_accuracy(class_accs,class_names= None, save_path=None):
    num_classes = len(class_accs)
    plt.figure(figsize=(14, 6))

    if class_names is not None:
        plt.bar(class_names, class_accs)
        plt.xticks(rotation=45, ha="right")
    else:
        plt.bar(range(num_classes), class_accs)
        plt.xlabel("Class Index")

    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_confusion_matrix(cm, class_names=None, normalize=True, save_path=None):
    
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()

    if class_names is not None:
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.yticks(range(len(class_names)), class_names)
    else:
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()
