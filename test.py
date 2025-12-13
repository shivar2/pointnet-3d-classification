import os, sys
import torch
from torch.utils.data import DataLoader
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasets.pointnet_dataset import PointNetDataset
from models.pointnet_cls import PointNetCls
from utils.visualization import plot_classes_accuracy, plot_confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Config
batch_size = 64
num_classes = 40
model_path = "best_pointnet.pth"

# Load Data
dataset = PointNetDataset("data/ModelNet40", split="test")
test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print("Data has loaded!")

# Load Model
model = PointNetCls(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model is ready!")

# ---------- Testing ----------
correct = 0
total = 0

cls_correct = np.zeros(num_classes)
cls_total = np.zeros(num_classes)

confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

with torch.no_grad():
    for points, labels in test_data:
        points = points.to(device)
        labels = labels.to(device)

        preds, _ = model(points)
        predicted = preds.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion[t.item(), p.item()] += 1


# Result
total = confusion.sum()
correct = np.trace(confusion)

overall_acc = 100 * correct / total
print(f"\nOverall Test Accuracy: {overall_acc:.2f}%\n")

print("Per-class Accuracy:")

cls_total = confusion.sum(axis=1)
cls_correct = np.diag(confusion)
class_accs = []

for i in range(num_classes):
    if cls_total[i] > 0:
        acc = 100 * cls_correct[i] / cls_total[i]
    else:
        acc = 0.0

    class_accs.append(acc)
    print(f"Class {i:02d}: {acc:.2f}%")

print(sum(cls_total) == total)
print(sum(cls_correct) == correct)

plot_classes_accuracy(class_accs,
                       class_names=dataset.classes, 
                       save_path="results/per_class_accuracy.png")

plot_confusion_matrix(
    confusion,
    class_names=dataset.classes,
    normalize=True,
    save_path="results/confusion_matrix.png"
)