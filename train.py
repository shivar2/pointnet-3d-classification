import os, sys
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np 

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


from datasets.pointnet_dataset import PointNetDataset
from models.pointnet_cls import PointNetCls
from utils.early_stopping import EarlyStopping

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU Available: ", torch.cuda.get_device_name(0))
print("Cuda version: ", torch.version.cuda)

# Set seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Consts
val_ratio = 0.15
batch_size = 64

epochs=50
lr=0.001

# Read and Split dataset
dataset = PointNetDataset("data/ModelNet40", split="train")

train_size = int((1-val_ratio) * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Model and Optimizer
model = PointNetCls(num_classes=40).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
early_stopper = EarlyStopping(patience=8, min_delta=1e-4)


# Training Loop

for epoch in range(epochs):
    
    # ---------- Training ----------
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for points, labels in train_loader:

        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        pred, trans_feature = model(points)

        #class loss
        loss_cls = criterion(pred, labels)

        #reg loss
        B, k, _ = trans_feature.size()
        I = torch.eye(k, device=trans_feature.device).unsqueeze(0).repeat(B, 1, 1)
        diff = torch.bmm(trans_feature, trans_feature.transpose(2, 1)) - I
        loss_reg = torch.norm(diff, dim=(1, 2)).mean()

        #total loss
        loss = loss_cls + 0.001 * loss_reg

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = pred.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # ---------- Validation ----------
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for points, labels in val_loader:
            points = points.to(device)
            labels = labels.to(device)

            pred, trans_feat = model(points)

            #class loss
            loss_cls = criterion(pred, labels)

            #reg loss
            B, k, _ = trans_feat.size()
            I = torch.eye(k, device=trans_feat.device).unsqueeze(0).repeat(B, 1, 1)
            diff = torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - I
            loss_reg = torch.norm(diff, dim=(1, 2)).mean()

            #total loss
            loss = loss_cls + 0.001 * loss_reg

            val_loss += loss.item()

            predicted = pred.argmax(dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    val_loss = val_loss / len(val_loader)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    early_stopper(val_loss, model)
    if early_stopper.early_stop:
        print("Early stopping triggered!")
        break


# ---------- Save Model ----------
torch.save(early_stopper.best_model_state, "best_pointnet.pth")
print("Model saved as pointnet_model.pth")