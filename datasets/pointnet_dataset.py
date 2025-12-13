import os
import torch
from torch.utils.data import Dataset
import numpy as np 
import open3d as o3d

class PointNetDataset(Dataset):
    def __init__(self, root_dir, split="train", num_points=1024):
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i,cls in enumerate(self.classes)}
        self.files = []     # [file_path, label]

        for cls in self.classes:
            path = os.path.join(root_dir, cls, split)
            for f in os.listdir(path):
                if f.endswith('.off'):
                    self.files.append((os.path.join(path, f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):

        file_path, label = self.files[idx]
        mesh = o3d.io.read_triangle_mesh(file_path)

        #sample a mesh
        pcd = mesh.sample_points_uniformly(number_of_points=self.num_points)
        pts = np.asarray(pcd.points, dtype=np.float32)

        # normalize points
        pts -= np.mean(pts, axis=0)
        pts /= np.max(np.linalg.norm(pts, axis=1))

        return torch.from_numpy(pts), torch.tensor(label)