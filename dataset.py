from torch.utils.data import DataLoader, Dataset
import os
import torch
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np
from pathlib import Path 
from typing import Union, Callable, Optional, List
import cv2 


MAX_DEPTH = 4000 # 最大拍摄距离4m

class ImitationDataset(Dataset):
    def __init__(self, root: Union[Path, str], transforms_fun: Optional[Callable], imgsz: Optional[List[int]]=[224,224]) -> None:
        
        self.imgsz = imgsz

        self.root = Path(root) 
        self.images_root, self.labels_root = self.root / 'images', self.root / 'labels'
        self.transforms_fun = transforms_fun

        self.label_files = os.listdir(self.labels_root)
        print(self.label_files)
    
    
    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        
        
        label_f = self.labels_root / self.label_files[idx]
        file_name = label_f.stem
        rgb_f = self.images_root / f"{file_name}.jpg"
        depth_f = self.images_root / f"{file_name.replace('_', '_depth_')}.jpg"

        # Read the image and annotation file
        rgb_img = Image.open(rgb_f).convert('RGB')
        depth_img = cv2.imread(f"{depth_f}", cv2.IMREAD_ANYDEPTH)
        assert rgb_img.size == depth_img.shape[:2][::-1], f"{rgb_f} shape != {depth_f} shape"
        
        rgb_img = rgb_img.resize(self.imgsz)
        depth_img = cv2.resize(depth_img, self.imgsz)
        rgb_img = self.transforms_fun(rgb_img)
        depth_img = transforms.ToTensor()((depth_img/MAX_DEPTH).astype(np.float32))

        rgbd_img = torch.cat([rgb_img, depth_img])

        with open(label_f, 'r') as f:
            label = f.readline()
            label = label.strip().split()
            label = np.array([float(x) for x in label], dtype=np.float32)
            label = torch.from_numpy(label)

        return rgbd_img, label


def load_dataset(root,batch_size):
    transforms_fun = transforms.Compose([
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.GaussianBlur(5),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    train_iter = DataLoader(ImitationDataset(root, transforms_fun), batch_size, shuffle=True)
    return train_iter

if __name__ == '__main__':
    print(os.getcwd())
    root='Dataset/train'
    batch_size=3
    train_iter=load_dataset(root,batch_size)
    X,y=next(iter(train_iter))
    print('X.shape:',X.shape) 
    print('y.shape:',y.shape)