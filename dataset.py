from torch.utils.data import DataLoader, Dataset
import os
import torch
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np
from pathlib import Path 
from typing import Union, Callable, Optional, List
import cv2 
import open3d as o3d

MAX_DEPTH = 800 # 拍摄范围 0-60cm，有发现超过600的深度值，改成800
IMAGE_DIR, LABELS_DIR = 'images', 'labels' 

class ImitationDataset(Dataset):
    # k = torch.tensor(([2.28, 1.6, 1.78, 1., 2.28, 2.]))
    # b = torch.tensor(([-0.11, -0.16, -0.444,  0.2, 0.11, 0.]))
    # k = torch.ones(6)
    # b = torch.zeros(6)

    def __init__(
            self, roots: List[Union[Path, str]], transforms_fun: Optional[Callable], imgsz: Optional[List[int]]=[224,224],
            img_channel: str='rgbd', select_label_index: List[int]=[0,1,2,3,4,5] 
        ) -> None:

        self.roots = roots
        self.data_list = self._bulid_data_list()
        self.transforms_fun = transforms_fun
        self.imgsz = imgsz
        if img_channel not in ['rgb', 'depth', 'rgbd']:
            raise ValueError("channel must be in ['rgb', 'depth', 'rgbd'].")
        self.img_channel = img_channel
        self.select_label_index = select_label_index

    def _bulid_data_list(self,):
        data_list = []
        for root in self.roots:
            root = Path(root)
            images_dir, labels_dir = root / IMAGE_DIR, root / LABELS_DIR
            n = 0 
            for txt_f in labels_dir.iterdir():
                if txt_f.suffix in ['.txt']:
                    jpg_f = images_dir / txt_f.with_suffix('.jpg').name
                    pcd_f = images_dir / txt_f.with_suffix('.pcd').name
                    if jpg_f.exists() and pcd_f.exists():
                        data_list.append((str(jpg_f), str(pcd_f), str(txt_f)))
                        n += 1
                    else:
                        print(f"{txt_f} does not have a corresponding jpg/pcd file.")
            print(f">>>{root} has {n} images.")
        return data_list
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        jpg_f, pcd_f, txt_f = self.data_list[idx]
        
        if self.img_channel == 'rgbd':
            rgb_img = Image.open(jpg_f).convert('RGB')
            pcd = o3d.io.read_point_cloud(pcd_f)
            depth_img = self.pcd2depth(pcd, rgb_img.size[::-1])
            rgb_img = rgb_img.resize(self.imgsz)
            depth_img = cv2.resize(depth_img, self.imgsz)
            rgb_img = self.transforms_fun(rgb_img)
            depth_img = transforms.ToTensor()((depth_img/MAX_DEPTH).astype(np.float32))
            data = torch.cat([rgb_img, depth_img])
        
        elif self.img_channel == 'rgb':
            rgb_img = Image.open(jpg_f).convert('RGB').resize(self.imgsz)
            data = self.transforms_fun(rgb_img)
        
        elif self.img_channel == 'depth':
            rgb_img = Image.open(jpg_f)
            pcd = o3d.io.read_point_cloud(str(pcd_f))
            depth_img = self.pcd2depth(pcd, rgb_img.size[::-1])
            depth_img = cv2.resize(depth_img, self.imgsz)
            data = transforms.ToTensor()((depth_img/MAX_DEPTH).astype(np.float32))
        
        else:
            raise NotImplementedError

        with open(txt_f, 'r') as f:
            label = f.readline()
            label = label.strip().split()
            label = np.array([float(x) for x in label], dtype=np.float32)
            label = torch.from_numpy(label)

        # return data, label[self.select_label_index]*self.k + self.b
        return data, label[self.select_label_index]

    @staticmethod
    def pcd2depth(pcd, img_shape):
        pcd_array = np.asarray(pcd.points)[:,2].reshape(img_shape)
        mask = np.isnan(pcd_array)
        pcd_array[mask] = 0
        return pcd_array



def load_dataset(root,batch_size):
    transforms_fun = transforms.Compose([
        # transforms.ColorJitter(0.3, 0.3, 0.3),
        # transforms.GaussianBlur(5),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    train_iter = DataLoader(ImitationDataset(root, transforms_fun), batch_size, shuffle=True)
    return train_iter

if __name__ == '__main__':
    print(os.getcwd())
    roots=['/media/datum/wangjl/data/active_vision_dataset/datasets4.26', '/media/datum/wangjl/data/active_vision_dataset/datasets4.26']
    batch_size=32
    transforms_fun = transforms.Compose([
        # transforms.ColorJitter(0.3, 0.3, 0.3),
        # transforms.GaussianBlur(5),
        transforms.ToTensor(),
        # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    dataset = ImitationDataset(roots, transforms_fun, img_channel='rgbd', select_label_index=[2])
    dataloader = DataLoader(dataset, batch_size=32)
    for i, (x, y) in enumerate(dataloader):
        print(x.shape, y.shape)
        img2d = transforms.ToPILImage()(x[1, :3, ...])
        img3d = transforms.ToPILImage()(x[1, 3, ...])
        img2d.save('img2d.png')
        img3d.save('img3d.png')
        break
