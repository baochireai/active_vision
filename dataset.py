from torch.utils.data import DataLoader, Dataset
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImitationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None,istrain=True):
        self.data_dir = os.path.join(data_dir, 'train' if istrain else 'val')
        self.features_path = os.path.join(self.data_dir, 'img')
        self.labels_path = os.path.join(self.data_dir, 'label')
        self.transforms = transforms
        self.label_ids = os.listdir(self.labels_path)
        print(self.label_ids)
    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, idx):
        label_id = self.label_ids[idx]
        label_path = os.path.join(self.labels_path, label_id)

        """读取label对应的RGB-D图像"""
        rgb_path = os.path.join(self.features_path, '{}.jpg'.format(label_id[:-4]))
        deep_path=os.path.join(self.features_path, '{}-depth.jpg'.format(label_id[:-4]))
        # Read the image and annotation file
        rgb = Image.open(rgb_path).convert('RGB')
        depth=Image.open(deep_path)
        # 将图像转换为NumPy数组
        rgb_array = np.asarray(rgb)#(1544,2064,3)
        depth_array = np.asarray(depth)#(1544,2064)  根据量程归一化
        # 确保图像形状相同
        rgbd=[]
        if rgb_array.shape[:2] == depth_array.shape:
            # 将深度通道转换为与RGB图像相同的形状
            depth_channel = np.expand_dims(depth_array, axis=2)#[1544,2064,1]
            # 合并RGB和深度通道
            rgbd = np.concatenate((rgb_array, depth_channel), axis=2)#(1544,2064,4)

        if self.transforms is not None:#对比度
            rgbd = self.transforms(rgbd)

        #读取label
        with open(label_path, 'r') as f:
            label=f.readline()
            label = label.strip().split()
            label =[float(x) for x in label]
            label=np.asarray(label)

        return rgbd,label


def load_dataset(data_dir,batch_size):
    train_iter = torch.utils.data.DataLoader(ImitationDataset(data_dir,istrain=True),batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(ImitationDataset(data_dir,istrain=False),batch_size)
    return train_iter,val_iter

if __name__ == '__main__':
    print(os.getcwd())
    data_dir='E:\\docs\\train_detection\\code\\active_vision\\Dataset'
    batch_size=3
    train_iter,_=load_dataset(data_dir,batch_size)
    X,y=next(iter(train_iter))
    print('X.shape:',X.shape)#2*W*H*4
    print('y.shape:',y.shape)