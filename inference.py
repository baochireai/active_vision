'''
Date: 2023-04-11 13:54:33
FirstEditors: pystar360 pystar360@py-star.com
LastEditors: pystar360 pystar360@py-star.com
LastEditTime: 2023-05-18 15:27:46
FilePath: /active_vision_decision_model/inference.py
'''
from model import ActiveDecisionModel
import torch
from torchvision import transforms
from PIL import Image
from dataset import ImitationDataset
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path 

MAX_DEPTH = 800
@torch.no_grad()
def infer(model, rgb_img, pcd, device):
    model.eval()
    model.to(device)
    shape = rgb_img.size[::-1]
    depth_img = ImitationDataset.pcd2depth(pcd, shape)

    imgsz = (224, 224)
    depth_img = cv2.resize(depth_img, imgsz)

    transforms_fun = transforms.Compose([
        transforms.Resize(imgsz),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    rgb_img = transforms_fun(rgb_img)
    depth_img = transforms.ToTensor()((depth_img/MAX_DEPTH).astype(np.float32))

    rgbd_img = torch.cat([rgb_img, depth_img]).to(device).unsqueeze(0)

    # print(rgbd_img.shape)
    output = model(rgbd_img)
    output = output.cpu() / ImitationDataset.scale
    return output.numpy()


if __name__ == "__main__":
    model = ActiveDecisionModel(4, output_dim=6)
    ckpt = torch.load("result/L1Loss/last.pth", map_location='cpu')
    model.load_state_dict(ckpt)

    device = "cuda:2"
    root = Path("/media/datum/wangjl/data/active_vision_dataset/datasets5.16/val")
    img_root = root / 'images'
    label_root = root / 'labels'

    loss = []
    
    for i in label_root.iterdir():
        imgf = img_root / i.with_suffix('.jpg').name
        pcdf = img_root / i.with_suffix('.pcd').name
        rgb_img = Image.open(imgf)
        pcd = o3d.io.read_point_cloud(str(pcdf))

        output = infer(model, rgb_img, pcd, device)
        # output = output.cpu().numpy()[0] 
        # print()

        with open(i, "r") as f:
            lines = f.readline()
        ground_thruth = np.array([float(f"{float(i):.4f}") for i in lines.strip().split(' ')])
        loss.append(abs(output-ground_thruth))
        print(f"output, ground thruth:  {output}, {ground_thruth}")

    print('average MAELoss = ', sum(loss)/len(loss))