from model import ActiveDecisionModel
import torch
from torchvision import transforms
from PIL import Image
from dataset import ImitationDataset
import cv2
import numpy as np
import open3d as o3d

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
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    rgb_img = transforms_fun(rgb_img)
    depth_img = transforms.ToTensor()((depth_img/MAX_DEPTH).astype(np.float32))

    rgbd_img = torch.cat([rgb_img, depth_img]).to(device).unsqueeze(0)

    print(rgbd_img.shape)
    output = model(rgbd_img)
    return output


if __name__ == "__main__":
    rgb_img = Image.open("/media/datum/wangjl/data/active_vision_dataset/train2/images/0_1681882162.029513.jpg")
    pcd = o3d.io.read_point_cloud("/media/datum/wangjl/data/active_vision_dataset/train2/images/0_1681882162.029513.pcd")
    
    model = ActiveDecisionModel(4)
    ckpt = torch.load("result/run1/last.pth")
    model.load_state_dict(ckpt)

    device = "cuda:0"

    output = infer(model, rgb_img, pcd, device)
    print(output)

    with open("/media/datum/wangjl/data/active_vision_dataset/train2/labels/0_1681882162.029513.txt", "r") as f:
        lines = f.readlines()
    
    print("ground thruth: ", lines)