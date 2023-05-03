import torch.nn as nn 
import torch

### PoseNet loss. refer to <Geometric loss functions for camera pose regression with deep learning>
class HomoscedasticUncertaintyLoss(nn.Module):
    def __init__(self, norm=2) -> None:
        super(HomoscedasticUncertaintyLoss, self).__init__()
        self.norm = norm 

    def forward(self, output, label):
        l_p = torch.norm(output[:, :3]-label[:, :3], dim=1, p=self.norm)
        l_theta = torch.norm(output[:, 3:]-label[:, 3:], dim=1, p=self.norm)
        s_p = torch.log(torch.var(output[:, :3]))
        s_theta = torch.log(torch.var(output[:, 3:]))
        Loss = l_p*torch.exp(-s_p) + s_p + l_theta*torch.exp(-s_theta) + s_theta
        return Loss.mean()

if __name__ == '__main__':
    output, label = torch.rand([64, 6]), torch.rand([64, 6])
    criterion = HomoscedasticUncertaintyLoss(norm=1)
    loss = criterion(output, label)
    print(loss)