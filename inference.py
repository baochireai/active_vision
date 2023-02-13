import model
import torch
model=model.ActiveDecisionModel(3)
x=torch.rand(size=(1,3,2064,1544))

for layer in model.network:
    x=layer(x)
    print("shape:",x.shape)