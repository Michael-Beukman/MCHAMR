import numpy as np
from torchvision.models import resnet18
import torch
import torch.nn as nn
device = 'cuda'
model = resnet18('IMAGENET1K_V1').to(device)
model.eval()

convs = nn.Sequential(*list(model.children())[:-2])

def resnet_dist(a: np.ndarray, b: np.ndarray) -> float:
    ""
    if a.shape[1] == 1: a = a[:, 0]
    if b.shape[1] == 1: b = b[:, 0]
    a_gpu = torch.tensor(a).to(device).float()
    b_gpu = torch.tensor(b).to(device).float()
    
    a_gpu = torch.repeat_interleave(a_gpu[None], 3, axis=0).unsqueeze(0)
    b_gpu = torch.repeat_interleave(b_gpu[None], 3, axis=0).unsqueeze(0)
    
    a_feat = convs(a_gpu).flatten()
    b_feat = convs(b_gpu).flatten()
    
    return 1 - torch.clip(torch.cosine_similarity(a_feat, b_feat, dim=0), 0, 1).item()