import numpy as np
import torch
from torch.autograd import Variable

# input = tensor([x,y,angle,size])
from models.lol.patching.gridder import Gridder


def extract_tensor_patch(img_tensor, input, size=64):
    gridder = Gridder(size)
    img = gridder.get_grid(img_tensor, input)

    # img = grid[0].clone().detach().cpu().numpy()
    # # img = img.transpose()[None, ...]
    # img = img.astype(np.float32)
    # img = torch.from_numpy(img)
    # img = img / 128.0 - 1.0

    return img
