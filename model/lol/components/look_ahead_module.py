import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

from model.lol.patching.extract_tensor_patch import extract_tensor_patch


class LookAheadModule(nn.Module):

    def __init__(self, degrees=0, patch_size=64):
        super(LookAheadModule, self).__init__()
        self.degrees = Parameter(torch.tensor(degrees, dtype=torch.float32)).cuda()

        self.patch_size = patch_size
        self.c1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.r1 = nn.ReLU()
        self.p1 = nn.MaxPool2d(2, 2)

        self.c2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=1)
        self.b2 = nn.BatchNorm2d(32)
        self.r2 = nn.ReLU()
        self.p2 = nn.MaxPool2d(2, 2)

        self.c3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, padding=1)
        self.b3 = nn.BatchNorm2d(16)
        self.r3 = nn.ReLU()
        self.p3 = nn.MaxPool2d(2, 2)
        self.fully_connected = nn.Linear(576, 1)

        self.last = None

    def forward(self, img, base, angle, height):
        # y = self.l2(y)

        used_base = Variable(base, requires_grad=False)
        patching_angle = Variable(torch.add(self.degrees, angle), requires_grad=False)

        patch_parameters = torch.stack([used_base[0],  # x
                                        used_base[1],  # y
                                        torch.mul(torch.deg2rad(patching_angle), -1),  # angle
                                        height]).unsqueeze(0)
        patch_parameters = patch_parameters.cuda()
        patch = extract_tensor_patch(img, patch_parameters, size=self.patch_size)  # size
        #patch = patch.transpose(2, 3)
        self.last = patch[0]

        y = patch
        y = self.c1(y)
        y = self.b1(y)
        y = self.r1(y)
        y = self.p1(y)

        y = self.c2(y)
        y = self.b2(y)
        y = self.r2(y)
        y = self.p2(y)

        y = self.c3(y)
        y = self.b3(y)
        y = self.r3(y)
        y = self.p3(y)

        y = torch.flatten(y)
        y = self.fully_connected(y)
        y = torch.sigmoid(y)

        return y.squeeze(0)
