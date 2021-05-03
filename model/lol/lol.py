import os
import random

import torch
from shapely.geometry import Point
from torch import nn
from torch.autograd import Variable
from model.lol.components.look_ahead_module import LookAheadModule
from model.lol.components.lol_conv_lstm import MemoryLayer
from model.lol.components.lol_convolutions import PreLstmConvolutions, PostLstmConvolutions
from model.lol.patching.extract_tensor_patch import extract_tensor_patch
from model.lol.components.tsa import TemporalAttension

class Stepper(nn.Module):

    def __init__(self, path=None, patch_ratio=5, tsa_size=3, patch_size=64):
        super(Stepper, self).__init__()
        self.tsa_size = tsa_size
        self.patch_size = patch_size
        self.patch_ratio = patch_ratio
        self.tsa = TemporalAttension(3)
        self.initial_convolutions = PreLstmConvolutions().cuda()
        self.memory_layer = MemoryLayer().cuda()
        self.final_convolutions = PostLstmConvolutions().cuda()
        self.fully_connected = nn.Linear(512, 8)
        self.fully_connected.bias.requires_grad = False
        size = self.patch_size / self.patch_ratio
        self.fully_connected.bias.data[0] = size
        self.fully_connected.bias.data[3] = size
        self.fully_connected.bias.data[4] = -size
        self.fully_connected.bias.data[5] = size
        self.fully_connected.bias.data[6] = 0

    def forward(self, patches):
        input = Variable(torch.stack([p for p in patches[-self.tsa_size:]]).unsqueeze(0).cuda(), requires_grad=False)
        y = self.tsa(input)
        # y = y[:, 1:, :, :, :]
        after_tsa_copy = y.clone().detach().cpu()
        y = y.squeeze(0)
        y = self.initial_convolutions(y)
        y = y.unsqueeze(0)
        y = self.memory_layer(y)
        y = y.unsqueeze(0)
        y = self.final_convolutions(y)
        y = y.unsqueeze(0)
        y = torch.flatten(y, 1)
        y = torch.flatten(y, 0)
        y = self.fully_connected(y)
        y[7] = torch.sigmoid(y[7])
        return y


class LineOutlinerTsa(nn.Module):

    def __init__(self, path=None, patch_ratio=5, tsa_size=3, min_height=32, patch_size=64):
        super(LineOutlinerTsa, self).__init__()
        self.tsa_size = tsa_size
        self.min_height = min_height
        self.patch_size = patch_size
        self.patch_ratio = patch_ratio
        self.stepper = Stepper(patch_ratio=patch_ratio, tsa_size=tsa_size, patch_size=patch_size)
        self.up_measurer = LookAheadModule(-90)
        self.down_measurer = LookAheadModule(90)
        self.stop_measurer = LookAheadModule(0)

        if path is not None and os.path.exists(path):
            state = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(state)
            self.eval()
        elif path is not None:
            print("\nCould not find path", path, "\n")

        self.visualization = None

    def forward(self,
                img,
                steps,
                sol_index=0,
                disturb_sol=True,
                min_height=64,
                height_disturbance=0.5,
                angle_disturbance=45,
                translate_disturbance=15):

        self.visualization = None
        img = Variable(img, requires_grad=False).cuda()

        # Take last 5 positions before current step

        steps_used = steps[max(0, sol_index - self.tsa_size):sol_index + 1].cuda()
        heights = [Point(s[0][0].item(), s[0][1].item()).distance(Point(s[1][0].item(), s[1][1].item())) for s in
                   steps_used]
        heights = [max(min_height, h) for h in heights]
        heights = [torch.tensor(h, dtype=torch.float32).cuda() for h in heights]

        if any([h == 0 for h in heights]):
            return None

        if disturb_sol:
            base_disturbance = torch.tensor(
                [0.0, 0.0] if not disturb_sol else [random.uniform(-translate_disturbance, translate_disturbance),
                                                    random.uniform(-translate_disturbance, translate_disturbance)],
                dtype=torch.float32).cuda()
            angle_disturbance = torch.tensor(
                0.0 if not disturb_sol else random.uniform(-angle_disturbance, angle_disturbance),
                dtype=torch.float32).cuda()
            steps_used[-1][3][0] = torch.add(steps_used[-1][3][0], angle_disturbance)
            steps_used[-1][1] = torch.add(steps_used[-1][1], base_disturbance)
            heights[-1] = heights[-1] * (1 + random.uniform(-height_disturbance, height_disturbance))

        current_angle = steps_used[-1][3][0]
        current_scale = (self.patch_ratio * heights[-1] / self.patch_size).cuda()
        current_base = steps_used[-1][1]

        patches = [extract_tensor_patch(img, torch.stack([step[1][0],  # x
                                                          step[1][1],  # y
                                                          torch.mul(torch.deg2rad(step[3][0]), -1).cuda(),
                                                          heights[index]]).unsqueeze(0).cuda(),
                                        size=self.patch_size).squeeze(0)
                   for index, step in enumerate(steps_used)]

        y = self.stepper(patches)

        # size = input[-1, :, :, :].shape[1] / self.patch_ratio

        base_rotation_matrix = torch.stack(
            [torch.stack([torch.cos(torch.deg2rad(current_angle)), -1.0 * torch.sin(torch.deg2rad(current_angle))]),
             torch.stack(
                 [1.0 * torch.sin(torch.deg2rad(current_angle)), torch.cos(torch.deg2rad(current_angle))])]).cuda()

        scale_matrix = torch.stack(
            [torch.stack([current_scale, torch.tensor(0., dtype=torch.float32, requires_grad=True).cuda()]),
             torch.stack([torch.tensor(0., dtype=torch.float32, requires_grad=True).cuda(), current_scale])]).cuda()

        base_point = torch.stack([y[0], y[1]])
        base_point = torch.matmul(base_point, base_rotation_matrix.t())
        base_point = torch.matmul(base_point, scale_matrix)

        # Assuming a sigmoid indicating how much % of the patch is the upper height
        #
        upper_point = torch.stack([y[3], y[4]])
        upper_point = torch.matmul(upper_point, base_rotation_matrix.t())
        upper_point = torch.matmul(upper_point, scale_matrix)
        upper_point = torch.add(upper_point, current_base)

        lower_point = torch.stack([y[5], y[6]])
        lower_point = torch.matmul(lower_point, base_rotation_matrix.t())
        lower_point = torch.matmul(lower_point, scale_matrix)
        lower_point = torch.add(lower_point, current_base)

        # current_height = torch.mul(current_height, y[3])
        current_base = torch.add(current_base, base_point)
        current_angle = torch.add(current_angle, y[2])

        # up = self.up_measurer(img, current_base, current_angle, base_height)
        # down = self.down_measurer(img, current_base, current_angle, base_height)
        # stop = self.stop_measurer(img, current_base, current_angle, base_height)

        del patches

        return torch.stack([
            upper_point,
            current_base,
            lower_point,
            torch.stack([current_angle, torch.tensor(0.0, dtype=torch.float32).cuda()]).cuda(),
            torch.stack([y[7], torch.tensor(0.0, dtype=torch.float32).cuda()]).cuda(),
        ])
