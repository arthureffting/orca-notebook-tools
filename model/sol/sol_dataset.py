import math

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

from utils import safe_load, augmentation


def collate(batch):
    batch_size = len(batch)
    imgs = []
    scales = []
    paths = []
    label_sizes = []
    for b in batch:
        if b is None:
            continue
        scales.append(b["scale"])
        paths.append(b["img_path"])
        imgs.append(b["img"])
        if b['sol_gt'] is None:
            label_sizes.append(0)
        else:
            label_sizes.append(b['sol_gt'].size(1))
    if len(imgs) == 0:
        return None
    batch_size = len(imgs)
    largest_label = max(label_sizes)
    labels = None
    if largest_label != 0:
        labels = torch.zeros(batch_size, largest_label, 4)
        for i, b in enumerate(batch):
            if label_sizes[i] == 0:
                continue
            labels[i, :label_sizes[i]] = b['sol_gt']
    imgs = torch.cat(imgs)
    return {
        'sol_gt': labels,
        'img': imgs,
        "label_sizes": label_sizes,
        "img_paths": paths,
        "scales": scales
    }


# CNT = 0
class SolDataset(Dataset):
    def __init__(self, set_list, rescale_range=None, transform=None, random_subset_size=None):

        self.rescale_range = rescale_range

        self.ids = set_list
        self.ids.sort()

        new_ids = []
        for json_path, img_path in self.ids:

            gt_json = safe_load.json_state(json_path)
            if gt_json is None:
                continue
            failed = False
            for j, gt_item in enumerate(gt_json):
                if 'sol' not in gt_item:
                    failed = True
                    break

            if failed:
                continue
            new_ids.append([
                json_path, img_path
            ])

        self.ids = new_ids

        if random_subset_size is not None:
            self.ids = random.sample(self.ids, min(random_subset_size, len(self.ids)))
        print("[SOL count]", len(self.ids))
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        gt_json_path, img_path = self.ids[idx]

        gt_json = safe_load.json_state(gt_json_path)
        if gt_json is None:
            return None

        org_img = cv2.imread(img_path)
        target_dim1 = int(np.random.uniform(self.rescale_range[0], self.rescale_range[1]))

        s = target_dim1 / float(org_img.shape[1])
        target_dim0 = int(org_img.shape[0] / float(org_img.shape[1]) * target_dim1)
        org_img = cv2.resize(org_img, (target_dim1, target_dim0), interpolation=cv2.INTER_CUBIC)

        gt = np.zeros((1, len(gt_json), 4), dtype=np.float32)

        positions = []
        positions_xy = []

        for j, gt_item in enumerate(gt_json):
            if 'sol' not in gt_item:
                continue

            x0 = gt_item['sol']['x0'] * s
            x1 = gt_item['sol']['x1'] * s
            y0 = gt_item['sol']['y0'] * s
            y1 = gt_item['sol']['y1'] * s

            positions_xy.append([(torch.Tensor([[x1, x0], [y1, y0]]))])
            dx = x0 - x1
            dy = y0 - y1
            d = math.sqrt(dx ** 2 + dy ** 2)
            mx = (x0 + x1) / 2.0
            my = (y0 + y1) / 2.0
            # Not sure if this is right...
            theta = -math.atan2(dx, -dy)
            positions.append([torch.Tensor([mx, my, theta, d / 2, 1.0])])

            gt[:, j, 0] = x0
            gt[:, j, 1] = y0
            gt[:, j, 2] = x1
            gt[:, j, 3] = y1

        if self.transform is not None:
            out = self.transform({
                "img": org_img,
                "sol_gt": gt
            })
            org_img = out['img']
            gt = out['sol_gt']
            org_img = augmentation.apply_random_color_rotation(org_img)
            org_img = augmentation.apply_tensmeyer_brightness(org_img)

        img = org_img.transpose([2, 1, 0])[None, ...]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        img = img / 128.0 - 1.0

        if gt.shape[1] == 0:
            gt = None
        else:
            gt = torch.from_numpy(gt)

        return {
            "scale": s,
            "img_path": img_path,
            "img": img,
            "sol_gt": gt,
            "lf_xyrs": positions,
            "lf_xyxy": positions_xy,
        }
