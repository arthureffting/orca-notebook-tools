import argparse
import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.sol import sol_dataset
from model.sol.alignment_loss import alignment_loss
from model.sol.crop_transform import CropTransform
from model.sol.sol import StartOfLineFinder
from model.sol.sol_dataset import SolDataset
from model.sol import transformation_utils
from utils.dataset_parser import load_file_list, load_file_list_direct
from utils.wrapper import DatasetWrapper

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument("--dataset_folder", default="data/orcas/prepared/pages")
parser.add_argument("--base0", default=16)
parser.add_argument("--base1", default=16)
parser.add_argument("--alpha_alignment", default=0.1)
parser.add_argument("--alpha_backdrop", default=0.1)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--crop_prob_label", default=0.5)
parser.add_argument("--crop_size", default=256)
parser.add_argument("--rescale_range", default=[384, 640])
parser.add_argument("--batch_size", default=1)
parser.add_argument("--images_per_epoch", default=100)
parser.add_argument("--stop_after_no_improvement", default=20)
parser.add_argument("--max_epochs", default=1000)
parser.add_argument("--output", default="snapshots/sol/training")
args = parser.parse_args()

training_set_list_path = os.path.join(args.dataset_folder, "training.json")
training_set_list = load_file_list_direct(training_set_list_path)
train_dataset = SolDataset(training_set_list,
                           rescale_range=args.rescale_range,
                           transform=CropTransform({
                               "prob_label": args.crop_prob_label,
                               "crop_size": args.crop_size,
                           }))

train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=sol_dataset.collate)

batches_per_epoch = int(args.images_per_epoch / args.batch_size)
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

testing_set_list_path = os.path.join(args.dataset_folder, "testing.json")
testing_set_list = load_file_list_direct(testing_set_list_path)
test_dataset = SolDataset(testing_set_list,
                          rescale_range=args.rescale_range,
                          transform=None)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=sol_dataset.collate)

sol = StartOfLineFinder()
sol.cuda()
dtype = torch.cuda.FloatTensor

alpha_alignment = 0.1
alpha_backprop = 0.1

optimizer = torch.optim.Adam(sol.parameters(), lr=args.learning_rate)

lowest_loss = np.inf
cnt_since_last_improvement = 0
for epoch in range(1000):
    print("Epoch", epoch)

    sol.train()
    sum_loss = 0.0
    steps = 0.0

    for step_i, x in enumerate(train_dataloader):
        img = Variable(x['img'].type(dtype), requires_grad=False)
        sol_gt = None
        if x['sol_gt'] is not None:
            # This is needed because if sol_gt is None it means that there
            # no GT positions in the image. The alignment loss will handle,
            # it correctly as None
            sol_gt = Variable(x['sol_gt'].type(dtype), requires_grad=False)

        predictions = sol(img)
        predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
        loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        steps += 1

    print("Train Loss", sum_loss / steps)
    # print "Real Epoch", train_dataloader.epoch

    sol.eval()
    sum_loss = 0.0
    steps = 0.0

    for step_i, x in enumerate(test_dataloader):
        img = Variable(x['img'].type(dtype), requires_grad=False)
        sol_gt = Variable(x['sol_gt'].type(dtype), requires_grad=False)
        predictions = sol(img)
        predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)
        loss = alignment_loss(predictions, sol_gt, x['label_sizes'], alpha_alignment, alpha_backprop)
        sum_loss += loss.item()
        steps += 1

    cnt_since_last_improvement += 1
    if lowest_loss > sum_loss / steps:
        cnt_since_last_improvement = 0
        lowest_loss = sum_loss / steps
        print("Saving Best")
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        torch.save(sol.state_dict(), os.path.join(args.output, 'sol.pt'))

    print("Test Loss", sum_loss / steps, lowest_loss)
    print("")

    if cnt_since_last_improvement >= args.stop_after_no_improvement:
        break
