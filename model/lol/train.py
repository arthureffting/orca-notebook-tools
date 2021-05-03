import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.lol.lol import LineOutlinerTsa
from model.lol import dataset
from model.lol.dataset import LolDataset
from utils.dataset_parser import load_file_list_direct
from utils.dice_utils import steps_to_points, points_to_polygon
from utils.files import create_folders, save_to_json
from utils.wrapper import DatasetWrapper

parser = argparse.ArgumentParser(description='Prepare data for training')

# Folder containing sets
parser.add_argument("--dataset_folder", default="dataset")

# Learning parameters
parser.add_argument("--batch_size", default=1)
parser.add_argument("--images_per_epoch", default=500)
parser.add_argument("--testing_images_per_epoch", default=50)
parser.add_argument("--stop_after_no_improvement", default=200)
parser.add_argument("--learning_rate", default=0.000015)

# Patching
parser.add_argument("--tsa_size", default=5)
parser.add_argument("--patch_ratio", default=5)
parser.add_argument("--patch_size", default=64)
parser.add_argument("--min_height", default=8)

# Training techniques
parser.add_argument("--name", default="lol_training_patch_64")
parser.add_argument("--reset-threshold", default=128)
parser.add_argument("--max_steps", default=6)
parser.add_argument("--random-sol", default=True)
parser.add_argument("--output", default="snapshots/lol")

args = parser.parse_args()

### SAVE ARGUMENTS
args_filename = os.path.join(args.output, args.name, 'args.json')
create_folders(args_filename)
with open(args_filename, 'w') as fp:
    json.dump(args.__dict__, fp, indent=4)

training_set_list_path = os.path.join(args.dataset_folder, "training.json")
training_set_list = load_file_list_direct(training_set_list_path)
train_dataset = LolDataset(training_set_list, augmentation=True)
train_dataloader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=dataset.collate)
batches_per_epoch = int(int(args.images_per_epoch) / args.batch_size)
train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list_path = os.path.join(args.dataset_folder, "testing.json")
test_set_list = load_file_list_direct(test_set_list_path)
test_dataset = LolDataset(test_set_list)
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=dataset.collate)
test_dataloader = test_dataloader if args.testing_images_per_epoch is None else DatasetWrapper(test_dataloader,
                                                                                               int(
                                                                                                   args.testing_images_per_epoch))

validation_path = os.path.join(args.dataset_folder, "validation.json")
validation_list = load_file_list_direct(validation_path)
validation_set = LolDataset(validation_list[0:1])
validation_loader = DataLoader(validation_set,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=dataset.collate)

print("Loaded datasets")

lol = LineOutlinerTsa(tsa_size=args.tsa_size,
                      patch_size=args.patch_size,
                      min_height=args.min_height,
                      patch_ratio=args.patch_ratio)
lol.cuda()

optimizer = torch.optim.Adam(lol.parameters(), lr=float(args.learning_rate))

dtype = torch.cuda.FloatTensor

best_loss = np.inf
cnt_since_last_improvement = 0
all_epoch_data = []

def loss_function(predicted, desired):
    upper = torch.nn.MSELoss()(predicted[:, 0], desired[:, 0].cuda())
    baseline = torch.nn.MSELoss()(predicted[:, 1], desired[:, 1].cuda())
    lower = torch.nn.MSELoss()(predicted[:, 2], desired[:, 2].cuda())
    angle = torch.nn.MSELoss()(predicted[:, 3, 0], desired[:, 3, 0].cuda())
    confidence = torch.nn.MSELoss()(predicted[:, 4, 0], desired[:, 4, 0].cuda())
    return [upper, baseline, lower, angle, confidence]

for epoch in range(1000):

    epoch_data = {
        "epoch": epoch,
    }

    print("[Epoch", epoch, "]")

    sum_loss = 0.0
    steps = 0.0
    total_steps_ran = 0
    lol.train()

    for index, x in enumerate(train_dataloader):
        # Only single batch for now
        x = x[0]
        if x is None:
            continue
        img = Variable(x['img'].type(dtype), requires_grad=False)[None, ...]
        ground_truth = x["steps"]

        predictions = []

        for i in range(len(ground_truth) - 1):
            predicted_step = lol(img,
                                 ground_truth,
                                 sol_index=i,
                                 disturb_sol=True)
            predictions.append(predicted_step)
        loss = loss_function(torch.stack(predictions), ground_truth[1:])
        optimizer.zero_grad()
        torch.autograd.backward(loss)
        optimizer.step()

        sum_loss += sum(loss).item()
        steps += 1

        sys.stdout.write("\r[Training] " + str(1 + index) + "/" + str(len(train_dataloader)) + " | loss: " + str(
            round(sum_loss / steps, 3)) + " | " + "avg steps: " + str(round(total_steps_ran / steps, 3)))

    print()

    epoch_data["train"] = {
        "loss": sum_loss / steps,
        "avg_steps": total_steps_ran / steps,
    }

    sum_loss = 0.0
    steps = 0.0

    lol.eval()

    # Save epoch snapshot using some validation image
    model_path = os.path.join(args.output, args.name, 'last.pt')
    screenshot_path = os.path.join(args.output, args.name, "screenshots", str(epoch) + ".png")
    create_folders(screenshot_path)
    torch.save(lol.state_dict(), model_path)
    time.sleep(1)
    # paint_model_run(model_path, validation_loader, destination=screenshot_path)

    with torch.no_grad():
        for index, x in enumerate(test_dataloader):
            if x is None:
                continue
            x = x[0]
            img = Variable(x['img'].type(dtype), requires_grad=False)[None, ...]
            ground_truth = x["steps"]
            predictions = [ground_truth[1]]
            for i in range(60):
                torch.cuda.empty_cache()
                predicted_step = lol(img,
                                     torch.stack(predictions),
                                     sol_index=len(predictions) - 1,
                                     disturb_sol=False)
                if predicted_step is None:
                    break
                if torch.dist(predicted_step[0], predicted_step[1]).item() > 160:
                    break
                if predicted_step[4][0].item() > 0.5:
                    break
                predictions.append(predicted_step.clone().detach().cpu())

            inverse_dice_coefficient = 1.0

            try:
                predicted_points = steps_to_points(predictions)
                desired_point = steps_to_points(ground_truth)
                predicted_polygon = points_to_polygon(predicted_points)
                desired_polygon = points_to_polygon(desired_point)
                if predicted_polygon.intersects(desired_polygon):
                    intersection = predicted_polygon.intersection(desired_polygon)
                    inverse_dice_coefficient = 1.0 - float(
                        2.0 * intersection.area / float(predicted_polygon.area + desired_polygon.area))
            except Exception as e:
                inverse_dice_coefficient = 1.0

            sum_loss += inverse_dice_coefficient
            steps += 1
            sys.stdout.write(
                "\r[Testing] " + str(1 + index) + "/" + str(len(test_dataloader)) + " | loss: " + str(
                    round(sum_loss / steps, 3)) + " | " + "avg steps: " + str(round(total_steps_ran / steps, 3)))

    cnt_since_last_improvement += 1

    epoch_data["test"] = {
        "loss": float(sum_loss / steps)
    }
    all_epoch_data.append(epoch_data)

    fig, ax1 = plt.subplots()
    t = range(len(all_epoch_data))
    ax1.plot(t, [epoch["train"]["loss"] for epoch in all_epoch_data], color="dodgerblue")
    ax1.set_xlabel('Epoch')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('MSE Loss', color='dodgerblue')
    ax1.tick_params('y', colors='dodgerblue')
    max_mse = max([epoch["train"]["loss"] for epoch in all_epoch_data])
    ax1.set_ylim([-0.1 * max_mse, 1.1 * max_mse])

    ax2 = ax1.twinx()
    ax2.plot(t, [epoch["test"]["loss"] for epoch in all_epoch_data], color="orangered", dashes=[2, 2])
    ax2.set_ylabel('Inverse Dice Coefficient', color='orangered')
    ax2.tick_params('y', colors='orangered')
    ax2.set_ylim([-0.1, 1.1])

    # fig.tight_layout()
    plt.savefig(os.path.join(args.output, args.name, "plot.png"))

    loss_used = (sum_loss / steps)

    if loss_used < best_loss:
        cnt_since_last_improvement = 0
        best_loss = loss_used
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        torch.save(lol.state_dict(), os.path.join(args.output, args.name, 'best.pt'))
        print("\n[New best achieved]")
    else:
        print("\n[Current best]: ", round(best_loss, 3))

    epoch_json_path = os.path.join(args.output, args.name, "epochs", str(epoch) + ".json")
    create_folders(epoch_json_path)
    save_to_json(epoch_data, epoch_json_path)

    print()

    if cnt_since_last_improvement >= args.stop_after_no_improvement:
        break
