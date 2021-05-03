import os
from pathlib import Path
from torch.utils.data import DataLoader
from models.lol import lol_dataset
from models.lol.lol_3 import LineOutlinerTsa
from models.lol.lol_dataset import LolDataset
from utils.dataset_parser import load_file_list_direct
from utils.paint_lol_run import paint_model_run

### This validation runs the inverse dice coefficient to evaluate segmentation accuracy only!


model_path = os.path.join("snapshots", "lol", "training", "best.pt")
validation_path = os.path.join("data", "validation.json")

validation_list = load_file_list_direct(validation_path)
validation_set = LolDataset(validation_list)
validation_loader = DataLoader(validation_set,
                               batch_size=1,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=lol_dataset.collate)

lol = LineOutlinerTsa(path=model_path)
lol.cuda()
sum_loss = 0.0
steps = 0

saved = []

stop_threshold = 0.5

for index, x in enumerate(validation_loader):
    if x is None:
        continue
    x = x[0]

    stem = Path(x["img_path"]).stem

    destination = os.path.join("screenshots", "validation", stem, "full.png")
    if x["img_path"] not in saved and not os.path.exists(destination):
        print(str(index), destination)
        paint_model_run(model_path, x["img_path"], validation_loader, destination=destination)
        saved.append(x["img_path"])

    continue

    img = Variable(x['img'].type(torch.cuda.FloatTensor), requires_grad=False)[None, ...]
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
        if predicted_step[4][0].item() > stop_threshold:
            break
        predictions.append(predicted_step.clone().detach().cpu())

    inverse_dice_coefficient = 1.0

    try:
        predicted_points = steps_to_points(predictions)

        training_image = TrainingImage({
            "index": str(index),
            "filename": x["img_path"],
            "lines": [{
                "index": str(index),
                "text": "",
                "steps": [{
                    "upper_point": [step["upper_point"].x, step["upper_point"].y],
                    "base_point": [step["base_point"].x, step["base_point"].y],
                    "lower_point": [step["lower_point"].x, step["lower_point"].y],
                } for step in predicted_points]
            }]
        })

        for line in training_image.lines:
            line_filename = os.path.join("screenshots", "validation", stem,
                                         stem + "-" + str(x["line_idx"]) + ".png")
            create_folders(line_filename)
            dewarped_line = Dewarper2(line.steps).dewarped(line.masked())
            cv2.imwrite(line_filename, dewarped_line)

        desired_point = steps_to_points(ground_truth)
        predicted_polygon = points_to_polygon(predicted_points)
        desired_polygon = points_to_polygon(desired_point)
        if predicted_polygon.intersects(desired_polygon):
            intersection = predicted_polygon.intersection(desired_polygon)
            inverse_dice_coefficient = 1.0 - float(
                2.0 * intersection.area / float(predicted_polygon.area + desired_polygon.area))
    except Exception as e:
        print(e)
        inverse_dice_coefficient = 1.0

    sum_loss += inverse_dice_coefficient
    steps += 1
    sys.stdout.write(
        "\r[Validating] " + str(1 + index) + "/" + str(len(validation_loader)) + " | loss: " + str(
            round(sum_loss / steps, 3)))

print("\nValidation dice: ", str(sum_loss / steps))
