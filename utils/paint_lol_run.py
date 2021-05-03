import torch
from shapely.geometry import Point, LineString

from models.lol.lol_3 import LineOutlinerTsa
from utils.files import create_folders
from utils.geometry import angle_between_points, get_new_point
from utils.painter import Painter


def paint_model_run(model_path, img_path, dataloader, destination="screenshots/run.png"):
    dtype = torch.cuda.FloatTensor

    painter = Painter(path=img_path)

    lol = LineOutlinerTsa(path=model_path)
    lol.cuda()

    for index, x in enumerate(dataloader):
        x = x[0]

        belongs = img_path == x["img_path"]

        if not belongs:
            continue

        img = x['img'].type(dtype)[None, ...]
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
            if predicted_step[4][0].item() > 0.85:
                break
            predictions.append(predicted_step.clone().detach().cpu())

        upper_points = [Point(step[0][0].item(), step[0][1].item()) for step in predictions]
        ground_truth_baseline_steps = [Point(step[1][0].item(), step[1][1].item()) for step in predictions]
        lower_points = [Point(step[2][0].item(), step[2][1].item()) for step in predictions]
        confidences = [step[4][0].item() for step in predictions]
        # for index, step in enumerate(ground_truth_baseline_steps[:-1]):
        #     upper_height, lower_height = base_height * predicted_steps[index][2].item(), \
        #                                  base_height * predicted_steps[index][3].item()
        #     next_step = ground_truth_baseline_steps[index + 1]
        #     angle_between_them = angle_between_points(step, next_step)
        #     upper_point = get_new_point(step, angle_between_them - 90, upper_height)
        #     lower_point = get_new_point(step, angle_between_them + 90, lower_height)
        #     painter.draw_line([upper_point, lower_point], line_width=1, color=(0, 0, 0, 0.5))
        #     upper_points.append(upper_point)
        #     lower_points.append(lower_point)
        #
        # painter.draw_line(upper_points, line_width=2, color=(1, 0, 1, 1))
        # painter.draw_line(lower_points, line_width=2, color=(1, 0, 1, 1))
        painter.draw_line(lower_points, line_width=2, color=(1, 0, 1, 0.5))
        painter.draw_line(ground_truth_baseline_steps, line_width=2, color=(0, 0, 1, 0.5))
        painter.draw_line(upper_points, line_width=2, color=(1, 0, 1, 0.5))

        for i in range(len(confidences) - 1):
            confidence = confidences[i]
            painter.draw_area([upper_points[i], upper_points[i + 1], lower_points[i + 1], lower_points[i]],
                              fill_color=(confidence, 1 - confidence, 0, 0.05 + confidence))
        sol = {
            "upper_point": ground_truth[0][0],
            "base_point": ground_truth[0][1],
            "angle": ground_truth[0][3][0],
        }

        sol_upper = Point(sol["upper_point"][0].item(), sol["upper_point"][1].item())
        sol_lower = Point(sol["base_point"][0].item(), sol["base_point"][1].item())

        painter.draw_line([sol_lower, sol_upper], color=(0, 1, 0, 1), line_width=4)
        painter.draw_point(sol_lower, color=(0, 1, 0, 1), radius=4)
        painter.draw_point(sol_upper, color=(0, 1, 0, 1), radius=4)

    create_folders(destination)
    painter.save(destination)
