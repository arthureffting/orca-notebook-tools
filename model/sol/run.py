import torch
from torch.autograd import Variable

from model.sol import transformation_utils
from model.sol.sol import StartOfLineFinder
import cv2
import numpy as np


def run_sol(model_path, img_path, rescale_range=[640, 640]):
    '''
    Runs the Start-of-Line module on an image and returns all predictions
    '''
    sol = StartOfLineFinder(path="snapshots/sol/training/sol.pt")
    sol.cuda()
    dtype = torch.cuda.FloatTensor

    org_img = cv2.imread(img_path)
    target_dim1 = int(np.random.uniform(rescale_range[0], rescale_range[1]))
    s = target_dim1 / float(org_img.shape[1])
    target_dim0 = int(org_img.shape[0] / float(org_img.shape[1]) * target_dim1)
    org_img = cv2.resize(org_img, (target_dim1, target_dim0), interpolation=cv2.INTER_CUBIC)

    img = org_img.transpose([2, 1, 0])[None, ...]
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    img = img / 128.0 - 1.0
    img = Variable(img.type(dtype), requires_grad=False).cuda()

    predictions = sol(img)
    predictions = transformation_utils.pt_xyrs_2_xyxy(predictions)

    # [confidence, x0, y0, x1, y1]
    return predictions


def filter_predictions_by_threshold(predictions, threshold):
    results = []
    for batch in predictions:
        for prediction in batch:
            if prediction[0].item() >= threshold:
                results.append(prediction[1:5])
    return results


results = run_sol("snapshots/sol/training/sol.pt", "test_image.jpg")
accepted = filter_predictions_by_threshold(results, 0.1)

