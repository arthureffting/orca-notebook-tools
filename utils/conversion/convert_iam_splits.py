import os

from model.lol.domain.training_image import TrainingImage
from utils.conversion.concave import run_transformation_approach
from utils.conversion.img_xml_pair import ImageXmlPair
from utils.conversion.stepper import to_steps
from utils.files import create_folders, save_to_json
from utils.line_augmentations import LineAugmentation
from utils.line_dewarper import Dewarper2
from utils.read_splits import Split

# Folder containing the standard splits for the IAM data
database_splits_folder = "data/iam/splits"

# Folder containing the IAM data
database_original_folder = "data/iam"

# Folder in which to store the transformed data
target_folder = "prepared/data/iam"

splits = Split.read_iam_data(database_splits_folder)
for dataset in splits.set_map:
    dataset_json_data_path = os.path.join(target_folder, "pages", dataset + ".json")
    dataset_json_data = []
    pairs = []
    training_indexes = set([line.page_index for line in splits.sets[dataset].items])
    for page_index in training_indexes:
        img_path = os.path.join(database_original_folder, "pages", page_index + ".png")
        xml_path = os.path.join(database_original_folder, "xml", page_index + ".xml")
        assert os.path.exists(img_path) and os.path.isfile(img_path)
        assert os.path.exists(xml_path) and os.path.isfile(xml_path)
        # index, base_folder, img_filename, xml_filename
        pair = ImageXmlPair(page_index, database_original_folder, img_path, xml_path)
        pairs.append(pair)

    # For each pair
    # Create a folder with its name
    # Extract lines
    # Create JSON with line information
    # Add to data json data
    for pair in pairs:
        folder_path = os.path.join(target_folder, "pages", "data", pair.index)
        create_folders(os.path.join(folder_path, "something.txt"))
        image_json_path = os.path.join(target_folder, "pages", "data", pair.index, pair.index + ".json")
        image_json = []
        pair.set_height_threshold(0.1)
        image_data = run_transformation_approach(pair, alpha=0.0025)
        steps = to_steps({pair.index: image_data}, [pair])
        image_steps = steps["images"][0]
        dataset_json_data.append([image_json_path, image_steps["filename"]])
        image = TrainingImage(image_steps)

        for line in image.lines:
            LineAugmentation.normalize(line)
            LineAugmentation.extend_backwards(line)
            LineAugmentation.extend(line, by=6, size_decay=0.9, confidence_decay=0.825)
            LineAugmentation.enforce_minimum_height(line, minimum_height=32)
            LineAugmentation.prevent_wrong_start(line, angle_threshold=30.0)
            line_filename = os.path.join(folder_path, pair.index + "-" + str(line.index) + ".png")
            dewarped_line = Dewarper2(line.steps).dewarped(line.masked())
            cv2.imwrite(line_filename, dewarped_line)
            image_json.append({
                "gt": line.text,
                "image_path": line_filename,
                "steps": [{
                    "upper_point": [step.upper_point.x, step.upper_point.y],
                    "base_point": [step.base_point.x, step.base_point.y],
                    "lower_point": [step.lower_point.x, step.lower_point.y],
                    "stop_confidence": step.stop_confidence,
                } for step in line.steps],
                "sol": {
                    "x0": line.steps[1].upper_point.x,
                    "x1": line.steps[1].lower_point.x,
                    "y0": line.steps[1].lower_point.y,
                    "y1": line.steps[1].lower_point.y,
                }
            })
        save_to_json(image_json, image_json_path)

    save_to_json(dataset_json_data, dataset_json_data_path)