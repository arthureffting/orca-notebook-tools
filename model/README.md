
# Pre-processing

A pre-processing step is needed in order to extract individual lines from page-level annotations. It should generate individual folders for each page, containing a JSON file describing each line as well as the images for each extracted line.

```bash
├── data/iam/prepared/a01-000x
│   ├── a01-000x.json
│   ├── a01-000x-0.png
│   ├── a01-000x-1.png
│   ├── a01-000x-2.png
│   ├── ...
```

Each image JSON consists of an array of lines containing the following information, as exemplified in the JSON snippet below:

|Name | Description |
--- | --- |
|gt|Ground truth | 
|image_path| Path to the extracted line png|
|sol| Start-of-line position|
|steps| Array of steps that define the line's contour|

```json
[
 {
  "gt": "A MOVE to stop Mr. Gaitskell from nominating      ",
  "image_path": "data/iam/prepared/a01-000x/a01-000x-0.png",
  "sol": {
   "x0": 375.0,
   "x1": 375.0,
   "y0": 834.0,
   "y1": 834.0
  },
  "steps": [
   {
    "stop_confidence": 0.0,
    "base_point": [
     308.0670633544026,
     836.9969971632357
    ],
    "lower_point": [
     308.0670633544026,
     836.9969971632357
    ],
    "upper_point": [
     305.0700661911669,
     770.0640605176383
    ]
   }]
 }
]
```
A script for converting page annotations similar to the IAM dataset can be found [here](/utils/conversion/convert_iam_splits.py).

# Training

The training scripts expects a `dataset_folder` containing three JSON files: `training.json, testing.json and validation.json`
These files are arrays of two-dimensional arrays in the format `[<path to page JSON file>, <path to full-page PNG image>]`.

```json
[
 [
  "data/iam/prepared/a01-000x/a01-000x.json",
  "data/iam/pages/data/a01-000x.png"
 ]
]
```

### Line-Outliner training parameters

[The LOL training script](/model/lol/train.py) can be started using the following arguments

|Name | Description  | Default|
--- | --- | ---
|dataset_folder|Path to folder containing the data split JSON files| "dataset"|
|batch_size | Training batch size| 1|
|images_per_epoch| Amount of pages evaluated per epoch | 10000|
|testing_images_per_epoch| Can be used to limit the number of testing images per epoch for debugging| None|
|stop_after_no_improvement| Amount of epochs to stop training when no improvement happens | 20 |
|learning_rate| Learning rate of the model | 1.5e-5|
|tsa_size| Amount of images considered in the temporal spatial attention mechanism | 5 | 
|patch_ratio| How much bigger is each patch according to predicted handwriting size | 5 | 
| patch_size | Size in pixels of extracted patches | 64 | 
| min_height | Sets a limit for minimal handwriting size in pixels | 8 |
| name | Name used to store models | "training" |
| output | Output folder to store models | "snapshots/lol" |


### Start-of-line Finder training parameters

[The SOL training script](/model/sol/train.py) can be started using the following arguments

|Name | Description  | Default|
--- | --- | ---
|dataset_folder|Path to folder containing the data split JSON files| "dataset"|
|batch_size | Training batch size| 1|
|images_per_epoch| Amount of pages evaluated per epoch | 1000 |
|stop_after_no_improvement| Amount of epochs to stop training when no improvement happens | 20 |
|learning_rate| Learning rate of the model | 1e-4|
| name | Name used to store models | "training" |
| output | Output folder to store models | "snapshots/sol" |