
# Training

The training script expects a `dataset_folder` containing three JSON files: `training.json, testing.json and validation.json`
These files are arrays of two-dimensional arrays in the format `[<path to page JSON file>, <path to page PNG image>]`.

```json
[
 [
  "data/iam/pages/a01-000x/a01-000x.json",
  "data/iam/pages/a01-000x/a01-000x-0.png"
 ]
]
```

Each image JSON should contain page information in the following format:

```json

[
 {
  "gt": "A MOVE to stop Mr. Gaitskell from nominating      ",
  "image_path": "data/iam/pages/a01-000x/a01-000x-0.png",
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

A script for converting page annotations similar to the IAM dataset can be found under [todo](TODO).


### Arguments

[The training script](/model/lol/train.py) can be started using the following arguments

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
