# 2601-Advanced-Data-Science-Final-Project

This repository provides the training and evaluation code used for the final project:

**Explainable Collision Warnings using the V-JEPA 2.1 Foundation Model**

The project investigates whether a V-JEPA 2.1 video foundation model can be fine-tuned for dashcam-based collision prediction while also predicting structured semantic warning fields that explain why a warning is triggered.

## Repository Scope

This repository is intended to provide the code and documentation necessary to understand the main training/evaluation pipeline.

The repository publicly includes:

```text
train_vjepa21_7fields_collision_for_v2_supervised.py
README.md
LICENSE
```

The dataset, processed frames, semantic label files, train/validation/test split files, model checkpoints, and experiment outputs are **not redistributed** in this repository because they are derived from or depend on externally licensed resources.

## Data and Annotation Availability

The raw videos used in this project come from the **Nexar Dashcam Collision Prediction Dataset**, which is governed by the **Nexar Open Data License**. Therefore, the raw dataset and extracted frame files are not publicly redistributed in this repository.

In addition, the semantic labels used in this project were generated and refined for the Nexar video windows. Because these labels are directly tied to Nexar clip identifiers, frame indices, and processed video windows, they are not publicly uploaded as part of this repository.

For academic review or reproducibility verification, the following materials may be shared privately upon reasonable request, subject to dataset license restrictions:

- semantic label schema
- example annotation rows
- preprocessing details
- train/validation/test split format
- command-line configuration used for the main reported run
- evaluation output format
- limited metadata needed to verify the reported experimental pipeline

Requests can be made by contacting the repository owner:minjun9282@khu.ac.kr. The requester may be required to separately obtain the Nexar dataset and V-JEPA 2.1 checkpoint according to their respective licenses.

## Project Summary

Conventional collision warning systems usually output only a binary alert or risk score. This project extends that setup by jointly predicting:

1. **Collision risk**
   - alert
   - no-alert

2. **Seven semantic warning fields**
   - primary hazard type
   - hazard position
   - hazard motion state
   - hazard proximity
   - path relation
   - gap trend
   - lateral origin

The semantic fields are used as structured explanatory outputs. The collision head determines whether a warning should be triggered, while the semantic heads provide the information used to describe the reason for the warning.

## Main Code File

### `train_vjepa21_7fields_collision_for_v2_supervised.py`

This script implements:

- Nexar window-level dataset loading
- train/validation/test split loading
- 11-frame video window construction
- V-JEPA 2.1 official encoder loading
- V-JEPA 2.1 checkpoint loading
- multi-task model definition
- collision prediction head
- seven semantic classification heads
- class-weighted semantic loss
- collision BCE-with-logits loss with positive class weighting
- optional LoRA support
- optional supervised contrastive loss
- validation-based model selection
- early stopping
- evaluation with AP, AUC, mTTA, semantic accuracy, and macro-F1
- test-window prediction dumping

## Semantic Fields

The script uses the following semantic fields:

| Field | Meaning | Classes |
|---|---|---|
| PHT | Primary hazard type | car, truck, bus, motorcyclist, pedestrian, cyclist, roadside_object, none |
| HPOS | Hazard position | ego_lane_front, adjacent_left, adjacent_right, crossing_ahead, roadside, none |
| HMOT | Hazard motion state | stationary, slowing, moving_steady, accelerating, entering_lane, crossing, parked, none |
| HPROX | Hazard proximity | very_close, close, medium, far, none |
| PATH | Path relation | in_path, entering_path, crossing_path, parallel_adjacent, none |
| GAP | Gap trend | closing, stable_gap, none |
| LATORIG | Lateral origin | left, right, none |

## Semantic Label Format

The training script expects one JSON object per line in the semantic label file.

Default path:

```text
output2/labels_for_vjepa_v2_supervised.jsonl
```

Each row should contain at least the following fields:

```json
{
  "category": "crash",
  "clip_name": "example_clip",
  "target_frame_idx": 10,
  "frame_indices": [0, 5, 10],
  "alert": "alert",
  "analysis": {
    "fields": {
      "primary_hazard_type": "car",
      "hazard_position": "ego_lane_front",
      "hazard_motion_state": "stationary",
      "hazard_proximity": "very_close",
      "path_relation": "in_path",
      "gap_trend": "closing",
      "lateral_origin": "none"
    }
  }
}
```

The labels used in the project were produced using representative frames from each window and then manually refined.

## Train/Validation/Test Split Format

The split file is expected at:

```text
data/train_test_clips_for_vjepa2.jsonl
```

Each row should contain:

```json
{
  "split": "train",
  "category": "crash",
  "clip_name": "example_clip"
}
```

The training script uses this file to assign each clip to the train, validation, or test split.

## Frame Directory Format

The extracted frames are expected under:

```text
data/frames/
```

Expected layout:

```text
data/frames/
├── crash/
│   └── <clip_name>/
│       ├── 00000.jpg
│       ├── 00001.jpg
│       └── ...
└── normal/
    └── <clip_name>/
        ├── 00000.jpg
        ├── 00001.jpg
        └── ...
```

## Preprocessing Summary

The main preprocessing procedure used in the project is:

1. Crop crash videos to the final 5 seconds before the collision timestamp.
2. Sample 5-second clips from normal videos using a fixed random seed.
3. Down-sample clips to 10 Hz.
4. Construct 1-second temporal windows with a stride of 0.5 seconds.
5. Use 11 frames per window.
6. Assign each window an alert/no-alert label.
7. Generate and refine seven semantic field labels for each window.

The resulting split used for the main reported experiment contained:

```text
train windows: 8091
validation windows: 2700
test windows: 2691
```

## Expected Data Layout

The code expects the following local paths by default:

```text
output2/labels_for_vjepa_v2_supervised.jsonl
data/train_test_clips_for_vjepa2.jsonl
data/frames/
checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt
external/vjepa2_official/
```

Expected directory structure:

```text
2601-Advanced-Data-Science-Final-Project/
├── README.md
├── LICENSE
├── requirements.txt
├── train_vjepa21_7fields_collision_for_v2_supervised.py
├── output2/
│   └── labels_for_vjepa_v2_supervised.jsonl        # not included
├── data/
│   ├── train_test_clips_for_vjepa2.jsonl           # not included
│   └── frames/                                     # not included
├── checkpoints/
│   └── vjepa2_1/
│       └── vjepa2_1_vitl_dist_vitG_384.pt          # not included
└── external/
    └── vjepa2_official/                            # not included
```

## Dataset Notice

This project uses the **Nexar Dashcam Collision Prediction Dataset**.

The dataset is not included in this repository. Users must obtain the dataset separately from the official source and comply with the **Nexar Open Data License**.

The processed frames, generated window-level annotations, semantic labels, and train/validation/test split files are also not redistributed.

## Model Checkpoint Notice

This project uses **Meta V-JEPA 2.1**.

The V-JEPA 2.1 checkpoint is not included in this repository. Users must obtain the model checkpoint according to Meta's official instructions and license terms.

By default, the script expects the checkpoint at:

```text
checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt
```

## Installation

Create a Python environment:

```bash
conda create -n ads-final python=3.10
conda activate ads-final
```

Install the required packages:

```bash
pip install torch torchvision
pip install numpy pillow scikit-learn pandas
pip install peft
```

The script also depends on the official V-JEPA 2.1 repository being available locally. Clone or place the official repository under:

```text
external/vjepa2_official/
```

The script imports V-JEPA 2.1 modules from that local directory.

## Training

Example command:

```bash
python train_vjepa21_7fields_collision_for_v2_supervised.py \
  --model_name vjepa2_1_vitl_dist_vitG_384 \
  --official_repo_root external/vjepa2_official \
  --labels_path output2/labels_for_vjepa_v2_supervised.jsonl \
  --splits_path data/train_test_clips_for_vjepa2.jsonl \
  --frames_root data/frames \
  --output_dir out_score/vjepa21_7fields_collision \
  --epochs 20 \
  --batch_size 4 \
  --accumulation_steps 16 \
  --lr 2e-5 \
  --weight_decay 1e-4 \
  --dropout 0.1 \
  --window_frames 11 \
  --lambda_semantic 1.0 \
  --lambda_collision 1.0 \
  --model_select_metric semantic_macro_f1 \
  --early_stopping_patience 5 \
  --backbone_ckpt checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt \
  --backbone_ckpt_key ema_encoder \
  --backbone_ckpt_strict 1 \
  --processor_crop_size 384 \
  --device cuda:0
```

## Evaluation Only

After training, the best model is saved as:

```text
<output_dir>/best_model.pt
```

To evaluate an existing checkpoint:

```bash
python train_vjepa21_7fields_collision_for_v2_supervised.py \
  --eval_only_ckpt <output_dir>/best_model.pt \
  --model_name vjepa2_1_vitl_dist_vitG_384 \
  --official_repo_root external/vjepa2_official \
  --labels_path output2/labels_for_vjepa_v2_supervised.jsonl \
  --splits_path data/train_test_clips_for_vjepa2.jsonl \
  --frames_root data/frames \
  --output_dir out_score/eval_only \
  --window_frames 11 \
  --backbone_ckpt checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt \
  --backbone_ckpt_key ema_encoder \
  --device cuda:0
```

## Outputs

The script writes the following files to the output directory:

```text
run_meta.json
train_log.jsonl
best_model.pt
val_metrics.json
test_metrics.json
val_operating.csv
test_operating.csv
test_window_preds_best_ap_semantic.jsonl    # optional
```

The evaluation includes:

- window-level AP
- window-level AUC
- clip-level AP using pre-crash max score
- clip-level AUC using pre-crash max score
- threshold-based operating metrics
- mTTA at the best-recall operating point
- mean semantic field accuracy
- mean semantic field macro-F1
- field-wise accuracy
- field-wise macro-F1

## Main Reported Configuration

The main reported run used:

```text
model_name: vjepa2_1_vitl_dist_vitG_384
epochs: 20
batch_size: 4
accumulation_steps: 16
learning_rate: 2e-5
weight_decay: 1e-4
window_frames: 11
lambda_semantic: 1.0
lambda_collision: 1.0
model_select_metric: semantic_macro_f1
early_stopping_patience: 5
freeze_backbone: 0
processor_crop_size: 384
```

## Main Reported Results

The main report used V-JEPA 2.1 300M with a collision head and seven semantic heads.

```text
AP:   0.9525
AUC:  0.9474
mTTA: 1.865 s
Mean semantic field accuracy: 0.787
```

## Reproducibility Notes

This repository does not include all files required to run the experiment directly after cloning. To reproduce the full experiment, users must separately prepare:

- the Nexar dataset
- extracted 10 Hz frame directories
- semantic annotation JSONL file
- train/validation/test split JSONL file
- V-JEPA 2.1 official repository
- V-JEPA 2.1 pretrained checkpoint

This restriction is due to dataset and model licensing constraints, not because the training or evaluation procedure is intentionally hidden.

## License

The source code in this repository is licensed under the **MIT License**.

This project utilizes **Meta V-JEPA 2.1**, which is restricted to non-commercial research use under the **CC-BY-NC 4.0 license**.

The dataset used is the **Nexar Collision Prediction Dataset**, governed by the **Nexar Open Data License**.

The dataset, processed data, annotations, V-JEPA 2.1 weights, and trained checkpoints are not redistributed in this repository. Users are responsible for obtaining and using each resource according to its respective license.

## Citation

If this repository is useful for your work, please cite the related dataset and model papers:

```bibtex
@inproceedings{moura2025nexar,
  title={Nexar Dashcam Collision Prediction Dataset and Challenge},
  author={Moura, T. and others},
  booktitle={CVPR Workshops},
  year={2025}
}

@article{murlabadia2026vjepa21,
  title={V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning},
  author={Mur-Labadia, L. and others},
  journal={arXiv preprint arXiv:2603.14482},
  year={2026}
}

@article{assran2025vjepa2,
  title={V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, M. and others},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}
```
