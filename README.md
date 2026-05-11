# WeedsData — Drone-based Weed Detection (YOLOv8)

Short guide and reproduction instructions for the Weeds dataset and training experiments.

## Quick Links
- **Dataset export:** [README.roboflow.txt](README.roboflow.txt#L1)
- **Training script:** [yolo_training.py](yolo_training.py#L1)
- **Data config:** [data.yaml](data.yaml#L1)
- **Experiment runs:** [experimentResults](experimentResults)
- **Algorithm comparison:** [algorithmResults/algorithm_comparison.json](algorithmResults/algorithm_comparison.json)

## Description
This repository contains a YOLOv8-based dataset and training code to detect weeds from drone imagery. The dataset (exported from Roboflow) contains 576 images annotated in YOLO format. The goal is to train and evaluate object detection models for automated herbicide spraying.

## Requirements
- Python 3.8+
- A GPU with CUDA (recommended) for training

Install required Python packages (recommended inside a virtual environment):

```bash
python -m venv .venv
.
# Windows PowerShell
.
# Activate the venv (PowerShell)
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (./.venv/Scripts/Activate.ps1)

pip install --upgrade pip
pip install ultralytics
# Install PyTorch according to your CUDA version (see https://pytorch.org)
```

## Quick Start — Training
1. Confirm `data.yaml` is set up (paths to `train/`, `valid/`, `names` file). See [data.yaml](data.yaml#L1).
2. Edit `yolo_training.py` if you want different model or training args.
3. Run training:

```bash
# Using the script in this repo
python yolo_training.py

# Or with the ultralytics CLI (example)
yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=50 imgsz=640 batch=16
```

The script `yolo_training.py` uses the `ultralytics` package and by default writes outputs to the folder specified in the script (`experimentResults` in the repository). See [yolo_training.py](yolo_training.py#L1).

## Inference / Prediction
Load a trained model and run predictions on a folder of images:

```python
from ultralytics import YOLO

# load trained weights (example path)
model = YOLO('experimentResults/weights/best.pt')
results = model.predict(source='test/images', imgsz=640, save=True)
```

Outputs and visualizations will be saved by the `ultralytics` library into the run directory.

## Experiments & Results
- Experiment runs are stored under [experimentResults](experimentResults) and other experiment folders (e.g., `experimentResults2`, `experimentResults3`). Each run may include an `args.yaml` and `weights/` subfolder.
- Summary comparisons and evaluation outputs are available at [algorithmResults/algorithm_comparison.json](algorithmResults/algorithm_comparison.json) and under [algorithmResults](algorithmResults).

### Best Model: YOLOv8s
```
Training Epochs: 10
mAP50:     0.751
mAP50-95:  0.529
Precision: 0.743
Recall:    0.706
FPS:       28 (real-time)
```

### Per-Class Performance
- **Broadleaf Weeds:** 0.78 mAP50 (Best)
- **Invasive Vines:** 0.74 mAP50
- **Sedge Species:** 0.75 mAP50
- **Grassy Weeds:** 0.71 mAP50 (Most challenging)
- **Other Vegetation:** 0.68 mAP50


## Repository Structure (key files)
- `data.yaml` — dataset paths and class names ([data.yaml](data.yaml#L1))
- `yolo_training.py` — sample training script ([yolo_training.py](yolo_training.py#L1))
- `README.roboflow.txt` — Roboflow export details ([README.roboflow.txt](README.roboflow.txt#L1))
- `train/`, `valid/`, `test/` — image and label splits
- `experimentResults*/` — training run outputs and weights
- `algorithmResults/` — aggregated results and comparisons

Important Note: added Faster R-CNN in train_multiple_algorithms.py but could not compare performance as I had Limited GPU resources.
## Reproducibility notes
- Many training experiments include an `args.yaml` file in their run folder. Use those values to reproduce a specific run.
- If you move or rename directories, ensure `data.yaml` paths are updated accordingly.

## Contributing & Next Steps
- If you want to improve training or experiments: add a new script under `scripts/` (create if needed) and document the commands here.
- Suggested next steps: hyperparameter sweep, data augmentation experiments, cross-validation, or model ensemble.

## License
This repository does not include a license file. Add a `LICENSE` if you intend to publish under a specific license (e.g., MIT).

## Contact
For questions about the dataset or experiments, open an issue in this repository.
