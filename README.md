# 3D Kidney Cancer Aggressiveness Classification

This project provides a complete pipeline for training and evaluating 3D deep learning models to classify the aggressiveness of kidney cancer from multi-sequence CT scans. It uses the MONAI framework and supports various fusion strategies for combining information from different CT phases.

## Features

- **Multi-Sequence Fusion**: Supports combining multiple CT sequences (e.g., Arterial, Venous, etc.) using:
  - **Concatenation**: A simple and robust feature fusion method.
  - **Cross-Attention**: A more advanced method to weigh the importance of each sequence.
- **Pre-trained Models**: Easily leverages powerful 3D models (ResNet) pre-trained on large-scale medical datasets (MedicalNet) to boost performance.
- **Efficient Data Loading**: Uses SmartCacheDataset to accelerate training by caching pre-processed data, solving common I/O bottlenecks.
- **Modular Codebase**: Clean separation of concerns with dedicated files for the dataset (dataset.py), model (model.py), and training loop (train.py).
- **Comprehensive Evaluation**: Calculates and logs key metrics including AUC, accuracy, F1-score, precision, and recall using TensorBoard.

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a Conda Environment

It is highly recommended to use a Conda environment to manage dependencies.

```bash
conda create -n kidney_cancer python=3.10 -y
conda activate kidney_cancer
```

### 3. Install Dependencies

This project requires PyTorch with CUDA support, MONAI, and several other packages.

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MONAI and other required packages
pip install monai scikit-learn tqdm tensorboard x-transformers
```

## Data Preparation

The model expects the data to be organized in a specific structure and referenced by JSON manifest files.

### 1. Directory Structure

Organize your data files as follows:

```
<project_root>/
├── Data/
│   ├── data_nifty/
│   │   └── 1.Training_DICOM_603/
│   │       └── PATIENT_ID/
│   │           └── CT/
│   │               ├── A_8/
│   │               │   └── ...nii.gz
│   │               └── V_8/
│   │                   └── ...nii.gz
│   ├── VOI_nifty/
│   │   └── 1.Training_VOI_603/
│   │       └── PATIENT_ID/
│   │           └── VOI/
│   │               └── A.nii.gz
│   ├── training_manifest.json
│   └── internal_test_manifest.json
├── pre_trained/
│   └── tencent_pretrain/
│       └── resnet_18_23dataset.pth
├── dataset.py
├── model.py
└── train.py
```

### 2. JSON Manifest Files

Create `training_manifest.json` and `internal_test_manifest.json` inside the `Data/` directory. Each file should be a list of JSON objects, where each object represents one patient and contains file paths to their data.

**Example entry in training_manifest.json:**

```json
[
  {
    "label": 0,
    "patient_id": "P2121787",
    "A": "Data/data_nifty/1.Training_DICOM_603/AO_XIAO_MAO_P2121787/CT/A_8/A_8_01_ThoraxRoutine_20191010092049_10.nii.gz",
    "V": "Data/data_nifty/1.Training_DICOM_603/AO_XIAO_MAO_P2121787/CT/V_8/V_8_01_ThoraxRoutine_20191010092049_12.nii.gz",
    "mask": "Data/VOI_nifty/1.Training_VOI_603/AO_XIAO_MAO_P2121787/VOI/A.nii.gz"
  }
]
```

## How to Run

### Training

The `train.py` script is the main entry point for training the model. You can configure experiments using command-line arguments.

**Example 1: Train a pre-trained ResNet-18 with concat fusion**

This command uses the Arterial ('A') and Venous ('V') phases, applies the VOI mask, and saves the output to a new experiment directory.

```bash
python train.py \
    --ct_types A V \
    --model_depth 18 \
    --fusion_method concat \
    --apply_voi_mask \
    --batch_size 4 \
    --lr 1e-4 \
    --epochs 100 \
    --pretrained \
    --local_pretrained_path "pre_trained/tencent_pretrain/resnet_18_23dataset.pth"
```

**Example 2: Train a ResNet-34 with attention fusion**

This command trains a deeper model using the more complex attention-based fusion.

```bash
python train.py \
    --ct_types A D N V \
    --model_depth 34 \
    --fusion_method attention \
    --apply_voi_mask \
    --batch_size 2 \
    --lr 1e-5 \
    --epochs 150 \
    --pretrained \
    --local_pretrained_path "pre_trained/tencent_pretrain/resnet_34_23dataset.pth"
```

### Monitoring

You can monitor the training progress, including loss and metrics curves, using TensorBoard:

```bash
tensorboard --logdir=outputs
```

## File Structure

- **train.py**: The main script for running the training and validation loops. It handles argument parsing, model initialization, and experiment management.
- **dataset.py**: Contains the `MONAITumorDataset` and `MONAITumorDataLoader` classes, which manage data loading, transformations, and caching.
- **model.py**: Defines the `MultiSequenceResNet` architecture, including the fusion logic for concat and attention.
- **test_model.py / test_dataset.py**: Utility scripts for verifying that the model and data loaders are working correctly.