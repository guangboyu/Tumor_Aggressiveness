# Tumor Aggressiveness Classification

A deep learning project for classifying tumor aggressiveness using multi-sequence CT scans and VOI (Volume of Interest) segmentation masks.

## Project Overview

This project implements various fusion strategies for multi-sequence CT data (A, D, N, V sequences) to classify tumor aggressiveness. The system supports early fusion, intermediate fusion, and ensemble approaches using 3D ResNet architectures.

## Features

- **Multiple Fusion Strategies**: Early fusion, intermediate fusion, single sequence, and ensemble methods
- **Flexible Data Loading**: Supports NIfTI and NRRD formats with automatic orientation handling
- **3D ResNet Models**: Based on MONAI's 3D ResNet implementations
- **Comprehensive Training Pipeline**: Includes validation, testing, and model checkpointing
- **Medical Image Preprocessing**: VOI masking, normalization, and resampling

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd tumor-aggressiveness-classification
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Data Structure

The project expects the following data structure:
```
Data/
├── ccRCC_Survival_Analysis_Dataset_english/
│   ├── training_set_603_cases.csv
│   ├── internal_test_set_259_cases.csv
│   └── external_verification_set_308_cases.csv
├── data_nifty/
│   ├── 1.Training_DICOM_603/
│   ├── 2.Internal Test_DICOM_259/
│   └── 3.External Test_DICOM_308/
└── ROI/
    ├── 1.Training_ROI_603/
    ├── 2.Internal Test_ROI_259/
    └── 3.External Test_ROI_308/
```

## Usage

### Training

Train a model with intermediate fusion and attention mechanism:
```bash
python train.py --fusion_strategy intermediate --fusion_method attention --batch_size 8 --epochs 50 --lr 1e-4
```

### Available Options

- `--fusion_strategy`: `early`, `intermediate`, `single`, `ensemble`
- `--fusion_method`: `concat`, `attention`, `weighted_sum` (for intermediate fusion)
- `--ct_types`: List of CT sequences to use (default: `A D N V`)
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--target_size`: Target volume size (default: 128 128 128)

### Testing VOI Alignment

Check if CT and VOI files are properly aligned:
```bash
python z_test.py
```

## Model Architectures

### Early Fusion
- Concatenates all CT sequences as channels
- Input: `(B, 4, D, H, W)` where 4 = number of CT sequences
- Single 3D ResNet processes the concatenated input

### Intermediate Fusion
- Processes each CT sequence separately with individual ResNet backbones
- Fuses features using attention, concatenation, or weighted sum
- Supports different fusion methods for optimal performance

### Ensemble
- Trains separate models for each CT sequence
- Combines predictions using voting, weighted voting, or stacking

## Project Structure

```
├── dataset.py          # Dataset and DataLoader classes
├── model.py            # Model architectures and fusion strategies
├── train.py            # Training script
├── z_test.py           # VOI alignment testing
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
└── README.md          # This file
```

## Performance Metrics

The training pipeline tracks:
- Accuracy
- Precision, Recall, F1-score
- AUC-ROC
- Loss curves

Results are saved in the `outputs/` directory with TensorBoard logs for visualization.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tumor_aggressiveness_2024,
  title={Tumor Aggressiveness Classification using Multi-Sequence CT Scans},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Acknowledgments

- MONAI for medical imaging deep learning tools
- 3D Slicer for medical image visualization
- PyTorch for deep learning framework
