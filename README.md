# Transfer Learning and Ensemble Learning for PET-MPI Polar Map Classification

Code for transfer learning and ensemble learning experiments on myocardial ischemia classification from PET-MPI (Positron Emission Tomography - Myocardial Perfusion Imaging) polar maps. The methods evaluate multiple pre-trained CNN architectures and combine them using ensemble strategies for improved classification performance.

**Note:** The medical imaging data used in this study cannot be shared due to patient privacy regulations. However, the code can be adapted to any binary image classification task with the appropriate data format (see Data Setup below).

## What's Included

- `TL_fixedvalidation.py` - Transfer learning with fixed train/validation split
- `TL_crossvalidation.py` - Transfer learning with 5-fold cross-validation  
- `EL_ensemble.ipynb` - Ensemble analysis combining multiple models
- `run_TL_fixedvalidation.sh` - Script to run validation split experiments
- `run_TL_crossvalidation.sh` - Script to run cross-validation experiments

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and TensorFlow 2.8+.

## Data Setup

Your data should be organized as:

```
data/
├── training/
│   ├── *.jpg                    # Training images
│   └── ica_lables.txt           # Labels (one per line: 0 or 1)
└── test/
    ├── *.jpg                    # Test images
    └── ica_lables.txt           # Labels (one per line: 0 or 1)
```

See `data/README.md` for detailed format requirements.

## Usage

### Run Transfer Learning

Edit the bash scripts to select which models to train, then:

```bash
# Fixed validation split (2:1 train/val)
bash run_TL_fixedvalidation.sh

# Cross-validation (5-fold)
bash run_TL_crossvalidation.sh
```

### Run Ensemble Analysis

After training models, open the notebook:

```bash
jupyter notebook EL_ensemble.ipynb
```

## Available Models

VGG16, VGG19, ResNet50, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, DenseNet169, DenseNet201, Xception, EfficientNet (B0-B7, V2B0-V2B2), MobileNetV2, NASNet

## Results

Results are saved to:
- `results/models/` - Trained model weights
- `reports/metrics.xlsx` - Performance metrics
- `log/` - Training logs

## License

MIT License - see LICENSE file.

## Author

**Seyed M. Hosseini**  
Turku PET Centre, University of Turku  
smhoss@utu.fi
or smrh.1372@gmail.com

For questions, open an issue on GitHub or contact via email.
