# BdSL-SPOTER: A Transformer-Based Framework for Bengali Sign Language Recognition

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Conference-blue)](link-to-paper)
[![Dataset](https://img.shields.io/badge/Dataset-BdSLW60-green)](link-to-dataset)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-orange)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red)](https://pytorch.org)

## 🎯 Overview

**BdSL-SPOTER** is a breakthrough transformer-based framework for Bengali Sign Language (BdSL) recognition that achieves **97.92% Top-1 accuracy** on the BdSLW60 dataset—a remarkable **22.82 percentage point improvement** over previous baselines. Our culturally-adapted approach addresses the communication needs of **13.7 million hearing-impaired individuals** in Bangladesh.

### Key Achievements
- 🏆 **97.92% Top-1 accuracy** on BdSLW60 dataset
- ⚡ **4.8 minutes** training time (vs. 45 minutes for baseline)
- 🚀 **127 FPS** inference speed
- 💾 **60% parameter reduction** compared to existing methods
- 🎯 **Perfect classification** on 52 out of 60 sign classes

## 🚀 Features

- **Cultural Adaptation**: BdSL-specific pose normalization and signing space characteristics
- **Efficient Architecture**: Optimized 4-layer transformer encoder with 9 attention heads
- **Advanced Training**: Curriculum learning with targeted augmentations and label smoothing
- **Real-time Performance**: 127 FPS inference suitable for mobile deployment
- **Comprehensive Evaluation**: Extensive ablation studies and cross-validation results

## 📋 Requirements

```bash
Python >= 3.8
PyTorch >= 1.12
CUDA >= 11.6
MediaPipe >= 0.8.10
NumPy >= 1.21.0
OpenCV >= 4.5.0
Matplotlib >= 3.5.0
Scikit-learn >= 1.0.0
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/bdsl-spoter.git
cd bdsl-spoter
```

2. **Create conda environment**
```bash
conda create -n bdsl-spoter python=3.8
conda activate bdsl-spoter
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install MediaPipe**
```bash
pip install mediapipe==0.8.10
```

## 📊 Dataset

### BdSLW60 Dataset
- **9,307 videos** across **60 BdSL word classes**
- **18 native signers** with diverse demographics
- **Split**: 70% training, 15% validation, 15% testing
- **Preprocessing**: MediaPipe Holistic pose extraction

### Data Structure
```
data/
├── BdSLW60/
│   ├── train/
│   │   ├── class_01/
│   │   ├── class_02/
│   │   └── ...
│   ├── val/
│   └── test/
├── annotations/
│   ├── train_labels.json
│   ├── val_labels.json
│   └── test_labels.json
└── pose_features/
    ├── train_poses.pkl
    ├── val_poses.pkl
    └── test_poses.pkl
```

## 🏃‍♂️ Quick Start

### 1. Pose Extraction
```bash
python scripts/extract_poses.py \
    --input_dir data/BdSLW60/train \
    --output_dir data/pose_features \
    --split train
```

### 2. Training
```bash
python train.py \
    --config configs/bdsl_spoter.yaml \
    --data_dir data/pose_features \
    --output_dir experiments/bdsl_spoter \
    --gpu 0
```

### 3. Evaluation
```bash
python evaluate.py \
    --model_path experiments/bdsl_spoter/best_model.pth \
    --test_data data/pose_features/test_poses.pkl \
    --config configs/bdsl_spoter.yaml
```

### 4. Inference
```bash
python inference.py \
    --model_path experiments/bdsl_spoter/best_model.pth \
    --video_path sample_videos/sign_example.mp4 \
    --config configs/bdsl_spoter.yaml
```

## 🏗️ Architecture

### Core Components

1. **Cultural Pose Preprocessing**
   - MediaPipe Holistic extraction (108-dimensional features)
   - BdSL-specific signing space normalization (α = 0.85)
   - Confidence-aware frame filtering
   - Temporal smoothing

2. **Transformer Encoder**
   - 4-layer transformer encoder
   - 9 multi-head attention heads
   - Learnable positional encodings
   - Model dimension: 108, FFN dimension: 512

3. **Classification Head**
   - Global average pooling
   - LayerNorm → Linear(108→54) → GELU → Dropout → Linear(54→60)

### Training Strategy
- **Curriculum Learning**: Two-stage approach (short → full sequences)
- **Data Augmentation**: Temporal stretch, spatial jitter, random rotation
- **Optimization**: AdamW with OneCycleLR scheduler
- **Regularization**: Label smoothing (0.1), dropout (0.15)

## 📈 Results

### Performance Comparison

| Method | Top-1 Acc (%) | Top-5 Acc (%) | Macro F1 | Training Time (min) |
|--------|---------------|---------------|----------|-------------------|
| Bi-LSTM | 75.10 | 89.20 | 0.742 | 45 |
| Standard SPOTER | 82.40 | 94.10 | 0.801 | 13 |
| CNN-LSTM Hybrid | 79.80 | 91.50 | 0.785 | 39 |
| **BdSL-SPOTER (Ours)** | **97.92** | **99.80** | **0.979** | **4.8** |
| **Improvement** | **+22.82** | **+10.60** | **+0.237** | **-89.3%** |

### Ablation Studies

| Component | Top-1 Acc (%) | Δ Acc (pp) |
|-----------|---------------|------------|
| 2 layers | 89.20 | -- |
| **4 layers (ours)** | **97.92** | **+8.72** |
| 6 layers | 96.80 | +7.60 |
| BdSL-specific normalization | **97.92** | **+4.30** |
| Curriculum learning | 97.92 | +3.62 |
| Learnable encoding | **97.92** | **+2.32** |

## 🔧 Configuration

### Model Configuration (`configs/bdsl_spoter.yaml`)
```yaml
model:
  name: "BdSL_SPOTER"
  num_classes: 60
  pose_dim: 108
  max_seq_length: 150
  
  encoder:
    num_layers: 4
    num_heads: 9
    hidden_dim: 108
    ffn_dim: 512
    dropout: 0.15
    
training:
  batch_size: 32
  epochs: 20
  learning_rate: 3e-4
  weight_decay: 1e-4
  label_smoothing: 0.1
  
  curriculum:
    stage1_max_frames: 50
    stage1_epochs: 10
    
  augmentation:
    temporal_stretch: 0.1
    spatial_noise_std: 0.02
    rotation_range: 5
```

## 📊 Monitoring Training

```bash
# View training progress
tensorboard --logdir experiments/bdsl_spoter/logs

# Monitor system resources
python scripts/monitor_training.py --exp_dir experiments/bdsl_spoter
```

## 🚀 Deployment

### Real-time Inference
```python
from bdsl_spoter import BdSLSPOTER
import cv2

# Load model
model = BdSLSPOTER.load_pretrained('experiments/bdsl_spoter/best_model.pth')

# Real-time inference
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        prediction = model.predict_frame(frame)
        print(f"Predicted sign: {prediction['class']}, Confidence: {prediction['confidence']:.2f}")
```

### Mobile Deployment
```bash
# Convert to TorchScript for mobile
python scripts/export_mobile.py \
    --model_path experiments/bdsl_spoter/best_model.pth \
    --output_path models/bdsl_spoter_mobile.pt
```

## 🧪 Experiments

### Run Full Experimental Suite
```bash
# Complete ablation studies
bash scripts/run_ablations.sh

# Cross-validation experiments
python experiments/cross_validation.py --k_folds 5

# Attention visualization
python experiments/visualize_attention.py \
    --model_path experiments/bdsl_spoter/best_model.pth \
    --sample_video data/sample_signs/hello.mp4
```

## 📊 Evaluation Metrics

- **Top-1/Top-5 Accuracy**: Classification accuracy
- **Macro F1-Score**: Balanced performance across classes
- **Per-class Analysis**: Individual class performance
- **Confusion Matrix**: Detailed error analysis
- **Statistical Significance**: Paired t-tests with 95% confidence intervals

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution
- [ ] Additional BdSL datasets
- [ ] Continuous sign language recognition
- [ ] Mobile optimization
- [ ] Multi-modal fusion (RGB + pose)
- [ ] Real-world deployment studies

## 📚 Citation

If you use BdSL-SPOTER in your research, please cite:

```bibtex
@article{bdsl_spoter2024,
  title={BdSL-SPOTER: A Transformer-Based Framework for Bengali Sign Language Recognition with Cultural Adaptation},
  author={Sayad Ibna Azad, Md Atiqur Rahman},
  journal={Accepted in ISVC},
  year={2025},
  note={Accepted}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Contributors to the BdSLW60 dataset
- Bangladesh's deaf community members who participated in data collection
- MediaPipe team for pose estimation tools
- PyTorch community for deep learning framework

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: [Sayad Ibna Azad](mailto:sayadkhan0555@gmail.com)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-repo/bdsl-spoter&type=Date)](https://star-history.com/#your-repo/bdsl-spoter&Date)

---

**Making sign language recognition accessible for Bangladesh's 13.7 million hearing-impaired citizens** 🇧🇩
