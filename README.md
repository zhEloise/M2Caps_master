# M2Caps: Learning Multi-Modal Capsules of Optical and SAR Images for Land Cover Classification

Official PyTorch implementation of **M2Caps** published in *International Journal of Digital Earth*.

[![Paper](https://img.shields.io/badge/Paper-IJDE-blue)](https://doi.org/10.1080/17538947.2024.2447347)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Introduction

M2Caps is a novel deep learning framework for multi-modal land cover classification that integrates optical and SAR (Synthetic Aperture Radar) images. The network leverages two key innovations:

- **MMC (Multi-Modal Capsule)**: A capsule-based routing mechanism that learns discriminative multi-modal representations through dynamic routing between optical and SAR features.
- **CFF (Cascaded Feature Fusion)**: A progressive feature fusion module that refines multi-scale features through cascaded channel embedding.

## 🌟 Key Features

- Multi-modal fusion of optical and SAR imagery
- Capsule network-based feature learning
- Cascaded feature fusion for progressive refinement
- Support for multiple benchmark datasets (ISPRS Vaihingen, ISPRS Potsdam, WHU-OPT-SAR)
- PyTorch Lightning training framework

## 🏗️ Network Architecture

The M2Caps network consists of:

1. **Dual-stream Backbone**: ResNeXt-50 encoders for optical and SAR modalities
2. **MMC Module**: Multi-modal capsule routing for discriminative feature learning
3. **CFF Modules**: Cascaded feature fusion at multiple scales
4. **Segmentation Head**: Final prediction layer for land cover classification

## 🔧 Requirements

- Python 3.7+
- PyTorch 1.10+
- PyTorch Lightning
- torchvision
- einops
- timm
- numpy
- opencv-python

## 📦 Installation

```bash
git clone https://github.com/yourusername/M2Caps.git
cd M2Caps
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### WHU-OPT-SAR

1. Download the dataset from [WHU website](http://gpcv.whu.edu.cn/data/)
2. Organize the data as follows:
```
data/whu-opt-sar/
├── train/
│   ├── optical/
│   ├── sar/
│   └── labels/
└── val/
    ├── optical/
    ├── sar/
    └── labels/
```

## 🚀 Training
```

### Train on WHU-OPT-SAR

```bash
python train_supervision.py -c config/whu-opt-sar/config.py
```

### Training Tips

- Adjust batch size in config files based on your GPU memory
- Training logs are saved in `lightning_logs/` directory
- Model checkpoints are saved in `model_weights/[dataset]/` directory
- Monitor training with: `tensorboard --logdir lightning_logs`

## 🧪 Testing

### WHU-OPT-SAR

```bash
python whu_test.py -c config/whu-opt-sar/config.py -o fig_results/whu-opt-sar/out
```

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{zhang2025m2caps,
  title={M2Caps: learning multi-modal capsules of optical and SAR images for land cover classification},
  author={Zhang, H. and Yu, A. and Gao, K. and Lu, X. and Cao, X. and Guo, W. and Lian, W.},
  journal={International Journal of Digital Earth},
  volume={18},
  number={1},
  year={2025},
  publisher={Taylor \& Francis},
  doi={10.1080/17538947.2024.2447347}
}
```

## 🙏 Acknowledgements

This work is based on PyTorch and PyTorch Lightning. We thank the authors of the ISPRS and WHU datasets for making their data publicly available.

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.


## ⭐ Star History

If you find this project helpful, please consider giving it a star ⭐!
