# Physical Prior-Guided Contrastive Learning and Hierarchical Semantic Refinement for Fine-Grained Aircraft Detection in SAR Imagery

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

Official PyTorch implementation Physical Prior-Guided Contrastive Learning and Hierarchical Semantic Refinement Fine-Grained Aircraft Detection in SAR Imagery.

> **Physical Prior-Guided Contrastive Learning and Hierarchical Semantic Refinement for Fine-Grained Aircraft Detection in SAR Imagery**  
> *Ru Luo, Lingjun Zhao, Qishan He, Siqian Zhang, Kefeng Ji*

---

## 📖 Overview

**The proposed method** addresses two core bottlenecks in fine-grained SAR aircraft detection:

| Bottleneck | Our Solution |
|:---|:---|
| **Azimuth-sensitive scattering signature** → unstable feature representations | **Physical Prior-Guided Contrastive Learning (PPGCL)**: geometric structural consistency with contrastive learning to enhance intra-class feature learning against azimuth-sensitive scattering |
| **Flat classification** → semantic confusion among similar subcategories | **HPR**: progressive coarse-to-fine hierarchical semantic refinement |

**Key results:**
- SAR-RADD: **80.5% mAP**
- FAIR-CSAR: **54.3% mAP**
- Model: **31.4 GFLOPs**, **13.6M params**

<p align="center">
  <img src="figures/Overall framework.png" width="80%">
  <br>
  <em>Figure 1: Overall framework of SHPNet.</em>
</p>

---

## 📊 Experimental Results

### Performance on SAR-RADD Dataset

| Category | AP (%) | Category | AP (%) |
|:---|:---:|:---|:---:|
| A | 92.3 | H | 71.7 |
| B | 92.8 | I | 90.1 |
| C | 91.0 | J | 76.1 |
| D | 78.4 | K | 43.1 |
| E | 84.6 | L | 79.2 |
| F | 79.1 | M | 90.5 |
| G | 78.2 | - | - |

**Overall: mAP = 80.5%**

### Performance on FAIR-CSAR Dataset

| Category | AP (%) | Category | AP (%) |
|:---|:---:|:---|:---:|
| A220 | 61.6 | B767 | 56.1 |
| A320 | 37.7 | B777 | 22.3 |
| A330 | 69.7 | Fokker-50 | 76.2 |
| Airfreighter | 48.0 | Gulfstream | 37.3 |
| B737 | 70.7 | Helicopter | 77.7 |
| B747 | 70.1 | Other | 24.6 |

**Overall: mAP = 54.3%**

### Comparison with State-of-the-Art Methods

| Method | SAR-RADD mAP (%) | FAIR-CSAR mAP (%) |
|:---|:---:|:---:|
| Faster R-CNN | 64.0 | 38.0 |
| RetinaNet | 63.9 | 38.1 |
| RepPoints | 70.9 | 41.6 |
| RoI Transformer | 67.0 | 40.3 |
| S²ANet | 69.6 | 41.6 |
| YOLOv11s | 71.4 | 45.4 |
| DenoDet | 72.3 | 42.0 |
| RTMDet-RSIM | 75.6 | 43.3 |
| SAR-SFNet | 79.3 | 50.6 |
| **Ours** | **80.5** | **54.3** |

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.1+ (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/GeoSARprocess/SHPNet.git
cd SHPNet

# Create conda environment (optional)
conda create -n shpnet python=3.8 -y
conda activate shpnet

# Install dependencies
pip install -r requirements.txt
