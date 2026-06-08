# SHPNet: Shape-Aware Contrastive Learning with Hierarchical Semantic Prediction for Fine-Grained SAR Aircraft Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

Official PyTorch implementation of **SHPNet** for fine-grained aircraft detection in SAR imagery.

> **SHPNet: Shape-Aware Contrastive Learning and Hierarchical Semantic Prediction for Fine-Grained SAR Aircraft Detection**  
> *Ru Luo, Lingjun Zhao, Qishan He, Siqian Zhang, Kefeng Ji*  


## 📖 Overview

**SHPNet** addresses two core bottlenecks in fine-grained SAR aircraft detection:

| Bottleneck | Our Solution |
|:---|:---|
| **Azimuth-sensitive scattering** → unstable feature representations | **ShapeCL**: geometric structural consistency with contrastive learning to enhance intra-class consistency feature learning against azimuth-sensitive scattering signature |
| **Flat classification** → semantic confusion among similar subcategories | **HPH**: progressive coarse-to-fine hierarchical prediction |

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

| Dataset | Overall mAP | GFLOPs | Params (M) |
|:---|:---:|:---:|:---:|
| SAR-RADD | **80.5%** | 31.4 | 13.6 |
| FAIR-CSAR | **54.3%** | 31.4 | 13.6 |

<details>
<summary>📈 Per-category results on SAR-RADD (click to expand)</summary>

| Category | A | B | C | D | E | F | G | H | I | J | K | L | M |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AP (%) | 92.3 | 92.8 | 91.0 | 78.4 | 84.6 | 79.1 | 78.2 | 71.7 | 90.1 | 76.1 | 43.1 | 79.2 | 90.5 |

</details>

<details>
<summary>📈 Per-category results on FAIR-CSAR (click to expand)</summary>

| Category | A220 | A320 | A330 | Airfree. | B737 | B747 | B767 | B777 | Fokker | Gulf | Heli | Other |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AP (%) | 61.6 | 37.7 | 69.7 | 48.0 | 70.7 | 70.1 | 56.1 | 22.3 | 76.2 | 37.3 | 77.7 | 24.6 |

</details>

<details>
<summary>🏆 Comparison with state-of-the-art (click to expand)</summary>

| Method | SAR-RADD | FAIR-CSAR | GFLOPs | Params |
|:---|:---:|:---:|:---:|:---:|
| Faster R-CNN | 64.0 | 38.0 | - | - |
| RetinaNet | 63.9 | 38.1 | - | - |
| YOLOv11s | 71.4 | 45.4 | 22.3 | 9.7 |
| SAR-SFNet | 79.3 | 50.6 | - | - |
| **SHPNet** | **80.5** | **54.3** | 31.4 | 13.6 |

</details>
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


