# SHPNet: Shape-Aware Contrastive Learning with Hierarchical Semantic Prediction for Fine-Grained SAR Aircraft Detection

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

Official PyTorch implementation of **SHPNet** for fine-grained aircraft detection in SAR imagery.

> **SHPNet: Shape-Aware Contrastive Learning and Hierarchical Semantic Prediction for Fine-Grained SAR Aircraft Detection**  
> *Ru Luo, Lingjun Zhao, Qishan He, Siqian Zhang, Kefeng Ji*  


## 🚀 News

- **[2026-06]** 🔥 Code and pretrained models released!
- **[2026-06]** 🎉 Paper accepted by ISPRS Journal of Photogrammetry and Remote Sensing.

---

## 📖 Overview

**SHPNet** addresses two core bottlenecks in fine-grained SAR aircraft detection:

| Bottleneck | Our Solution |
|:---|:---|
| **Azimuth-sensitive scattering** → unstable feature representations | **ShapeCL**: enforces geometric structural consistency across azimuths |
| **Flat classification** → semantic confusion among similar subcategories | **HPH**: progressive coarse-to-fine hierarchical prediction |

**Key results:**
- SAR-RADD: **80.5% mAP**
- FAIR-CSAR: **54.3% mAP**
- Model: **31.4 GFLOPs**, **13.6M params**

<p align="center">
  <img src="figures/framework.png" width="80%">
  <br>
  <em>Figure 1: Overall framework of SHPNet.</em>
</p>

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.1+ (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/SHPNet.git
cd SHPNet

# Create conda environment (optional)
conda create -n shpnet python=3.8 -y
conda activate shpnet

# Install dependencies
pip install -r requirements.txt
