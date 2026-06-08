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

<h2>📊 Experimental Results</h2>

<h3>Performance on SAR-RADD Dataset</h3>
<table>
  <tr><th>Category</th><th>AP (%)</th><th>Category</th><th>AP (%)</th></tr>
  <tr><td>A</td><td>92.3</td><td>H</td><td>71.7</td></tr>
  <tr><td>B</td><td>92.8</td><td>I</td><td>90.1</td></tr>
  <tr><td>C</td><td>91.0</td><td>J</td><td>76.1</td></tr>
  <tr><td>D</td><td>78.4</td><td>K</td><td>43.1</td></tr>
  <tr><td>E</td><td>84.6</td><td>L</td><td>79.2</td></tr>
  <tr><td>F</td><td>79.1</td><td>M</td><td>90.5</td></tr>
  <tr><td>G</td><td>78.2</td><td>-</td><td>-</td></tr>
</table>
<p><strong>Overall: mAP = 80.5%</strong></p>

<h3>Performance on FAIR-CSAR Dataset</h3>
<table>
  <tr><th>Category</th><th>AP (%)</th><th>Category</th><th>AP (%)</th></tr>
  <tr><td>A220</td><td>61.6</td><td>B767</td><td>56.1</td></tr>
  <tr><td>A320</td><td>37.7</td><td>B777</td><td>22.3</td></tr>
  <tr><td>A330</td><td>69.7</td><td>Fokker-50</td><td>76.2</td></tr>
  <tr><td>Airfreighter</td><td>48.0</td><td>Gulfstream</td><td>37.3</td></tr>
  <tr><td>B737</td><td>70.7</td><td>Helicopter</td><td>77.7</td></tr>
  <tr><td>B747</td><td>70.1</td><td>Other</td><td>24.6</td></tr>
</table>
<p><strong>Overall: mAP = 54.3%</strong></p>

Comparison with State-of-the-Art Methods
Method	SAR-RADD mAP (%)	FAIR-CSAR mAP (%)
Faster R-CNN	64.0	38.0
RetinaNet	63.9	38.1
RepPoints	70.9	41.6
RoI Transformer	67.0	40.3
S²ANet	69.6	41.6
YOLOv11s	71.4	45.4
DenoDet	72.3	42.0
RTMDet-RSIM	75.6	43.3
SAR-SFNet	79.3	50.6
SHPNet (Ours)	80.5	54.3



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


