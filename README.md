<div align="center">

# 🚆 RailDet-YOLO

### Lightweight Railway Wheel Defect Detection with Direction-Aware and Representation Learning

<p align="center">
  <img src="assets/architecture.png" alt="RailDet-YOLO Architecture" width="85%"/>
</p>

---

[![Paper](https://img.shields.io/badge/Paper-Scientific%20Reports-blue?style=flat-square&logo=read-the-docs)](https://www.nature.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![YOLOv10](https://img.shields.io/badge/Base-YOLOv10n-purple?style=flat-square)](https://github.com/THU-MIG/yolov10)

> **📢 Note:** This repository will be made fully public following the official publication of the paper. Code, model weights, and training scripts will be released at that time.

</div>

---

## 📄 Abstract

Railway wheels are safety-critical components, and undetected surface defects including spalling, cracks, scratches, and shelling can precipitate derailment. Periodic manual or ultrasonic inspection is labour-intensive and often misses early micro-defects, while generic object detectors are not designed for the directional, low-contrast, and cross-modality nature of wheel-surface defects.

We introduce **RailDet-YOLO**, a YOLOv10n-based detector incorporating five purpose-built modules:
- **RWBottleneck**: a three-path gated bottleneck fusing partial convolution, asymmetric dilated depthwise convolution, and an identity path
- **RWC2f**: a railway-aware CSP block built on RWBottleneck
- **DSSA**: Dual Strip Squeeze Attention combining horizontal/vertical strip pooling with channel attention
- **ASF**: Adaptive Scale Fusion, a cross-conditioned spatial gate for the neck
- **DySample**: a learnable upsampler that preserves thin-crack detail

On **FaultSeg** (real-world wayside RGB imagery, 4 classes) and **WLI** (structured laser-stripe imagery, 2 classes), RailDet-YOLO reaches **96.2%** and **95.9% mAP@0.5**, and **72.3%** and **77.5% mAP@0.5:0.95**, respectively, outperforming all eight lightweight YOLO baselines (YOLOv5n–YOLOv12n and YOLO26) on both metrics while using only **2.49M parameters** and **8 GFLOPs**.

---

## ✨ Highlights

| Feature | Detail |
|---|---|
| 🏗️ **Base Architecture** | YOLOv10n |
| 🔢 **Parameters** | 2.497M |
| ⚡ **GFLOPs** | 8.0 |
| 🖥️ **GPU Inference** | ~4.9 ms (FaultSeg) / ~4.6 ms (WLI) |
| 💻 **CPU Inference** | ~23.8 ms (FaultSeg) / ~24.7 ms (WLI) |
| 📦 **Datasets** | FaultSeg (4 classes, RGB) + WLI (2 classes, Laser) |
| 🎯 **Best mAP@0.5** | 96.23% (FaultSeg) / 95.90% (WLI) |
| 🎯 **Best mAP@0.5:0.95** | 72.31% (FaultSeg) / 77.51% (WLI) |

---

## 🏛️ Architecture

RailDet-YOLO extends YOLOv10n with five domain-specific modules designed around the directional and low-contrast nature of railway wheel surface defects. The complete pipeline processes a `3 × H × W` RGB input through a hierarchical backbone, a multi-scale PAN-FPN neck, and a YOLOv10n decoupled detection head generating predictions at three feature pyramid levels (P3/P4/P5).

```
Input Image (640×640)
        │
   ┌────▼────────────────────────────────────────────┐
   │              BACKBONE                           │
   │   Conv → Conv → RWC2f → SCDown → RWC2f →      │
   │   SCDown → RWC2f → SCDown → RWC2f → SPPF →    │
   │   DSSA                                          │
   └────┬────────────────────────────────────────────┘
        │  P3 / P4 / P5
   ┌────▼────────────────────────────────────────────┐
   │          NECK (PAN-FPN)                         │
   │   DySample → ASF → RWC2f  (top-down)           │
   │   SCDown   → ASF → RWC2f  (bottom-up)          │
   └────┬────────────────────────────────────────────┘
        │  N3 / N4 / N5
   ┌────▼────────────────────────────────────────────┐
   │          HEAD (YOLOv10 Decoupled)               │
   │   P3/8 → Small defects (scratches, fine cracks) │
   │   P4/16 → Medium defects (spalling, shelling)   │
   │   P5/32 → Large defects                         │
   └─────────────────────────────────────────────────┘
```

### 🔧 Novel Modules

<details>
<summary><b>RWBottleneck — Railway-Aware Bottleneck</b></summary>

A three-branch gated bottleneck that captures directional defect characteristics:

- **Path A (PConv):** Partial convolution on 1/4 of channels for efficient spatial feature extraction
- **Path B (Asymmetric Dilated DWConv):** A `3×1` depthwise conv (dilation=1) capturing horizontal structures summed with a `1×3` depthwise conv (dilation=2) capturing vertical structures — enabling simultaneous modeling of longitudinal and transverse defect patterns
- **Path C (Identity):** Preserves original feature representation

The three branches are fused via learnable softmax-normalized scalar weights, reducing FLOP cost to **~17.4%** of a standard `3×3` bottleneck.

</details>

<details>
<summary><b>RWC2f — Railway CSP Block</b></summary>

Extends the conventional C2f block by replacing standard bottleneck units with RWBottleneck modules, propagating directional defect-awareness throughout the full feature extraction hierarchy while preserving CSP computational efficiency.

</details>

<details>
<summary><b>DSSA — Dual Strip Squeeze Attention</b></summary>

Replaces the PSA block at the end of the backbone. Three complementary attention branches operate in parallel:

- **Horizontal Strip Branch:** Adaptive avg-pool along height → `1×3` depthwise conv → Sigmoid
- **Vertical Strip Branch:** Adaptive avg-pool along width → `3×1` depthwise conv (dilation=2) → Sigmoid
- **SE Channel Branch:** Global avg-pool → FC → ReLU → FC → Sigmoid

The asymmetric receptive fields enable DSSA to model directional defects while preserving global contextual awareness.

</details>

<details>
<summary><b>ASF — Adaptive Scale Fusion</b></summary>

Replaces conventional concatenation-based fusion in the neck with a cross-conditioned spatial gating mechanism that generates pixel-wise adaptive fusion coefficients conditioned on the content of both the main and skip feature streams — unlike BiFPN's fixed learnable scalar weights.

</details>

<details>
<summary><b>DySample — Learnable Dynamic Upsampling</b></summary>

Replaces bilinear interpolation in the top-down neck pathway. A lightweight offset prediction network generates content-adaptive sampling offsets, which are applied via grid sampling and PixelShuffle. Adds less than **0.1%** additional parameters while preserving high-frequency structural details critical for thin crack localization.

</details>

---

## 📊 Results

### FaultSeg Dataset (RGB Wayside Camera — 4 Classes)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|
| YOLOv5n | 0.9125 | 0.8750 | 0.8369 | 0.6934 | 2.509 | 7.18 |
| YOLOv6n | 0.6473 | 0.6296 | 0.6597 | 0.4835 | 4.239 | 11.79 |
| YOLOv8n | **0.9309** | 0.8716 | 0.8644 | 0.6462 | 3.012 | 8.20 |
| YOLOv9t | 0.9167 | 0.9167 | 0.9017 | 0.6554 | 2.006 | 7.85 |
| YOLOv10n | 0.8957 | 0.8736 | 0.8750 | 0.6437 | 2.709 | 8.40 |
| YOLOv11n | 0.9173 | 0.8750 | 0.8578 | 0.6347 | 2.591 | 6.44 |
| YOLOv12n | 0.8675 | 0.8661 | 0.8571 | 0.6141 | 2.569 | 6.48 |
| YOLOv26 | 0.8833 | 0.9167 | 0.8732 | 0.6768 | 2.510 | 8.90 |
| Faster R-CNN | 0.8036 | 0.6537 | 0.8036 | 0.6112 | 41.31 | 200000 |
| SSD | 0.4983 | 0.3728 | 0.4983 | 0.3567 | 24.15 | 35200 |
| FCOS | 0.5827 | 0.4374 | 0.5827 | 0.3937 | 32.07 | 128000 |
| RetinaNet | 0.7657 | 0.6413 | 0.7657 | 0.5780 | 32.23 | 152000 |
| RTDETRv4-s | 0.7124 | 0.8010 | **0.9774** | 0.7124 | 10.37 | 24833 |
| **RailDet-YOLO (Ours)** | **0.9183** | **0.9583** | **0.9623** | **0.7231** | **2.497** | **8.00** |

### WLI Dataset (Structured Laser Stripe — 2 Classes)

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|
| YOLOv5n | 0.9410 | 0.9058 | 0.9153 | 0.7646 | 2.509 | 7.18 |
| YOLOv6n | 0.9395 | 0.9112 | 0.9159 | 0.7606 | 4.238 | 11.79 |
| YOLOv8n | 0.9387 | 0.9048 | 0.9107 | 0.7635 | 3.011 | 8.20 |
| YOLOv9t | 0.9245 | **0.9259** | 0.9154 | 0.7682 | 2.006 | 7.85 |
| YOLOv10n | 0.9342 | 0.9093 | 0.9067 | 0.7619 | 2.708 | 8.40 |
| YOLOv11n | 0.9197 | 0.9193 | 0.9115 | 0.7650 | 2.590 | 6.44 |
| YOLOv12n | 0.9325 | 0.9163 | 0.9159 | 0.7671 | 2.568 | 6.48 |
| YOLOv26 | **0.9659** | 0.8761 | 0.8696 | 0.6743 | 2.500 | 8.90 |
| Faster R-CNN | 0.9110 | 0.7438 | 0.9110 | 0.7047 | 41.30 | 200000 |
| SSD | 0.7225 | 0.5334 | 0.7225 | 0.4921 | 23.88 | 35200 |
| FCOS | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 32.07 | 128000 |
| RetinaNet | 0.8998 | 0.7468 | 0.8998 | 0.6970 | 32.19 | 152000 |
| RTDETRv4-s | 0.6839 | 0.7935 | **0.9697** | 0.6839 | 10.37 | 24833 |
| **RailDet-YOLO (Ours)** | 0.9335 | 0.9141 | **0.9590** | **0.7751** | **2.497** | **8.00** |

> **Bold** values indicate the best score per column. RailDet-YOLO achieves the highest mAP@0.5:0.95 on **both datasets** — the strictest localization metric — while maintaining the lowest GPU inference latency among all lightweight YOLO variants tested.

### Ablation Study (FaultSeg)

| Configuration | Params (M) | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| Baseline (YOLOv10n) | 2.710 | 0.8957 | 0.8736 | 0.8750 | 0.6437 |
| + RWC2f only | 2.120 | 0.9167 | 0.9518 | 0.9263 | 0.6774 |
| + DSSA only | 2.750 | 0.9045 | 0.9557 | 0.9002 | 0.6437 |
| + DySample only | 2.750 | 0.8919 | 0.9568 | 0.9477 | 0.6912 |
| + ASF only | 3.000 | 0.9089 | 0.9407 | 0.9201 | 0.6884 |
| **Full RailDet-YOLO** | **2.497** | **0.9183** | **0.9583** | **0.9623** | **0.7231** |

---

## 📁 Datasets

### FaultSeg
- **Source:** Wayside GoPro Hero 9 camera system at 2704×1520 resolution
- **Classes:** Cracks/Scratches, Discoloration, Shelling, Wheel
- **Split:** 70% Train / 20% Validation / 10% Test
- **Total images:** 14,110 (after augmentation)
- **Download:** [Zenodo](https://zenodo.org/records/13162335) | [Figshare](https://springernature.figshare.com/articles/dataset/FaultSeg_A_Dataset_for_Train_Wheel_Defect_Detection/27996866)
- **Dataset Paper:** [Scientific Data, Nature (2025)](https://doi.org/10.1038/s41597-025-04557-0)

### WLI (Wheel Laser Image)
- **Source:** Structured laser-stripe wheelset inspection system
- **Classes:** Spot, Fracture
- **Subset used:** Segmentation subset (7,627 images)
- **Download:** [Figshare](https://figshare.com/articles/figure/origin_zip/25040747/3)

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/SahilJatoi744/RailDet-YOLO.git
cd RailDet-YOLO

# Create a virtual environment (recommended)
conda create -n raildet python=3.10 -y
conda activate raildet

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
numpy>=1.24.0
opencv-python>=4.7.0
Pillow>=9.5.0
PyYAML>=6.0
tqdm>=4.65.0
matplotlib>=3.7.0
scipy>=1.10.0
```

---

## 🚀 Usage

### Dataset Preparation

Organise your dataset in YOLO format:

```
data/
├── faultseg/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── wli/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
```

Update the fault-seg dataset YAML:

```yaml
# data/faultseg.yaml
path: data/faultseg
train: images/train
val:   images/val
test:  images/test

nc: 4
names: ['Cracks-Scratches', 'Discoloration', 'Shelling', 'Wheel']
```

Update the WLI dataset YAML:

```yaml
# data/faultseg.yaml
path: data/faultseg
train: images/train
val:   images/val
test:  images/test

nc: 2
names: ['fracture', 'spot']
```

<!-- ### Training

```bash
# Train on FaultSeg
python train.py \
  --model configs/raildet_yolo.yaml \
  --data data/faultseg.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --optimizer AdamW \
  --lr0 0.01 \
  --weight-decay 0.0005 \
  --device 0

# Train on WLI
python train.py \
  --model configs/raildet_yolo.yaml \
  --data data/wli.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --optimizer AdamW \
  --lr0 0.01 \
  --weight-decay 0.0005 \
  --device 0
```

### Evaluation

```bash
# Evaluate on FaultSeg test set
python val.py \
  --model weights/raildet_faultseg.pt \
  --data data/faultseg.yaml \
  --imgsz 640 \
  --batch 16

# Evaluate on WLI test set
python val.py \
  --model weights/raildet_wli.pt \
  --data data/wli.yaml \
  --imgsz 640 \
  --batch 16
```

### Inference

```bash
# Single image inference
python detect.py \
  --model weights/raildet_faultseg.pt \
  --source path/to/image.jpg \
  --imgsz 640 \
  --conf 0.25

# Video / real-time stream inference
python detect.py \
  --model weights/raildet_faultseg.pt \
  --source path/to/video.mp4 \
  --imgsz 640 \
  --conf 0.25 \
  --save
```

### Benchmarking

```bash
# Run full efficiency benchmark (GPU + CPU throughput, FLOPs, latency)
python benchmark.py \
  --model weights/raildet_faultseg.pt \
  --data data/faultseg.yaml \
  --imgsz 640 \
  --warmup 300 \
  --runs 1000
``` -->

---

## 📦 Model Weights

Pre-trained weights will be released upon publication.

| Model | Dataset | mAP@0.5 | mAP@0.5:0.95 | Download |
|---|---|---|---|---|
| RailDet-YOLO | FaultSeg | 96.23% | 72.31% | *Coming soon* |
| RailDet-YOLO | WLI | 95.90% | 77.51% | *Coming soon* |

---

## 🖥️ Hardware

All experiments were conducted on:

| Component | Specification |
|---|---|
| CPU | AMD Ryzen Threadripper 9960X (24-Core) |
| RAM | 32 GB |
| GPU | NVIDIA RTX A6000 (96 GB VRAM) |
| Framework | PyTorch 2.0+ |
| Input Resolution | 640 × 640 |
| Training Epochs | 100 |
| Batch Size | 16 |

---

## 📈 Efficiency Comparison

RailDet-YOLO achieves the **lowest GPU inference latency** among all lightweight YOLO variants tested, and the **highest Efficiency Ratio** (mAP@0.5 × Throughput / FLOPs) among all non-NMS-free detectors on both datasets.

| Model | GPU Inf. (ms) | CPU Inf. (ms) | Params (M) | FLOPs (M) | ER (GPU) |
|---|---|---|---|---|---|
| YOLOv5n | 6.76 | 25.59 | 2.509 | 7179.93 | 0.017 |
| YOLOv10n | 6.29 | 24.76 | 2.709 | 8399.41 | 0.017 |
| RTDETRv4-s | 5.27 | 92.16 | 10.370 | 24832.81 | 0.007 |
| **RailDet-YOLO** | **4.91** | **23.84** | **2.497** | **8001.38** | **0.024** |

---

<!-- ## 🗂️ Repository Structure

```
RailDet-YOLO/
├── assets/                  # Architecture diagrams and visualizations
├── configs/
│   └── raildet_yolo.yaml    # Model configuration
├── data/                    # Dataset YAML files
│   ├── faultseg.yaml
│   └── wli.yaml
├── models/
│   ├── backbone/
│   │   ├── rwbottleneck.py  # RWBottleneck module
│   │   └── rwc2f.py         # RWC2f block
│   ├── neck/
│   │   ├── asf.py           # Adaptive Scale Fusion
│   │   └── dysample.py      # DySample upsampler
│   ├── attention/
│   │   └── dssa.py          # Dual Strip Squeeze Attention
│   └── raildet_yolo.py      # Full model definition
├── weights/                 # Pre-trained weights (released post-publication)
├── train.py                 # Training script
├── val.py                   # Evaluation script
├── detect.py                # Inference script
├── benchmark.py             # Efficiency benchmarking
├── requirements.txt
├── LICENSE
└── README.md
``` -->

---

## 📝 Citation

> **This paper is currently under review. The citation below will be updated with the final publication details upon acceptance.**

```bibtex
@article{jatoi2026raildet,
  title     = {RailDet-YOLO: Lightweight Railway Wheel Defect Detection
               with Direction-Aware and Representation Learning},
  author    = {Jatoi, Sahil and Islam, Muhammad Zubair and Abro, Bushra
               and Kim, Hyung Seok},
  journal   = {Scientific Reports},
  year      = {2026},
  publisher = {Nature Publishing Group},
  note      = {Under review}
}
```

If you use the **FaultSeg dataset**, please also cite:

```bibtex
@article{shaikh2025faultseg,
  title   = {FaultSeg: A Dataset for Train Wheel Defect Detection},
  author  = {Shaikh, Muhammad Zakir and others},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {1--12},
  year    = {2025},
  doi     = {10.1038/s41597-025-04557-0}
}
```

If you use the **WLI dataset**, please also cite:

```bibtex
@article{wang2024wli,
  title   = {A comprehensive laser image dataset for real-time measurement
             of wheelset geometric parameters},
  author  = {Wang, Y. and others},
  journal = {Scientific Data},
  year    = {2024},
  doi     = {10.1038/s41597-024-03288-y}
}
```

---

## 🤝 Acknowledgements

This work was supported by:

- **Ministry of Trade, Industry and Energy (MOTIE)**, Republic of Korea with Grant No. RS-2025-02311228
- **IITP**: Grant No. RS-2026-25516382
- **National Research Foundation of Korea (NRF)**: Grant No. RS-2025-19722970
- **Sejong University**, Department of Artificial Intelligence and Robotics, Seoul, Republic of Korea
- **NCRA (National Center of Robotics and Automation)**, MUET, Jamshoro, Pakistan

We thank the authors of [FaultSeg](https://doi.org/10.1038/s41597-025-04557-0) and [WLI](https://doi.org/10.1038/s41597-024-03288-y) for publicly releasing their datasets, which made this benchmarking study possible.

---

## 📜 License

This project is licensed under the MIT License see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ If you find this work useful, please consider starring the repository and citing the paper.**

</div>
