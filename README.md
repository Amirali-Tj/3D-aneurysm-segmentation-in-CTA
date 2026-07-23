# 🧠 3D Intracranial Aneurysm Segmentation in CTA

A deep learning framework for **3D intracranial aneurysm segmentation** from **Computed Tomography Angiography (CTA)** using a **3D U-Net** architecture. This project investigates how different **CT windowing strategies** and **3D patch sizes** influence segmentation performance.

---

## 📖 Overview

Intracranial aneurysm segmentation is a challenging medical image analysis task due to the small size of aneurysms, complex vascular anatomy, and the low contrast of CTA images.

Since **CT windowing** directly affects the visibility of anatomical structures, this project explores how different windowing approaches impact the performance of a 3D U-Net.

Before any windowing operation, each CTA volume is **cropped** to focus on the region of interest and remove unnecessary background. The cropped volume is then processed using one of three windowing strategies before being divided into 3D patches for training.

In addition, we investigate how the amount of spatial context, controlled through different patch sizes, influences segmentation quality.

---

## 📂 Dataset Structure

The dataset is organized on a **per-patient** basis. Each patient has a dedicated folder containing the CTA volume and one or more corresponding aneurysm segmentation masks.

```text
Dataset/
├── Patient_001/
│   ├── Patient_001.nii
│   ├── Patient_001_Small_MCA_Saccular_label.nii
│   └── Patient_001_Large_ACOM_Fusiform_label.nii
│
├── Patient_002/
│   ├── Patient_002.nii
│   └── Patient_002_Medium_ICA_Saccular_label.nii
│
└── ...
```

### 📄 File Naming Convention

**CTA Volume**

```text
PatientName.nii
```

**Segmentation Label**

```text
PatientName_<LesionSize>_<LesionLocation>_<LesionShape>_label.nii
```

Where:

| Field | Description |
|-------|-------------|
| `PatientName` | Unique patient identifier |
| `LesionSize` | Size category of the aneurysm |
| `LesionLocation` | Anatomical location of the aneurysm |
| `LesionShape` | Morphological type of the aneurysm |
| `label` | Binary segmentation mask |

### 📝 Notes

- 📁 Each patient folder contains **one CTA volume**.
- 🧠 A patient may have **one or multiple aneurysms**.
- 🏷️ Each aneurysm is stored as a **separate segmentation label**.
- 📌 Lesion metadata (**size**, **location**, and **shape**) is encoded directly in the label filename, making it easy to filter or analyze specific aneurysm characteristics.

---

## 🚀 Pipeline

Every experiment follows the same preprocessing workflow:

> **Original CTA Volume → ✂️ Crop → 🪟 Windowing → 📦 Patch Extraction → 🧠 3D U-Net → 🎯 Segmentation**

The only difference between the three pipelines is the windowing strategy.

---

### 🔹 1. Single Window

A single predefined CT window is applied to the cropped CTA volume before patch extraction.

```text
Original CTA Volume
        │
   ✂️ Crop Volume
        │
 Single CT Window
        │
 Patch Extraction
        │
     3D U-Net
        │
 Segmentation Mask
```

---

### 🔹 2. Predefined Multi-Window

Multiple clinically relevant CT windows are generated from the cropped volume and stacked as multiple input channels.

```text
Original CTA Volume
        │
   ✂️ Crop Volume
        │
   ┌────┼────┐
   │    │    │
 W1    W2    W3
   │    │    │
   └────┴────┘
 Channel Stacking
        │
 Patch Extraction
        │
     3D U-Net
        │
 Segmentation Mask
```

---

### 🔹 3. Random Windowing 🎲

Instead of fixed window settings, random CT window parameters are generated during training to perform intensity augmentation and improve model robustness.

```text
Original CTA Volume
        │
   ✂️ Crop Volume
        │
Random Window Generator
        │
 Patch Extraction
        │
     3D U-Net
        │
 Segmentation Mask
```

---

## 📦 Patch Size Investigation

To evaluate the influence of spatial context, each pipeline was trained using three different volumetric patch sizes.

| Patch Size | Description |
|------------|-------------|
| **32 × 32 × 32** | Fine local anatomical details with limited context |
| **64 × 64 × 64** | Balanced local detail and global context |
| **128 × 128 × 128** | Larger anatomical context with higher memory requirements |

This comparison helps analyze the trade-offs between:

- Segmentation accuracy
- Spatial context
- GPU memory consumption
- Computational efficiency

---

## 🏗️ Model

The segmentation network is based on a **3D U-Net with Resnet Backbone and SEs block** architecture for volumetric medical image segmentation.

---

## 🧪 Experiments

The project compares three CT windowing strategies across three different patch sizes.

### 🪟 Windowing Strategies

- ✅ Single Window
- ✅ Predefined Multi-Window
- ✅ Random Windowing

### 📐 Patch Sizes

- 32 × 32 × 32
- 64 × 64 × 64
- 128 × 128 × 128

The objective is to understand how preprocessing and spatial context jointly affect intracranial aneurysm segmentation.

---

## ✨ Features

- Automatic volume cropping
- Single-window preprocessing
- Multi-window channel stacking
- Random CT window augmentation
- Patch-based volumetric training
- Configurable 3D patch sizes
- Comparative experiment framework
- Intracranial aneurysm CTA segmentation

---

## ⚙️ Requirements

Typical dependencies include:

- Python
- PyTorch
- NumPy
- nibabel
- SciPy
- matplotlib

Install the required packages using:

```bash
pip install -r requirements.txt
```
---
