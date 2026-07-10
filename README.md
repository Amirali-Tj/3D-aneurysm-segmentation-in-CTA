# 3D Aneurysm Segmentation in CTA Images

## 📌 Project Overview
An **intracranial aneurysm** is a pathological dilation or "bulge" in a cerebral artery wall, occurring in approximately 3% of the general population. While often asymptomatic, their rupture leads to **subarachnoid hemorrhage (SAH)**, a catastrophic event with a mortality rate near 40% and high risk of long-term disability.

This repository focuses on the automated **3D segmentation of intracranial aneurysms** from Computed Tomography Angiography (CTA) images. Our research investigates how different data preprocessing and augmentation strategies—specifically **intensity windowing** and **patch size variations**—impact the segmentation performance of deep learning models on the hetrogeneous dataset.

## 🚀 Key Features
- **3D Segmentation Pipeline:** Optimized for volumetric medical data.
- **Windowing Augmentation Module:** 
  - **Random Windowing:** Dynamic adjustment of Window Level (WL) and Window Width (WW) during training.
  - **Multi-Windowing:** Multi-channel input strategy incorporating different clinical windows (e.g., Bone window vs. Vascular window).
- **Patch-Based Training:** Comprehensive evaluation of different 3D patch sizes and their effects on model performance

