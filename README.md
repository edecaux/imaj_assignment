# Large-Scale Machine Learning–Based Image Analysis of UNESCO/IMAJ Children’s Drawings

This project was developed as a **Semester Project** at the **EPFL Center for Imaging** in collaboration with **IMAJ** (Institut Mondial des Arts de la Jeunesse). 

The goal is to conduct a large-scale visual analysis of over 100,000 children's drawings (1985–2021) to explore how age, culture, and time influence graphic expression through machine learning.

## Project Overview
The repository contains a complete pipeline for:
1. **Data Curation**: Filtering, border removal, and standardization of 105,816 images (274 GB).
2. **Feature Extraction**: Pixel-based analysis (color palettes, spatial segmentation).
3. **Advanced Vision Tasks**: Object detection with YOLOv5 and AI-driven image captioning.
4. **Interactive Visualization**: A specialized tool to explore the database and its features.

## Key Features

### 1. Data Curation & Standardization
- **Outlier Detection**: Statistical analysis of dimensions and aspect ratios to filter unusable scans.
- **Image Processing**: Automatic cropping of black borders and resizing using cubic spline interpolation for uniform analysis.

### 2. Feature Extraction Methods
- **Color Quantization**: Implementation of the **Wu algorithm** and **K-means** to identify dominant color palettes and measure visual complexity.
- **Spatial Segmentation**: Testing of **SLIC (Simple Linear Iterative Clustering)** to generate superpixels and distinguish foreground elements.
- **Object Detection**: Fine-tuning of **YOLOv5** specifically for artistic/symbolic drawings (e.g., detecting "fish" in water-themed entries).

### 3. AI Agent for Image Captioning
- Experimental integration of **Llama-4-Maverick-17B** (via EPFL RCP) to evaluate the model's ability to describe artistic content and capture high-level semantic features.

### 4. Data Integration & Interactive Tool
- **Centralized Database**: All extracted features (colors, objects, metadata) are stored in a standardized CSV format.
- **Streamlit Interface**: A web-based GUI to interactively filter and visualize image distributions (e.g., height distribution by year).

## Tech Stack
- **Languages**: Python
- **Libraries**: `scikit-learn`, `scikit-image`, `opencv-python`, `pillow`, `pandas`, `numpy`
- **Vision Models**: `YOLOv5`, `Llama-4` (multimodal experiments)
- **Web App**: `Streamlit`
- **Infrastructure**: Remote server (SSH), EPFL RCP Portal (GPU acceleration)

## Repository Structure
```text
├── Color Quantization/          # Color analysis and quantization
│   ├── color_quantization.ipynb # Notebook for color quantization experiments
│   ├── colors_utils.py          # Utility functions for color processing
│   └── run_color_quant_database.py  # Batch processing script
├── Dataset Exploration and Curation/  # Data analysis and preprocessing
│   ├── outlier_removal.ipynb    # Outlier detection and removal
│   ├── standardize.ipynb        # Dataset standardization
│   └── statistical_analysis.ipynb  # Statistical analysis and visualization
├── dataset_eau/                 # Training dataset for object detection
│   ├── data.yaml                # YOLO dataset configuration
│   ├── images/train/            # Training images
│   └── labels/train/            # YOLO format annotations
├── runs/detect/                 # YOLOv5 training and inference results
│   ├── exp*/                    # Inference results on test images
│   └── train/                   # Training artifacts and checkpoints
│       ├── weights/             # Model weights (best.pt, last.pt)
│       ├── weights100/          # Checkpoint at epoch 100
│       └── weights1000/         # Checkpoint at epoch 1000
├── filtered_database.csv        # Curated metadata file
├── image_captioning.ipynb       # LLM-based image captioning experiments
├── interactive_plot.py          # Streamlit visualization tool
├── object_detection.ipynb       # YOLOv5 fine-tuning and evaluation
├── spatial_segementation.ipynb  # Spatial segmentation experiments
├── yolov5s.pt                   # Pre-trained YOLOv5 base model
└── README.md
