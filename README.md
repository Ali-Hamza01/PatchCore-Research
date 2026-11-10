# PatchCore-Based Industrial Anomaly Detection

An unsupervised industrial anomaly detection project using **PatchCore** and the **MVTec AD dataset**. This project uses a pretrained Wide ResNet50-2 backbone to identify anomalous regions in industrial images and generate anomaly prediction maps.

Four MVTec AD categories have been evaluated: Carpet, Bottle, Tile, and Wood.

## 1. Project Overview

This project explores PatchCore, an unsupervised anomaly detection method that learns visual features from normal training images and identifies deviations during testing.

### Objectives

- Implement PatchCore for industrial anomaly detection.
- Extract visual features using a pretrained CNN.
- Build a memory bank of normal image features.
- Detect anomalous images and defective regions.
- Generate anomaly maps and prediction visualizations.
- Evaluate performance using image-level and pixel-level metrics.

## 2. Dataset

**Dataset:** MVTec Anomaly Detection (MVTec AD)

Official dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad

Four categories were evaluated:

- Carpet
- Bottle
- Tile
- Wood

**Dataset limitation:** Standard MVTec AD provides RGB images, not paired RGB-NIR images. Therefore, the current experiments are RGB-based and do not implement RGB-NIR fusion.

## 3. Methodology

The project uses PatchCore for unsupervised anomaly detection.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model | PatchCore |
| Backbone | Wide ResNet50-2 |
| Feature layers | layer2, layer3 |
| Pretrained weights | Enabled |
| Coreset sampling ratio | 0.1 |
| Number of neighbors | 9 |
| Accelerator | CPU |
| Maximum epochs | 1 |

### Workflow

1. Load normal training images from MVTec AD.
2. Extract features using pretrained Wide ResNet50-2.
3. Select representative features using coreset sampling.
4. Build a memory bank of normal features.
5. Compare test image features against the memory bank.
6. Calculate anomaly scores and generate anomaly maps.
7. Evaluate image-level and pixel-level performance.

### Architecture

```text
       MVTec AD Dataset
               |
               v
       Normal Training Images
               |
               v
       Wide ResNet50-2
        Feature Extractor
               |
               v
        Feature Extraction
               |
               v
         Coreset Sampling
               |
               v
       PatchCore Memory Bank
               |
               v
           Test Images
               |
               v
      Nearest Neighbor Search
               |
               v
        Anomaly Detection
               |
          +----+----+
          |         |
          v         v
      Image Score  Anomaly Map
          |         |
          +----+----+
               |
               v
        Evaluation Metrics
```

## 4. Experimental Results

PatchCore was evaluated on four selected MVTec AD categories.

### Final Results

| Category | Image AUROC | Image F1-Score | Pixel AUROC | Pixel F1-Score |
|---|---:|---:|---:|---:|
| Carpet | 98.72% | 97.18% | 99.07% | Not reported |
| Bottle | 100.00% | Not reported | 98.57% | 72.72% |
| Tile | 100.00% | 99.40% | 95.54% | 61.92% |
| Wood | 98.51% | 95.73% | 93.20% | 46.77% |

**Important:** All values above are taken from the actual experiment terminal outputs. Metrics marked `Not reported` were not displayed in the corresponding output and have not been estimated.

### Evaluation Metrics

- **Image AUROC:** Measures the ability to distinguish normal images from anomalous images.
- **Image F1-Score:** Evaluates image-level classification performance using precision and recall.
- **Pixel AUROC:** Measures the ability to distinguish anomalous pixels from normal pixels.
- **Pixel F1-Score:** Evaluates pixel-level defect localization using precision and recall.

### Preliminary Observations

- Carpet achieved 99.07% Pixel AUROC.
- Bottle and Tile achieved 100% Image AUROC.
- Wood achieved 98.51% Image AUROC.
- Pixel-level F1 performance varies across categories, indicating opportunities to improve defect localization.

These are preliminary results from four selected categories, not a complete 15-category MVTec AD benchmark.

## 5. Project Structure

```text
PatchCore-Research/
│
├── train_patchcore.py
├── results/
│   └── Patchcore/
│       └── MVTecAD/
│           ├── carpet/
│           ├── bottle/
│           ├── tile/
│           └── wood/
│
├── Lib/
├── Scripts/
├── pyvenv.cfg
└── README.md
```

The results directory contains generated experiment outputs, including model checkpoints and prediction visualizations.

## 6. Requirements

- Windows 10 or later
- Python 3.12
- UV package manager
- Anomalib
- PyTorch
- Lightning
- MVTec AD dataset

Experiments were performed using CPU execution.

## 7. Setup and Execution

### Step 1: Open the project directory

```powershell
cd G:\AI-Research\PatchCore-Research
```

### Step 2: Configure the dataset path

In `train_patchcore.py`, configure:

```python
DATASET_PATH = Path(r"G:\AI-Research\Dataset\mv-tec ad")
```

### Step 3: Select a category

Change the category variable:

```python
CATEGORY = "carpet"
```

Available categories:

```python
"carpet"
"bottle"
"tile"
"wood"
```

### Step 4: Run the experiment

```powershell
uv run python train_patchcore.py
```

The script performs training and testing and saves generated outputs in the results directory.

**Important:** Run one category at a time and change the `CATEGORY` variable before starting another experiment.

## 8. Output Files

The experiment generates:

- Trained PatchCore model checkpoints.
- Prediction images.
- Anomaly maps and heatmaps.
- Evaluation metrics.
- Experiment logs.

Default output directory:

```text
G:\AI-Research\PatchCore-Research\results
```

## 9. Limitations

- Only four of the fifteen MVTec AD categories were evaluated.
- Experiments were performed using CPU, resulting in longer execution times.
- The current implementation uses RGB images only.
- No paired NIR or hyperspectral data was used.
- RGB-NIR fusion has not yet been implemented.
- Pixel-level defect localization requires further investigation, particularly for categories with lower Pixel F1-Scores.

## 10. Future Work

1. Evaluate PatchCore on additional MVTec AD categories.
2. Explore alternative anomaly detection architectures.
3. Improve pixel-level defect localization.
4. Obtain a suitable paired RGB-NIR industrial dataset.
5. Develop RGB-NIR feature fusion methods.
6. Compare RGB-only, NIR-only, and RGB-NIR fusion approaches.
7. Investigate multimodal anomaly detection for industrial surface inspection.

## 11. Research Direction

This project provides an initial RGB-based unsupervised anomaly detection baseline for industrial surface inspection.

The next research direction is to investigate multimodal defect detection by combining RGB and Near-Infrared information. This could help determine whether complementary spectral information improves detection of defects that are difficult to identify using conventional RGB imaging alone.

## 12. Acknowledgements

- MVTec AD dataset for industrial anomaly detection research.
- Anomalib framework for anomaly detection implementation.
- PatchCore methodology for memory-bank-based anomaly detection.

---

**Project Status:** Four-category preliminary RGB-based PatchCore experiments completed.

**Research Direction:** RGB-NIR multimodal industrial surface defect detection.