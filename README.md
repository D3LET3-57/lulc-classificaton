# Land Use and Land Cover (LULC) Classification Using Landsat 9 and Patch-Based CNNs

**DAT-103 | Mehta Family School of Data Science and Artificial Intelligence**

An end-to-end pipeline for LULC classification over Brampton, Ontario (Canada) using Landsat 9 Surface Reflectance imagery and multiple Convolutional Neural Network architectures.

---

## Project Overview

This project implements a complete workflow for multi-class Land Use / Land Cover classification from medium-resolution satellite imagery. The pipeline covers:

1. **AOI Definition** -- Programmatic study area setup with bounding box geometry
2. **Satellite Data Retrieval** -- Automated Landsat 9 scene download via the USGS Machine-to-Machine (M2M) API
3. **Preprocessing and Feature Engineering** -- Radiometric calibration, spatial clipping, NDVI computation, and ESRI 10 m LULC label integration
4. **Exploratory Visual Analytics** -- Band composites, spectral histograms, class distributions, and interactive maps
5. **Multi-Architecture CNN Training** -- Three CNN models trained and benchmarked under identical conditions
6. **Quantitative Evaluation** -- Multi-metric evaluation including accuracy, F1, confusion matrices, ROC curves, and spatial agreement analysis

### CNN Architectures Compared

| Model | Description | Key Characteristic |
|---|---|---|
| Custom Patch-Based CNN | Compact 3-block encoder designed for 7x7 patches | Best accuracy with fewest parameters |
| VGG-Style Patch CNN | 4-layer VGG-inspired architecture with paired convolutions | Deeper non-residual baseline |
| Patch-Adapted ResNet18 | Modified ResNet18 for 5-channel multispectral input | Deepest model (~11M params) |

### Key Results

| Metric | Custom CNN | VGG-style CNN | ResNet18 |
|---|---|---|---|
| Overall Accuracy | **0.9507** | 0.9308 | 0.9261 |
| Weighted F1 | **0.9502** | 0.9302 | 0.9255 |

The Custom CNN was selected as the primary model due to its superior efficiency-to-accuracy trade-off, outperforming deeper architectures by approximately 2--2.5 percentage points while maintaining a fraction of their parameter count.

---

## Project Structure

```
lulc_notebook_project/
├── notebooks/                          # Jupyter notebooks (run sequentially)
│   ├── 01_setup_aoi.ipynb              # AOI definition and project configuration
│   ├── 02_usgs_download_and_band_visuals.ipynb  # USGS M2M data retrieval
│   ├── 03_preprocess_ndvi_lulc_visuals.ipynb    # Preprocessing and NDVI
│   ├── 04_extra_visualizations.ipynb   # Exploratory visual analytics
│   ├── 05_cnn_training.ipynb           # Model training (Custom CNN + VGG + ResNet)
│   ├── 06_best_model_analysis.ipynb    # Best model detailed evaluation
│   └── 06_multimodal_analysis.ipynb    # Multi-model comparison
├── data/
│   ├── config.json                     # Central path and AOI configuration
│   ├── aoi/                            # AOI GeoJSON files
│   ├── landsat/                        # Downloaded Landsat scenes
│   └── processed/                      # Preprocessed arrays and predictions (.npz)
├── outputs/
│   ├── figures/                        # Generated plots and visualizations
│   └── models/                         # Saved model weights (.pth)
└── README.md                           # This file
```

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Google Colab (recommended) or a local environment with GPU support
- A USGS EarthExplorer account for M2M API access

### Environment Setup

#### Option 1: Google Colab (Recommended)

1. Upload the project folder to Google Drive or clone the repository:
   ```bash
   git clone https://github.com/D3LET3-57/lulc-classificaton.git
   ```
2. Open the notebooks in Colab and run them sequentially (01 through 06).
3. When prompted, set your USGS credentials as environment variables:
   ```python
   import os
   os.environ['USGS_USERNAME'] = 'your_username'
   os.environ['USGS_APP_TOKEN'] = 'your_token'
   ```

#### Option 2: Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/D3LET3-57/lulc-classificaton.git
   cd lulc-classificaton
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows

   pip install numpy pandas matplotlib seaborn scikit-learn
   pip install torch torchvision
   pip install rasterio geopandas shapely
   pip install jupyter nbconvert
   pip install pystac-client planetary-computer
   ```

3. Set USGS credentials:
   ```bash
   export USGS_USERNAME="your_username"
   export USGS_APP_TOKEN="your_token"
   ```

4. Run notebooks sequentially:
   ```bash
   cd notebooks
   jupyter notebook
   ```
   Open and execute each notebook in order: `01` through `06`.

---

## Notebook Descriptions

| Notebook | Description |
|---|---|
| `01_setup_aoi` | Defines the AOI bounding box over Brampton, Ontario. Creates project directories and exports `config.json` with all path definitions. |
| `02_usgs_download_and_band_visuals` | Authenticates with the USGS M2M API, searches for Landsat 9 scenes, downloads the best scene (lowest cloud cover), and visualizes raw spectral bands. |
| `03_preprocess_ndvi_lulc_visuals` | Clips Landsat bands to AOI, applies radiometric scaling to surface reflectance, computes NDVI, downloads ESRI 10 m LULC labels, and resamples to the Landsat grid. |
| `04_extra_visualizations` | Generates exploratory visualizations: band composites, spectral histograms, NDVI boxplots by class, correlation matrices, and class distribution charts. |
| `05_cnn_training` | Builds the `LULCPatchDataset`, performs stratified train/val/test split, trains the Custom CNN with hyperparameter tuning, then trains VGG-style and ResNet18 models. Exports prediction `.npz` files. |
| `06_best_model_analysis` | Detailed evaluation of the best (Custom CNN) model: confusion matrix, ROC curves, F1/IoU bar plots, spatial agreement map, and prediction visualization. |
| `06_multimodal_analysis` | Side-by-side comparison of all three CNN architectures: accuracy bar chart, confusion matrices, and classification reports. |

---

## Study Area

- **Location:** Brampton, Ontario, Canada
- **AOI Area:** ~52.6 km²
- **Coordinates:** 43.23 N -- 43.29 N, 79.87 W -- 79.96 W
- **Sensor:** Landsat 9 Collection 2 Level-2 Surface Reflectance
- **Bands Used:** B2 (Blue), B3 (Green), B4 (Red), B5 (NIR) + NDVI
- **LULC Classes:** Water, Trees, Flooded Vegetation, Crops, Built Area, Rangeland

---

## Team

| Name | Roll No. | Contribution |
|---|---|---|
| D.M. Kishan | 24125010 | AOI selection, project setup, configuration pipeline (NB 01) |
| Saketh Raj | 24125025 | USGS M2M data retrieval, scene filtering, band visualization (NB 02) |
| Ajay Nayak | 24125015 | Preprocessing: clipping, NDVI, ESRI LULC label integration (NB 03) |
| V.V. Akhil | 24125039 | Exploratory geospatial analysis, spectral visualization (NB 04) |
| Lalith Sai | 24125009 | CNN architecture design, training, evaluation, report (NB 05--06) |

---

## License

This project was developed as part of the DAT-103 coursework at the Mehta Family School of Data Science and Artificial Intelligence.
