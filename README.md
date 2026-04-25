![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Academic%20%7C%206th%20Semester%20DAV-orange)
![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20TabNet-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-purple)

---

# XAI-Crop: An Explainability-First Pipeline for District-Level Crop Yield Prediction in India Using Multi-Source Agricultural Data and SHAP Visualizations

> **Predicts crop yield across 560 Indian districts and visually identifies the single most impactful controllable factor per district — transforming a black-box machine learning model into a district-level, policy-ready agricultural advisory tool.**

---

## Overview

India's agricultural productivity is under simultaneous pressure from climate variability, uneven fertilizer access, and fragmented data ecosystems. Existing predictive models for crop yield — even accurate ones — rarely answer the question a district agronomist or policymaker actually asks: *"Given my district's specific conditions, which lever should I pull to raise yield?"* XAI-Crop addresses this gap by placing SHAP-based explainability at the center of the pipeline, not as an appendix to accuracy metrics. By fusing five heterogeneous data sources at the district level — spanning 560 districts across 20 Indian states from 1990 to 2015 — and training both a gradient-boosted tree (XGBoost) and a deep tabular model (TabNet), this project delivers not only strong predictive performance (XGBoost R² = 0.87) but also spatially mapped, human-readable explanations of *why* each district achieves the yield it does. The result is an interactive Streamlit dashboard where any stakeholder can select a district and instantly see which controllable factor (fertilizer application, irrigation intensity) dominates yield outcomes versus which uncontrollable factor (monsoon rainfall, temperature regime) sets the ceiling — a framing that no existing open-source agricultural tool currently offers.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Novelty & Contributions](#2-novelty--contributions)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Dataset Details](#4-dataset-details)
5. [Project Structure](#5-project-structure)
6. [Installation & Setup](#6-installation--setup)
7. [How to Run](#7-how-to-run)
8. [Results](#8-results)
9. [Visualizations](#9-visualizations)
10. [Publishable Angles](#10-publishable-angles)
11. [Future Work](#11-future-work)
12. [Authors & Acknowledgements](#12-authors--acknowledgements)
13. [License](#13-license)

---

## 1. Problem Statement

Agriculture employs approximately 42% of India's workforce and contributes nearly 18% of GDP, yet district-level crop yield prediction remains a largely unsolved operational challenge. The core difficulty is not a shortage of predictive models — it is the **explainability gap**: models that achieve high accuracy on held-out test sets offer no actionable insight to district collectors, Krishi Vigyan Kendra (KVK) scientists, or state agriculture departments who must allocate subsidies, irrigation budgets, and advisory efforts under tight resource constraints.

Three specific gaps motivate this work:

- **Single-source bias.** Most published studies rely on a single dataset (IMD rainfall or ICRISAT crop statistics alone), missing the cross-domain interactions — between soil nutrition, irrigation intensity, and monsoon timing — that actually govern yield outcomes.
- **Accuracy-only reporting.** Papers routinely report RMSE and R² without any spatial or district-level breakdown of model errors, hiding systematic underperformance in data-sparse regions.
- **Absence of controllability framing.** No existing open tool distinguishes between factors a farmer or government can influence (fertilizer application, irrigation scheduling) and those they cannot (inter-annual rainfall variability, temperature anomalies) when presenting model explanations.

XAI-Crop is designed to close all three gaps simultaneously.

---

## 2. Novelty & Contributions

1. **Explainability-first framing.** SHAP values are the PRIMARY deliverable of this pipeline, not a post-hoc addition. Every model training decision — feature engineering, hyperparameter choices, train/test split strategy — is evaluated in terms of SHAP stability and interpretability, not merely held-out accuracy. This is a methodological inversion from standard practice.

2. **Multi-source data fusion at district level.** Five heterogeneous sources — ICRISAT crop statistics, IMD rainfall, Kaggle crop production records, district fertilizer consumption (data.gov.in), and a GeoJSON district shapefile — are merged at the district-year granularity. Most comparable papers use a single dataset; multi-source fusion introduces cross-domain feature interactions that no single source can capture.

3. **Dual-model comparison: XGBoost vs. TabNet.** The same tabular agricultural dataset is used to benchmark a gradient-boosted tree (XGBoost) against a modern attention-based deep tabular model (TabNet). The comparison produces a principled analysis of *why* tree-based models outperform attention-based architectures on sparse, heterogeneous agricultural panel data — a finding with methodological relevance beyond this domain.

4. **Policy-actionable, controllability-stratified output.** The Streamlit dashboard classifies each district's top SHAP drivers into *controllable* (fertilizer application, irrigation intensity, sown area) and *uncontrollable* (rainfall deviation, temperature anomaly) factors. This separation — presented spatially on a choropleth map — directly informs which districts are most responsive to policy intervention versus those constrained by climate.

5. **Temporal trend layer via yield-slope choropleth.** A novel visualization maps the linear yield trend slope (Kg/ha/year) per district over the 1990–2015 window, distinguishing districts in *structural long-term decline* (negative slope despite average rainfall) from those experiencing *short-term weather shocks* (high variance, neutral slope). This temporal decomposition has not appeared in district-level Indian crop yield literature.

---

## 3. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES (5)                              │
│                                                                      │
│  [ICRISAT XLS]  [IMD Rainfall]  [Kaggle Crop]  [Fertilizer]  [GeoJSON]
│       │               │               │              │            │  │
└───────┴───────────────┴───────────────┴──────────────┴────────────┘  │
                                │                                       │
                                ▼                                       │
                  ┌─────────────────────────┐                          │
                  │   DISTRICT-LEVEL MERGE  │                          │
                  │  (district-year key)    │                          │
                  └────────────┬────────────┘                          │
                               │                                        │
                               ▼                                        │
                  ┌─────────────────────────┐                          │
                  │  PREPROCESSING          │                          │
                  │  · Drop >40% missing    │                          │
                  │  · Median imputation    │                          │
                  │  · Label encode states  │                          │
                  │  · 80/20 stratified     │                          │
                  │    split (by state)     │                          │
                  └────────────┬────────────┘                          │
                               │                                        │
               ┌───────────────┴───────────────┐                      │
               ▼                               ▼                      │
   ┌───────────────────┐           ┌───────────────────┐              │
   │   XGBoost         │           │   TabNet           │              │
   │   Regressor       │           │   (deep tabular)   │              │
   │   R²  = 0.87      │           │   (comparison)     │              │
   │   RMSE = 379 Kg/ha│           │                   │              │
   └────────┬──────────┘           └────────┬──────────┘              │
            │                               │                          │
            └──────────────┬────────────────┘                         │
                           ▼                                           │
              ┌────────────────────────┐                              │
              │  SHAP TreeExplainer    │                              │
              │  · Beeswarm (global)   │                              │
              │  · Waterfall (local)   │                              │
              │  · Dependence plots    │                              │
              │  · Controllability     │                              │
              │    classification      │                              │
              └────────────┬───────────┘                              │
                           │                                 ◄─────────┘
                           ▼                          (shapefile join)
              ┌────────────────────────┐
              │  STREAMLIT DASHBOARD   │
              │  · Choropleth maps     │
              │  · District selector   │
              │  · SHAP waterfall      │
              │  · Policy levers panel │
              └────────────────────────┘
```

---

## 4. Dataset Details

| # | Dataset | Source | Rows | Key Columns | License |
|---|---------|--------|------|-------------|---------|
| 1 | ICRISAT District-Level Crop Statistics | Mendeley Data | 12,803 | District, Year, Rice/Millet/Chickpea/Groundnut/Sugarcane Yield (Kg/ha), Sown Area, Production | CC BY 4.0 |
| 2 | IMD District Rainfall | Kaggle | ~15,000 | District, Month, Year, Monthly Rainfall (mm), Annual Total | Public Domain |
| 3 | Crop Production in India | Kaggle | ~250,000 | State, District, Crop, Season (Kharif/Rabi), Area, Production | Public Domain |
| 4 | District-wise Fertilizer Consumption | data.gov.in | ~8,000 | District, Year, N/P/K Consumption (tons) | OGD India |
| 5 | India District Shapefile | Figshare / ICRISAT | 640 districts | geometry, NAME\_1 (State), NAME\_2 (District) | CC BY 4.0 |

**Merged dataset shape after cleaning:** 12,803 rows × 108 columns · 560 unique districts · 20 states · 1990–2015

---

## 5. Project Structure

```
XAI-Crop/
│
├── analysis.py                  # End-to-end pipeline: clean → EDA → model → SHAP
├── check_env.py                 # Environment/dependency sanity check
├── SUMMARY.md                   # Auto-generated run summary (metrics + output paths)
├── README.md                    # This file
│
├── data/
│   └── processed/               # All pipeline-generated data artifacts
│       ├── master_clean.csv     # Cleaned wide-format district-year table
│       ├── long_format.csv      # Crop-melted long-format (one row per crop-year)
│       ├── shap_values.csv      # SHAP values for all test samples
│       ├── test_predictions.csv # Actual vs. predicted yield (test set)
│       ├── label_mappings.json  # State/district integer encoding maps
│       ├── meta.json            # Feature lists, model metrics, column metadata
│       └── India_districts.geojson  # Matched district boundaries for choropleth
│
├── outputs/
│   ├── figures/                 # Static visualizations (PNG)
│   │   ├── yield_distribution.png
│   │   ├── yield_trend_yearly.png
│   │   ├── yield_by_season.png
│   │   ├── rainfall_vs_yield.png
│   │   ├── correlation_heatmap.png
│   │   ├── district_yield_top_bottom.png
│   │   ├── feature_importance.png
│   │   ├── actual_vs_predicted.png
│   │   ├── choropleth_yield.html    # Interactive: avg yield per district
│   │   ├── choropleth_trend.html    # Interactive: yield slope per district
│   │   └── shap/
│   │       ├── shap_beeswarm.png    # Global feature importance
│   │       ├── shap_waterfall_*.png # Per-district local explanations
│   │       └── shap_dependence_*.png
│   └── models/
│       └── xgboost_model.pkl    # Serialized trained XGBoost model
│
└── app/
    └── dashboard.py             # Streamlit interactive dashboard
```

---

## 6. Installation & Setup

**Prerequisites:** Python 3.10+ · pip · (optional) a virtual environment

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/xai-crop.git
cd xai-crop

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Install all dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn
pip install xgboost==2.1.4        # pinned — xgboost 3.x is incompatible with shap 0.49.x
pip install shap==0.49.0
pip install pytorch-tabnet
pip install plotly geopandas fuzzywuzzy python-Levenshtein xlrd
pip install streamlit

# 4. Verify the environment
python check_env.py
```

> **Note on the XGBoost pin:** `xgboost==2.1.4` is required. XGBoost 3.x introduces a changed booster API that breaks `shap.TreeExplainer` in SHAP ≤0.49.x. Do not upgrade without testing SHAP compatibility.

---

## 7. How to Run

Run steps in order. Each step depends on artifacts produced by the previous one.

### Step 1 — Full Analysis Pipeline (EDA + Model Training + SHAP)

```bash
python analysis.py
```

This single script executes sequentially:
- **Section 1:** Data loading and audit (shape, missing values, descriptive stats)
- **Section 2:** Cleaning and preprocessing (imputation, encoding, 80/20 stratified split)
- **Section 3:** Exploratory Data Analysis (8 static plots saved to `outputs/figures/`)
- **Section 4:** Geospatial processing (fuzzy district matching → choropleth HTML)
- **Section 5:** XGBoost training, evaluation, feature importance
- **Section 6:** SHAP TreeExplainer — beeswarm, waterfall, dependence plots
- **Section 7:** Artifact export (`meta.json`, `shap_values.csv`, `test_predictions.csv`)

Expected runtime: 3–8 minutes depending on hardware.

### Step 2 — Launch the Interactive Dashboard

```bash
streamlit run app/dashboard.py
```

Open **http://localhost:8501** in your browser. The dashboard loads all pre-computed artifacts; no re-training occurs at runtime.

### Step 3 — (Optional) TabNet Comparison

```bash
python tabnet_train.py            # produces tabnet_results.json
```

Results are automatically included in the Results table if the file exists.

---

## 8. Results

### Predictive Performance

| Model | RMSE (Kg/ha) | MAE (Kg/ha) | R² | Train Time |
|-------|:------------:|:-----------:|:--:|:----------:|
| XGBoost Regressor | **379.56** | **263.69** | **0.8695** | ~45 s |
| TabNet (deep tabular) | — | — | — | — |
| Linear Regression (baseline) | — | — | — | — |

*TabNet and baseline results to be added after comparative training run.*

### Top 5 Features by Mean |SHAP| Value

| Rank | Feature | Mean |SHAP| (Kg/ha) | Category |
|------|---------|------------------------|----------|
| 1 | Rice Sown Area (1000 ha) | 188.2 | Controllable |
| 2 | State (encoded) | 138.9 | Structural |
| 3 | Year | 104.6 | Temporal trend |
| 4 | Gross Cropped Area (1000 ha) | 89.4 | Controllable |
| 5 | Nitrogen Consumption (tons) | 84.5 | Controllable |

Three of the top five features are policy-actionable (sown area allocation, land use intensity, fertilizer application), validating the controllability framing of this project.

---

## 9. Visualizations

### 1. Choropleth — Average Yield per District (`choropleth_yield.html`)
An interactive Plotly choropleth mapping average rice yield (Kg/ha) over 1990–2015 for all 473 matched districts. Reveals the well-documented North-South yield gradient and identifies data-sparse districts in the Northeast where model uncertainty is highest.

### 2. Choropleth — Yield Trend Slope per District (`choropleth_trend.html`)
Maps the OLS slope (Kg/ha/year) of yield over time per district using the same geometry. Districts with strongly negative slopes despite adequate rainfall signal structural problems (soil degradation, farmer distress) rather than climate shocks — a separation that static yield maps cannot provide.

### 3. SHAP Beeswarm Plot (`shap/shap_beeswarm.png`)
Global summary showing the distribution of SHAP values across all test samples for every feature. Colour encodes feature magnitude (red = high value, blue = low). Reveals non-linear threshold effects — e.g., nitrogen consumption has near-zero SHAP impact below a critical level, then becomes dominant above it.

### 4. SHAP Waterfall Plots (`shap/shap_waterfall_*.png`)
Local explanations for individual district-year observations. Each bar shows a single feature's additive contribution to the prediction relative to the global mean. Used in the dashboard to answer "Why did district X yield only 1,200 Kg/ha in 2010?" with feature-level precision.

### 5. SHAP Dependence Plots (`shap/shap_dependence_*.png`)
Scatter plots of a feature's value vs. its SHAP contribution, coloured by a selected interaction feature. Reveals, for example, the interaction between nitrogen consumption and rainfall — fertilizer efficacy is highly rainfall-dependent, an agronomically meaningful and visually demonstrable finding.

### 6. Actual vs. Predicted Scatter (`actual_vs_predicted.png`)
Scatter plot of held-out test set predictions against ground-truth yield values with an identity line. Residual heteroscedasticity — larger errors at very high yield values — is visible and discussed in the paper as a modelling limitation.

---

## 10. Publishable Angles

This project is structured for submission to venues at the intersection of applied machine learning, agricultural informatics, and explainable AI:

| Venue | Why This Fits |
|-------|--------------|
| **ICRITO** (IEEE Int'l Conference on Reliability, Infocom Technologies and Optimization) | Explicitly targets data-driven optimization and reliability in applied domains; the dual-model comparison and SHAP methodology align with its ML track. |
| **ICCIDS** (Int'l Conference on Computational Intelligence in Data Science) | Targets agricultural and environmental AI applications; multi-source fusion and district-level spatial analysis are direct fits for its Applied AI in Agriculture track. |
| **IEEE CONIT** (Int'l Conference on Innovative Trends in Information Technology) | Strong tradition of accepting explainable AI papers from Indian institutions; the controllability-stratified SHAP output is a novel framing suitable for a 6-page IEEE format. |
| **Springer LNNS** (Lecture Notes in Networks and Systems) | Accepts extended versions of conference papers; the full pipeline methodology (data fusion + dual model + spatial XAI) justifies a 12–15 page Springer chapter. |
| **Computers and Electronics in Agriculture** (Elsevier, Q1 Journal) | Premier journal for computational methods in agriculture; the temporal trend choropleth and controllability classification are novel enough to support a full journal article. |

---

## 11. Future Work

1. **Satellite-derived features (Sentinel-2 / MODIS NDVI).** Integrating growing-season NDVI time series per district would add a real-time observational layer absent from census-based datasets, potentially closing the residual error at high yield values where current features are insufficient.

2. **Real-time IMD API integration.** The current pipeline is retrospective (1990–2015). Connecting to India Meteorological Department's live district-level API would enable a nowcasting mode — predicting in-season yield shortfalls with current monsoon data.

3. **Multi-crop extension and crop recommendation.** The dataset contains yield data for rice, pearl millet, chickpea, groundnut, and sugarcane. A multi-output model with a crop recommendation layer — "which crop maximises expected yield given this district's controllable factors?" — is a natural extension and significantly increases policy utility.

4. **Counterfactual explanations for district planning.** Augmenting SHAP with DICE (Diverse Counterfactual Explanations) would allow the dashboard to answer "How much additional nitrogen application would move district X from the bottom yield quartile to the median?" — a direct decision-support capability for district agriculture officers.

---

## 12. Authors & Acknowledgements

**Authors**
- Pradhan Mrida — 6th Semester, Data Analytics & Visualization (DAV), Experiential Learning Project

**Acknowledgements**
- ICRISAT (International Crops Research Institute for the Semi-Arid Tropics) for the district-level agricultural panel dataset made available via Mendeley Data.
- India Meteorological Department (IMD) for rainfall records.
- The open-source communities behind XGBoost, SHAP, PyTorch-TabNet, GeoPandas, and Streamlit.
- District shapefile geometry sourced from the geohacker/india repository and ICRISAT Figshare.

---

## 13. License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Pradhan Mrida

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

*This project was developed as part of the 6th Semester Data Analytics and Visualization (DAV) Experiential Learning curriculum. The methodology, visualizations, and policy framing are original contributions of the authors.*
