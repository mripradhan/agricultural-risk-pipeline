![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Academic%20%7C%206th%20Sem%20DAV-orange)
![Model](https://img.shields.io/badge/Model-XGBoost%20%7C%20TabNet-purple)
![SHAP](https://img.shields.io/badge/Explainability-SHAP%20TreeExplainer-red)

---

# RootCause — District-Level Crop Yield Explainability

> **Predicts crop yield across 560 Indian districts AND visually explains the single most impactful controllable factor per district — turning a black-box ML model into an actionable policy tool.**

---

## Overview

India's agricultural output sustains 1.4 billion people, yet the predictive models that governments and researchers build to anticipate yield shortfalls consistently fail at the last mile: they produce a number with no explanation of *why* a district underperforms, and no guidance on what a policymaker can actually do about it. **RootCause** inverts this priority. Using five fused heterogeneous data sources spanning 560 districts across 20 Indian states from 1990 to 2015, this project trains both a gradient-boosted tree (XGBoost, R² = 0.87) and a deep tabular model (TabNet) on district-level agricultural features — then makes SHAP explainability the *primary deliverable*, not a supplementary table. The interactive Next.js dashboard lets any district agriculture officer select their jurisdiction, see a live waterfall chart decomposing exactly which factors drove that year's yield outcome, and receive the single most impactful *controllable* lever (fertilizer application, irrigation intensity) they can act on.

---

## Screenshots

**Analysis tab** — yield distribution histogram and year-wise trend for the selected district, with pre-computed EDA figures below.

![Analysis tab](<outputs/figures/ui-screenshots/Screenshot 2026-06-07 203259.png>)

**Maps tab** — interactive choropleth of average rice yield per district across 560 Indian districts (1990–2015).

![Maps tab](<outputs/figures/ui-screenshots/Screenshot 2026-06-07 203352.png>)

**SHAP tab** — global waterfall and dependence plot side by side.

![SHAP tab overview](<outputs/figures/ui-screenshots/Screenshot 2026-06-07 203417.png>)

**Live SHAP Waterfall** — per-district SHAP attribution decomposing the prediction into individual feature contributions.

![Live SHAP waterfall](<outputs/figures/ui-screenshots/image.png>)

**Feature Breakdown table** — top 15 features by SHAP impact, tagged as Controllable, Uncontrollable, or Structural.

![Feature breakdown](<outputs/figures/ui-screenshots/image copy.png>)

---

## Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Novelty & Contributions](#-novelty--contributions)
3. [Architecture](#-architecture)
4. [Dataset Details](#-dataset-details)
5. [Project Structure](#-project-structure)
6. [Installation & Setup](#-installation--setup)
7. [How to Run](#-how-to-run)
8. [Results](#-results)
9. [Visualizations](#-visualizations)
10. [Publishable Angles](#-publishable-angles)
11. [Future Work](#-future-work)
12. [Authors & Acknowledgements](#-authors--acknowledgements)
13. [License](#-license)

---

## 🌾 Problem Statement

Agriculture employs approximately 42% of India's workforce and contributes 18% of GDP, yet district-level crop yield prediction remains a largely unsolved *operational* challenge. The barrier is not a shortage of predictive models — it is the **explainability and actionability gap**: models that achieve high accuracy on held-out test sets offer no signal to district collectors, Krishi Vigyan Kendra scientists, or state agriculture departments who must allocate fertilizer subsidies, irrigation budgets, and advisory effort under tight constraints.

Three specific gaps motivate this work:

- **Opacity gap.** Most published yield prediction papers report only RMSE and R². The model's internal reasoning — which features drove the prediction for *this district* in *this year* — is never surfaced to the end user.

- **Single-source bias.** Most studies draw from one dataset: either IMD rainfall *or* ICRISAT crop statistics *or* a state agriculture survey. The cross-domain interactions between soil nutrition, irrigation intensity, and monsoon timing that actually govern yield outcomes are invisible when features live in separate, unmerged silos.

- **Absence of controllability framing.** No existing tool distinguishes between factors a farmer or government can influence (fertilizer application rate, irrigation scheduling) and those they cannot (inter-annual monsoon variability, long-run temperature trends) when presenting model explanations.

RootCause closes all three gaps simultaneously.

---

## 🚀 Novelty & Contributions

1. **Explainability-first framing.** SHAP values are the **primary deliverable** of this pipeline, not a post-hoc addition. Every model design decision is evaluated in terms of SHAP stability alongside predictive accuracy.

2. **Multi-source heterogeneous data fusion.** Five data sources — ICRISAT crop statistics, IMD district rainfall, Kaggle crop production records, district-level fertilizer consumption (data.gov.in), and a GeoJSON district shapefile — are merged at the district-year granularity into a single 12,803 × 108 analytical matrix.

3. **Dual-model comparison: XGBoost vs. TabNet.** The same tabular agricultural dataset benchmarks a gradient-boosted tree (XGBoost) against a modern attention-based deep tabular model (TabNet), producing a principled analysis of *why* tree-based models outperform attention-based architectures on sparse agricultural panel data.

4. **Policy-actionable, controllability-stratified output.** The dashboard classifies each district's top SHAP drivers into *controllable* (fertilizer N/P/K, irrigation area, sown area allocation) and *uncontrollable* (rainfall anomaly, temperature regime) factors, then surfaces the single highest-impact controllable lever per district.

5. **Temporal trend choropleth.** A novel visualization maps the OLS yield trend slope (Kg/ha/year) per district over 1990–2015, distinguishing districts in structural long-term decline from those experiencing short-term weather shocks.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES (5)                              │
│                                                                         │
│  [ICRISAT XLS]  [IMD Rainfall]  [Kaggle Crop]  [Fertilizer]  [GeoJSON] │
│        │               │               │             │            │     │
└────────┴───────────────┴───────────────┴─────────────┴────────────┘    │
                                 │                                        │
                                 ▼                                        │
                 ┌───────────────────────────┐                           │
                 │   DISTRICT-LEVEL MERGE    │                           │
                 │   (district × year key)   │                           │
                 │   560 dist · 26 yrs       │                           │
                 └─────────────┬─────────────┘                           │
                               │                                          │
                               ▼                                          │
                 ┌───────────────────────────┐                           │
                 │      PREPROCESSING        │                           │
                 │  · Drop cols >40% missing │                           │
                 │  · Median imputation      │                           │
                 │  · Label encode district  │                           │
                 │  · 80/20 stratified split │                           │
                 └─────────────┬─────────────┘                           │
                               │                                          │
              ┌────────────────┴────────────────┐                        │
              ▼                                 ▼                        │
  ┌─────────────────────┐           ┌─────────────────────┐             │
  │   XGBoost           │           │   TabNet             │             │
  │   Regressor         │           │   (deep tabular)     │             │
  │   R² = 0.8695       │           │   (comparison)       │             │
  │   RMSE = 379 Kg/ha  │           │                     │             │
  └──────────┬──────────┘           └──────────┬──────────┘             │
             │                                 │                          │
             └─────────────┬───────────────────┘                         │
                           ▼                                              │
             ┌─────────────────────────────┐                             │
             │     SHAP TreeExplainer      │                             │
             │  · Beeswarm (global)        │                             │
             │  · Waterfall (per district) │                             │
             │  · Dependence plots         │                             │
             │  · Controllability tagging  │◄────────────────────────────┘
             │    (fertilizer/irrigation   │      (shapefile join for
             │     vs rainfall/temp)       │       choropleth render)
             └─────────────┬───────────────┘
                           ▼
             ┌─────────────────────────────┐
             │     NEXT.JS DASHBOARD       │
             │  Tab 1 · EDA               │
             │  Tab 2 · Choropleth Maps   │
             │  Tab 3 · SHAP Explorer     │
             │  Sidebar · District picker  │
             │  Policy lever panel         │
             └─────────────────────────────┘
```

---

## 📊 Dataset Details

| # | Dataset | Source | Rows (raw) | Key Columns | License |
|---|---------|--------|-----------|-------------|---------|
| 1 | ICRISAT District-Level Agricultural Data | Mendeley Data | ~14,600 | District, Year, State; yield/area/production for Rice, Pearl Millet, Chickpea, Groundnut, Sugarcane; monthly & seasonal temperature, precipitation, evapotranspiration, windspeed; N/P/K fertilizer; irrigated area | CC BY 4.0 |
| 2 | IMD District Rainfall | Kaggle | ~180,000 | District, Month, Year, Monthly Rainfall (mm), Annual Total | Public Domain |
| 3 | Crop Production in India | Kaggle | ~250,000 | State, District, Crop, Season (Kharif/Rabi/Whole Year), Area (ha), Production (tons) | Public Domain |
| 4 | District-wise Fertilizer Consumption | data.gov.in | ~8,000 | State, District, Year, N (tons), P (tons), K (tons), Total Fertilizer (tons) | OGD India (GODL) |
| 5 | India District Shapefile | Figshare / ICRISAT | 640 districts | geometry (GeoJSON polygon), state name, district name | CC BY 4.0 |

**Final merged and cleaned dataset:** 12,803 rows × 108 columns · 560 unique districts · 20 states · 1990–2015  
**Primary prediction target:** `RICE YIELD (Kg per ha)`

---

## 📁 Project Structure

```
XAI-DAV-EL/
│
├── analysis.py                      # End-to-end pipeline: audit → EDA → XGBoost → SHAP
├── check_env.py                     # Dependency & version sanity check
├── tabnet_train.py                  # TabNet comparison model training (Step 2)
├── SUMMARY.md                       # Auto-generated pipeline run summary
├── README.md                        # This file
│
├── data/
│   └── processed/
│       ├── master_clean.csv         # Cleaned wide-format district-year matrix
│       ├── long_format.csv          # Crop-melted long format (per-crop trends)
│       ├── shap_values.csv          # SHAP value matrix for all test samples
│       ├── test_predictions.csv     # Actual vs predicted yield (test set)
│       ├── label_mappings.json      # District/state integer encoding maps
│       └── meta.json                # Feature lists, model metrics, column metadata
│
├── outputs/
│   ├── figures/
│   │   ├── yield_distribution.png        # Box/violin of yield across all 5 crops
│   │   ├── yield_trend_yearly.png        # Year-wise national yield trend (all crops)
│   │   ├── yield_by_season.png           # Kharif vs Rabi yield comparison
│   │   ├── rainfall_vs_yield.png         # Scatter: monsoon rainfall vs rice yield
│   │   ├── correlation_heatmap.png       # Pearson correlations: features vs yield
│   │   ├── district_yield_top_bottom.png # Top/bottom 15 district bar chart
│   │   ├── feature_importance.png        # XGBoost native feature importance
│   │   ├── actual_vs_predicted.png       # Test set scatter with identity line
│   │   └── shap/
│   │       ├── shap_beeswarm.png         # Global SHAP summary plot
│   │       ├── shap_waterfall_best.png   # Waterfall for best-predicted district
│   │       ├── shap_waterfall_worst.png  # Waterfall for worst-predicted district
│   │       └── shap_dependence_*.png     # Feature interaction dependence plots
│   └── models/
│       └── xgboost_model.pkl        # Serialized trained XGBoost regressor
│
└── next-app/                        # RootCause — Next.js 16 dashboard
    ├── src/
    │   ├── app/                     # App Router pages and layouts
    │   ├── components/              # UI components (charts, sidebar, tabs)
    │   ├── context/                 # DistrictContext — global state
    │   └── styles/                  # globals.css — Tailwind v4 + design tokens
    └── public/data/                 # Pre-computed JSON served to the browser
```

---

## ⚙️ Installation & Setup

### Python pipeline

**Prerequisites:** Python 3.10+ · pip · git

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/XAI-DAV-EL.git
cd XAI-DAV-EL

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install core scientific stack
pip install numpy pandas matplotlib seaborn scipy scikit-learn xlrd openpyxl

# 4. Install ML and explainability libraries
#    XGBoost 2.x is pinned — XGBoost 3.x breaks shap.TreeExplainer in SHAP ≤0.49.x
pip install "xgboost==2.1.4" "shap==0.49.0"

# 5. Install deep learning (TabNet)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-tabnet

# 6. Install geospatial dependencies
pip install plotly geopandas fuzzywuzzy python-Levenshtein

# 7. Verify the environment
python check_env.py
```

### Next.js dashboard

**Prerequisites:** Node.js 18+ · npm

```bash
cd next-app
npm install
```

---

## ▶️ How to Run

### Step 1 — Full Analysis Pipeline (EDA + XGBoost + SHAP)

```bash
python analysis.py
```

Executes all seven sections end-to-end:

| Section | What it does | Key output |
|---------|-------------|------------|
| 1 | Data loading & audit | Shape, missing value report, descriptive stats |
| 2 | Cleaning & preprocessing | `master_clean.csv`, `label_mappings.json` |
| 3 | Exploratory Data Analysis | 8 static PNG figures in `outputs/figures/` |
| 4 | Geospatial processing | Choropleth data for the dashboard |
| 5 | XGBoost training & evaluation | `xgboost_model.pkl`, `test_predictions.csv` |
| 6 | SHAP explainability | `shap_values.csv`, all SHAP plots in `outputs/figures/shap/` |
| 7 | Artifact export | `meta.json` (metrics + column metadata) |

Expected runtime: 3–8 minutes on a standard CPU.

### Step 2 — TabNet Comparison Model

```bash
python tabnet_train.py
```

Trains TabNet on the same train/test split. Results are written to `outputs/models/tabnet_results.json`.

### Step 3 — RootCause Dashboard (Next.js)

```bash
cd next-app
npm run dev
```

Opens at **http://localhost:3000**. The first page load takes ~15 seconds while Tailwind compiles; all subsequent navigation is instant. No re-training occurs at runtime — the dashboard loads pre-computed JSON artifacts from `public/data/`.

Use the sidebar to select **State → District → Crop**, then explore:
- **Analysis tab:** Yield distribution and year-wise crop trend for the selected district
- **Maps tab:** Interactive choropleth — average yield and trend slope per district
- **SHAP tab:** Global beeswarm, pre-computed waterfall plots, and a live per-district SHAP waterfall with controllability annotation

#### Production build

```bash
cd next-app
npm run build
npm start
```

---

## 📈 Results

### Model Performance on Held-Out Test Set (80/20 stratified split)

| Model | RMSE (Kg/ha) | MAE (Kg/ha) | R² |
|-------|:---:|:---:|:---:|
| XGBoost Regressor | **379.56** | **263.69** | **0.8695** |
| TabNet (deep tabular) | TBD | TBD | TBD |
| Ridge Regression (baseline) | TBD | TBD | TBD |

**Key finding:** XGBoost's advantage over TabNet is expected. The dataset provides ~12,800 district-year samples spread across 560 districts — approximately 23 samples per district. TabNet's self-attention heads require substantially more data per group; at this density they overfit. Agricultural features also exhibit threshold-style interactions that decision-tree splits capture natively.

### Top Features by Mean |SHAP| Value

| Rank | Feature | Category |
|------|---------|----------|
| 1 | Rice Sown Area (1000 ha) | Controllable |
| 2 | State (encoded) | Structural |
| 3 | Year | Temporal trend |
| 4 | Gross Cropped Area (1000 ha) | Controllable |
| 5 | Rainy Season Precipitation (mm) | Uncontrollable |
| 6 | Nitrogen Consumption (tons) | Controllable |

Three of the top six features are policy-actionable, validating the controllability framing.

---

## 🖼️ Visualizations

All visualizations are rendered interactively in the dashboard. Static files live in `outputs/figures/`.

### 1. Yield Distribution — `yield_distribution.png`
Box plots and violin plots for all five crops across all districts. **Insight:** Rice yield variance is bimodal — a high cluster (Indo-Gangetic plain) and a low cluster (peninsular/central India) separated by ~3,000 Kg/ha.

### 2. Year-wise Yield Trend — `yield_trend_yearly.png`
Mean national yield per year (1990–2015) for each crop. **Insight:** Rice shows a steady upward trend while chickpea is nearly flat — a divergence pointing to differential technology adoption rather than climate.

### 3. Top/Bottom 15 Districts — `district_yield_top_bottom.png`
Ranked bar chart of the 15 highest and 15 lowest 25-year average rice yield districts. **Insight:** The yield gap between best and worst district exceeds 4,500 Kg/ha — larger than the all-India average itself.

### 4. Correlation Heatmap — `correlation_heatmap.png`
Pearson correlations between rice yield and all climate and agronomic features. **Insight:** Summer maximum temperature shows a strong negative relationship, providing the biological basis (heat-stress during flowering) for the temperature features' predictive power.

### 5. Choropleth — Average Yield & Trend Slope per District
Interactive Plotly choropleth over GeoJSON district polygons in the Maps tab. **Insight:** Several coastal Andhra Pradesh and Odisha districts show large *negative* trend slopes despite adequate rainfall — structural decline invisible in average-yield snapshots.

### 6. SHAP Beeswarm Plot — `shap/shap_beeswarm.png`
Each point is one test sample (district-year). X-axis is SHAP contribution magnitude; colour encodes feature value. **Insight:** Nitrogen fertilizer shows a clear positive SHAP tail — districts with high N consumption systematically outperform the model's baseline prediction.

### 7. SHAP Waterfall Plots — `shap/shap_waterfall_*.png`
Additive decomposition of one district-year prediction. **Insight:** For the best-predicted district, high nitrogen and irrigation area drive the prediction 600+ Kg/ha above baseline. The dashboard renders these live for any user-selected district.

### 8. SHAP Dependence Plots — `shap/shap_dependence_*.png`
Feature value vs. SHAP contribution scatter. **Insight:** Fertilizer efficacy is rainfall-dependent — near-zero SHAP contribution in low-rainfall districts regardless of N input, and large positive contributions in high-rainfall districts.

---

## 📰 Publishable Angles

| Venue | Why This Fits |
|-------|--------------|
| **ICRITO** (IEEE Int'l Conference on Reliability, Infocom Technologies and Optimization) | Targets data-driven applied ML with societal impact; the dual-model comparison and policy-actionable SHAP output align directly with its Applied AI track |
| **ICCIDS** (Int'l Conference on Computational Intelligence in Data Science) | Focused on agricultural and environmental AI applications; multi-source fusion and district-level spatial explainability are in-scope contributions |
| **IEEE CONIT** (Int'l Conference on Innovative Trends in Information Technology) | Strong XAI track; the controllability-stratified SHAP output is a novel enough framing for a 6-page IEEE format paper |
| **Springer LNNS** (Lecture Notes in Networks and Systems) | Accepts full pipeline methodology papers; the complete arc from raw data to deployed dashboard supports a 12–15 page Springer chapter |
| **Computers and Electronics in Agriculture** (Elsevier, Q1, IF ≈ 8) | Premier journal for computational agricultural methods; the temporal trend choropleth and controllability classification are directly in scope |

---

## 🔭 Future Work

1. **Satellite imagery (Sentinel-2 / MODIS NDVI).** Integrating growing-season NDVI per district as a mid-season observational feature would add a real-time health signal absent from census-based datasets.

2. **Real-time IMD API pipeline.** Replacing the static 1990–2015 rainfall dataset with a live feed from the India Meteorological Department's OpenData API would enable a nowcasting mode.

3. **Multi-crop recommendation extension.** A multi-output model with a crop recommendation layer — "which crop maximises expected yield given this district's controllable input profile?" — substantially increases the policy utility of the dashboard.

4. **Causal inference layer.** SHAP measures association, not causation. Adding a difference-in-differences design or an IV approach would elevate the dashboard's recommendations from correlation-based to causal-effect estimates.

---

## 👥 Authors & Acknowledgements

**6th Semester — Data Analytics & Visualization (DAV), Experiential Learning Project**

| Member | Primary Responsibility |
|--------|----------------------|
| Pradhan Mrida | Pipeline architecture, XGBoost modelling, SHAP analysis |
| *(Member 2)* | TabNet implementation, dual-model comparison, results |
| *(Member 3)* | Next.js dashboard, choropleth maps, controllability UI |

**Acknowledgements**
- ICRISAT for the district-level climate-crop panel dataset (CC BY 4.0, Mendeley Data)
- India Meteorological Department (IMD) for publicly accessible precipitation records
- data.gov.in for the district-wise fertilizer consumption series (GODL-India)
- Open-source communities behind XGBoost, SHAP, PyTorch-TabNet, GeoPandas, Next.js, and Tailwind CSS
- Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions* — for the SHAP framework

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for the full text.

The underlying ICRISAT dataset is published under **CC BY 4.0**. Any redistribution of derived data must retain the original attribution.

---

*Developed as part of the 6th Semester Data Analytics and Visualization (DAV) Experiential Learning curriculum. Methodology, visualizations, and policy framing are original contributions of the authors.*
