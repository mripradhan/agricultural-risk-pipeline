# ICRISAT District-Level Crop Yield Analysis — Summary

## Dataset
| Metric | Value |
|---|---|
| Raw shape | 12,803 rows × 107 columns |
| After cleaning | 12,751 rows × 108 columns |
| Districts | 560 |
| States | 20 |
| Years | 1990–2015 |
| Target variable | `RICE YIELD (Kg per ha)` |
| Feature count | 90 |
| Crop yield columns | ['RICE YIELD (Kg per ha)', 'PEARL MILLET YIELD (Kg per ha)', 'CHICKPEA YIELD (Kg per ha)', 'GROUNDNUT YIELD (Kg per ha)', 'SUGARCANE YIELD (Kg per ha)'] |

## Model Performance (XGBoost, 80/20 split, stratified by state)
| Metric | Value |
|---|---|
| RMSE | 376.85 Kg/ha |
| MAE  | 263.24 Kg/ha |
| R²   | 0.8709 |

## Top 5 Features by Mean |SHAP|
|                              |        0 |
|:-----------------------------|---------:|
| RICE AREA (1000 ha)          | 179.603  |
| State Name_enc               | 139.888  |
| NITROGEN CONSUMPTION (tons)  | 104.959  |
| Year                         | 102.184  |
| GROSS CROPPED AREA (1000 ha) |  93.1054 |

## Outputs
| Output | Path |
|---|---|
| Cleaned data | `data/processed/master_clean.csv` |
| Long-format data | `data/processed/long_format.csv` |
| Label mappings | `data/processed/label_mappings.json` |
| Split indices | `data/processed/split_indices.json` |
| SHAP values | `data/processed/shap_values.csv` |
| Controllability | `data/processed/controllability.json` |
| Map metadata | `data/processed/map_meta.json` |
| XGBoost model | `outputs/models/xgboost_model.pkl` |
| EDA figures | `outputs/figures/*.png` |
| Choropleth maps | `outputs/figures/choropleth_*.html` |
| SHAP figures | `outputs/figures/shap/*.png` |
| Dashboard | `app/dashboard.py` |

## Run the Dashboard
```bash
streamlit run app/dashboard.py
```
Then open **http://localhost:8501** in your browser.