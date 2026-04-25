
# ICRISAT District-Level Crop Yield Analysis — Summary

## Dataset
| Metric | Value |
|---|---|
| Raw shape | 12,803 rows × 107 columns |
| After cleaning | 12,803 rows × 108 columns |
| Districts | 560 |
| States | 20 |
| Years | 1990–2015 |
| Target variable | `RICE YIELD (Kg per ha)` |
| Feature count | 90 |
| Crop yield columns | ['RICE YIELD (Kg per ha)', 'PEARL MILLET YIELD (Kg per ha)', 'CHICKPEA YIELD (Kg per ha)', 'GROUNDNUT YIELD (Kg per ha)', 'SUGARCANE YIELD (Kg per ha)'] |

## Model Performance (XGBoost, 80/20 split, stratified by state)
| Metric | Value |
|---|---|
| RMSE | 379.56 Kg/ha |
| MAE | 263.69 Kg/ha |
| R² | 0.8695 |

## Top 5 Features by Mean |SHAP|
|                              |        0 |
|:-----------------------------|---------:|
| RICE AREA (1000 ha)          | 188.221  |
| State Name_enc               | 138.881  |
| Year                         | 104.628  |
| GROSS CROPPED AREA (1000 ha) |  89.4101 |
| NITROGEN CONSUMPTION (tons)  |  84.4663 |

## Outputs
| Output | Path |
|---|---|
| Cleaned data | `data/processed/master_clean.csv` |
| Long-format data | `data/processed/long_format.csv` |
| Label mappings | `data/processed/label_mappings.json` |
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
