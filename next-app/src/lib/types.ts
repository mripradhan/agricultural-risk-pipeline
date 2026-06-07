export interface Figure {
  file: string;
  label: string;
  caption: string;
}

export interface Meta {
  target: string;
  district_col: string;
  state_col: string;
  year_col: string;
  model_rmse: number;
  model_mae: number;
  model_r2: number;
  enriched_rmse?: number;
  enriched_mae?: number;
  enriched_r2?: number;
  yield_cols: string[];
  top2_shap: string[];
  eda_figures: Figure[];
  shap_figures: Figure[];
}

export interface DistrictData {
  state: string;
  years: number[];
  pred_years?: number[];
  pred_values?: (number | null)[];
  actual_test?: (number | null)[];
  [yieldCol: string]: unknown;
}

export interface Districts {
  [district: string]: DistrictData;
}

export interface MapDistrictEntry {
  name: string;
  avg_yield: number | null;
  slope: number | null;
}

export interface MapStats {
  count: number;
  max_yield_district: string;
  max_yield: number;
  min_yield_district: string;
  min_yield: number;
  improving: number;
  declining: number;
  fastest_decline_district: string;
  fastest_decline_slope: number;
}

export interface MapsData {
  districts: MapDistrictEntry[];
  stats: MapStats;
}

export interface TabNetData {
  rmse: number;
  mae: number;
  r2: number;
  train_time_s: number;
}

export interface SHAPDistrictEntry {
  values: number[];
  year: number;
  actual: number | null;
  predicted: number | null;
}

export interface SHAPData {
  features: string[];
  categories: string[];
  base: number;
  districts: { [district: string]: SHAPDistrictEntry };
}
