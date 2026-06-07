'use client';
import { useDistrict } from '@/context/DistrictContext';
import FigCard from '@/components/ui/FigCard';
import SHAPWaterfall from '@/components/charts/SHAPWaterfall';

export default function SHAPTab() {
  const { meta, tabnet, shapData, selDistrict } = useDistrict();

  return (
    <div className="tab-panel">
      {/* Model comparison */}
      <div className="section-head">
        <span className="section-title">Model Comparison</span>
        <div className="section-rule" />
      </div>

      <div className="model-grid">
        <div className="model-card">
          <div className="model-card-title">XGBoost · Baseline</div>
          <div className="model-stat">
            <span className="s-label">RMSE</span>
            <span className="s-val accent">{meta?.model_rmse ? `${meta.model_rmse} Kg/ha` : '—'}</span>
          </div>
          <div className="model-stat">
            <span className="s-label">MAE</span>
            <span className="s-val">{meta?.model_mae ? `${meta.model_mae} Kg/ha` : '—'}</span>
          </div>
          <div className="model-stat">
            <span className="s-label">R²</span>
            <span className="s-val">{meta?.model_r2 ?? '—'}</span>
          </div>
        </div>

        <div className="model-card">
          <div className="model-card-title">TabNet · Attention-based</div>
          <div className="model-stat">
            <span className="s-label">RMSE</span>
            <span className="s-val">{tabnet?.rmse ? `${tabnet.rmse} Kg/ha` : '—'}</span>
          </div>
          <div className="model-stat">
            <span className="s-label">MAE</span>
            <span className="s-val">{tabnet?.mae ? `${tabnet.mae} Kg/ha` : '—'}</span>
          </div>
          <div className="model-stat">
            <span className="s-label">R²</span>
            <span className="s-val">{tabnet?.r2 ?? '—'}</span>
          </div>
          <div className="model-stat">
            <span className="s-label">Train time</span>
            <span className="s-val">{tabnet?.train_time_s ? `${tabnet.train_time_s} s` : '—'}</span>
          </div>
        </div>
      </div>

      <div className="panel info">
        XGBoost outperforms TabNet on this dataset for three reasons:
        (1) ~23 samples per district is too few for TabNet&apos;s attention heads to converge meaningfully;
        (2) agricultural features have threshold-style interactions that decision-tree splits capture natively;
        (3) the high feature-to-row ratio makes deep networks prone to overfitting on small-N tabular data.
      </div>

      {/* Global SHAP figures */}
      <div className="section-head">
        <span className="section-title">Global SHAP Explainability</span>
        <div className="section-rule" />
      </div>

      <div className="fig-grid" style={{ marginBottom: 32 }}>
        {(meta?.shap_figures ?? []).map((fig, i) => (
          <FigCard key={i} fig={fig} basePath="/figures" />
        ))}
      </div>

      {/* Live waterfall */}
      <div className="section-head">
        <span className="section-title">Live SHAP Waterfall</span>
        {selDistrict && <span className="district-tag">{selDistrict}</span>}
        <div className="section-rule" />
      </div>

      {shapData && selDistrict ? (
        <SHAPWaterfall shapData={shapData} district={selDistrict} />
      ) : (
        <div className="panel">Loading SHAP data…</div>
      )}
    </div>
  );
}
