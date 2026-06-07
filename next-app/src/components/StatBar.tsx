'use client';
import { useDistrict } from '@/context/DistrictContext';

export default function StatBar() {
  const { meta, districts, selDistrict, shapData } = useDistrict();
  const d = districts?.[selDistrict];
  const lastIdx = d?.pred_years ? d.pred_years.length - 1 : -1;
  const actual  = lastIdx >= 0 ? d?.actual_test?.[lastIdx]  : null;
  const pred    = lastIdx >= 0 ? d?.pred_values?.[lastIdx]  : null;
  const yr      = lastIdx >= 0 ? d?.pred_years?.[lastIdx]   : null;
  const delta   = actual != null && pred != null ? pred - actual : null;

  return (
    <div className="stat-bar">
      <div className="stat-card">
        <div className="stat-val">{actual != null ? `${Math.round(actual)} Kg/ha` : '—'}</div>
        <div className="stat-lbl">Actual Yield</div>
        <div className="stat-sub">{yr ? `latest: ${yr}` : 'latest test year'}</div>
      </div>
      <div className="stat-card">
        <div className="stat-val">{pred != null ? `${Math.round(pred)} Kg/ha` : '—'}</div>
        <div className="stat-lbl">Predicted</div>
        <div className="stat-sub">
          {delta != null ? `${delta >= 0 ? '+' : ''}${Math.round(delta)} Kg/ha vs actual` : 'model estimate'}
        </div>
      </div>
      <div className="stat-card">
        <div className="stat-val">{d?.years?.length ?? '—'}</div>
        <div className="stat-lbl">Years of Data</div>
        <div className="stat-sub">1990 – 2015</div>
      </div>
      <div className="stat-card">
        <div className="stat-val">{districts ? Object.keys(districts).length : '—'}</div>
        <div className="stat-lbl">Districts</div>
        <div className="stat-sub">in dataset</div>
      </div>
      <div className="stat-card">
        <div className="stat-val">{meta?.model_r2 ?? '—'}</div>
        <div className="stat-lbl">Model R²</div>
        <div className="stat-sub">XGBoost baseline</div>
      </div>
    </div>
  );
}
