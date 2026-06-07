'use client';
import { useDistrict } from '@/context/DistrictContext';

export default function Sidebar() {
  const {
    meta, districts,
    selState, setSelState,
    selDistrict, setSelDistrict,
    selCrop, setSelCrop,
    states, districtsForState,
  } = useDistrict();

  const distData = districts?.[selDistrict];
  const lastIdx  = distData?.pred_years ? distData.pred_years.length - 1 : -1;
  const actual   = lastIdx >= 0 ? distData?.actual_test?.[lastIdx] : null;
  const pred     = lastIdx >= 0 ? distData?.pred_values?.[lastIdx] : null;

  return (
    <aside className="sidebar">
      <div className="sb-inner">
        <div style={{ marginBottom: 28 }}>
          <div className="sb-overline">Yield Observatory</div>
          <div className="sb-title">RootCause</div>
          <div className="sb-subtitle">District Intelligence</div>
        </div>

        <div className="sb-divider" />

        <div className="sb-field">
          <label className="sb-label" htmlFor="state-select">State</label>
          <div className="sb-select-wrap">
            <select
              id="state-select"
              value={selState}
              onChange={e => setSelState(e.target.value)}
            >
              {states.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        </div>

        <div className="sb-field">
          <label className="sb-label" htmlFor="district-select">District</label>
          <div className="sb-select-wrap">
            <select
              id="district-select"
              value={selDistrict}
              onChange={e => setSelDistrict(e.target.value)}
            >
              {districtsForState.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
        </div>

        <div className="sb-field">
          <label className="sb-label" htmlFor="crop-select">Crop</label>
          <div className="sb-select-wrap">
            <select
              id="crop-select"
              value={selCrop}
              onChange={e => setSelCrop(e.target.value)}
            >
              {(meta?.yield_cols ?? []).map(c => (
                <option key={c} value={c}>{c.replace(' YIELD (Kg per ha)', '')}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="sb-divider" />

        <div className="sb-metrics-heading">XGBoost Model</div>
        <table className="sb-metrics-table">
          <tbody>
            <tr>
              <td className="sb-met-lbl">R²</td>
              <td className="sb-met-val accent">{meta?.model_r2 ?? '—'}</td>
            </tr>
            <tr>
              <td className="sb-met-lbl">RMSE</td>
              <td className="sb-met-val">{meta?.model_rmse ? `${meta.model_rmse} Kg/ha` : '—'}</td>
            </tr>
          </tbody>
        </table>

        <div className="sb-divider" style={{ marginTop: 16 }} />

        <div className="sb-metrics-heading">Selected District</div>
        <div className="sb-dr-row">
          <span className="sb-dr-lbl">Actual</span>
          <span className="sb-dr-val">{actual != null ? `${Math.round(actual)} Kg/ha` : '—'}</span>
        </div>
        <div className="sb-dr-row">
          <span className="sb-dr-lbl">Predicted</span>
          <span className="sb-dr-val amber">{pred != null ? `${Math.round(pred)} Kg/ha` : '—'}</span>
        </div>

        <div className="sb-divider" style={{ marginTop: 16 }} />

        <div className="sb-metrics-heading">Enriched Model</div>
        <table className="sb-metrics-table">
          <tbody>
            <tr>
              <td className="sb-met-lbl">R²</td>
              <td className="sb-met-val accent">{meta?.enriched_r2 ?? '—'}</td>
            </tr>
            <tr>
              <td className="sb-met-lbl">RMSE</td>
              <td className="sb-met-val">{meta?.enriched_rmse ? `${meta.enriched_rmse} Kg/ha` : '—'}</td>
            </tr>
            <tr>
              <td className="sb-met-lbl">MAE</td>
              <td className="sb-met-val">{meta?.enriched_mae ? `${meta.enriched_mae} Kg/ha` : '—'}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="sb-footer">
        ICRISAT · IMD · data.gov.in<br />
        560 districts · 1990–2015
      </div>
    </aside>
  );
}
