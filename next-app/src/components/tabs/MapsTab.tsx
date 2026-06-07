'use client';
import { useDistrict } from '@/context/DistrictContext';
import MetricCard from '@/components/ui/MetricCard';

export default function MapsTab() {
  const { mapsData } = useDistrict();
  const s = mapsData?.stats;

  return (
    <div className="tab-panel">
      {/* Yield map */}
      <div className="section-head">
        <span className="section-title">Average Rice Yield per District</span>
        <div className="section-rule" />
      </div>

      {s && (
        <div className="metric-row">
          <MetricCard label="Districts Mapped" value={s.count} />
          <MetricCard label="Highest Avg Yield" value={`${Math.round(s.max_yield)} Kg/ha`} sub={s.max_yield_district} />
          <MetricCard label="Lowest Avg Yield"  value={`${Math.round(s.min_yield)} Kg/ha`} sub={s.min_yield_district} />
        </div>
      )}

      <iframe className="map-frame" src="/maps/choropleth_yield.html" title="Yield choropleth" />

      {/* Trend map */}
      <div className="section-head">
        <span className="section-title">Yield Trend per District</span>
        <div className="section-rule" />
      </div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 14 }}>
        OLS slope (Kg/ha/yr) · 1990–2015 · green = improving · red = declining
      </p>

      {s && (
        <div className="metric-row">
          <MetricCard label="Improving Districts"  value={s.improving} sub="positive OLS slope" />
          <MetricCard label="Declining Districts"  value={s.declining} sub="negative OLS slope" />
          <MetricCard label="Fastest Decline" value={s.fastest_decline_district} sub={`${s.fastest_decline_slope.toFixed(1)} Kg/ha/yr`} />
        </div>
      )}

      <iframe className="map-frame" src="/maps/choropleth_trend.html" title="Trend choropleth" />
    </div>
  );
}
