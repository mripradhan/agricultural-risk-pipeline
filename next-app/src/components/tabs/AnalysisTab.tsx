'use client';
import { useDistrict } from '@/context/DistrictContext';
import HistogramChart from '@/components/charts/HistogramChart';
import TrendChart from '@/components/charts/TrendChart';
import FigCard from '@/components/ui/FigCard';

export default function AnalysisTab() {
  const { meta, districts, selDistrict, selCrop } = useDistrict();
  const d = districts?.[selDistrict];
  const target = meta?.target ?? '';

  return (
    <div className="tab-panel">
      {/* District charts */}
      <div className="section-head">
        <span className="section-title">District Deep-Dive</span>
        {selDistrict && <span className="district-tag">{selDistrict}</span>}
        <div className="section-rule" />
      </div>

      <div className="chart-grid">
        <div className="chart-wrap">
          {d && (
            <HistogramChart
              values={(d[target] as (number | null)[]) ?? []}
              district={selDistrict}
            />
          )}
        </div>
        <div className="chart-wrap">
          {d && (
            <TrendChart
              distData={d}
              crop={selCrop || target}
              district={selDistrict}
            />
          )}
        </div>
      </div>

      {/* Pre-computed EDA figures */}
      <div className="section-head" style={{ marginTop: 32 }}>
        <span className="section-title">Pre-computed EDA Figures</span>
        <div className="section-rule" />
      </div>

      <div className="fig-grid">
        {(meta?.eda_figures ?? []).map((fig, i) => (
          <FigCard key={i} fig={fig} basePath="/figures" />
        ))}
      </div>
    </div>
  );
}
