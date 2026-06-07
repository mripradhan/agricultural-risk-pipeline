'use client';
import Plot from './PlotWrapper';
import { CHART, LAYOUT_BASE, CONFIG, SERIF_FONT, SANS_FONT } from '@/lib/constants';
import type { DistrictData } from '@/lib/types';

interface Props {
  distData: DistrictData;
  crop:     string;
  district: string;
}

export default function TrendChart({ distData, crop, district }: Props) {
  const yVals = (distData[crop] as (number | null)[] | undefined) ?? [];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const traces: any[] = [
    {
      x: distData.years, y: yVals,
      type: 'scatter', mode: 'lines+markers',
      name: crop.replace(' YIELD (Kg per ha)', ''),
      line:   { color: '#111a11', width: 2 },
      marker: { size: 5, color: '#111a11' },
      fill: 'tozeroy', fillcolor: 'rgba(17,26,17,0.06)',
    },
  ];

  if (distData.pred_years?.length) {
    traces.push({
      x: distData.pred_years, y: distData.pred_values ?? [],
      type: 'scatter', mode: 'markers', name: 'Predicted',
      marker: { color: CHART.accent, size: 10, symbol: 'diamond', line: { color: '#fff', width: 1.5 } },
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        ...LAYOUT_BASE,
        title: {
          text: `Yield trend · ${district}`,
          font: { family: SERIF_FONT, size: 13, color: CHART.muted },
          x: 0.04,
        },
        xaxis: {
          title: { text: 'Year', font: { family: SERIF_FONT, size: 12 } },
          tickfont: { family: SANS_FONT, size: 11 },
          gridcolor: '#eae5dc', gridwidth: 1,
          linecolor: '#d6d0c4',
        },
        yaxis: {
          title: { text: 'Yield (Kg/ha)', font: { family: SERIF_FONT, size: 12 } },
          tickfont: { family: SANS_FONT, size: 11 },
          gridcolor: '#eae5dc', gridwidth: 1,
          linecolor: '#d6d0c4',
        },
        showlegend: true,
        legend: {
          x: 0.04, y: 0.97,
          font: { family: SERIF_FONT, size: 11, color: CHART.muted },
          bgcolor: 'rgba(255,255,255,0.8)',
          bordercolor: '#eae5dc', borderwidth: 1,
        },
        height: 420,
      }}
      config={CONFIG}
      style={{ width: '100%' }}
    />
  );
}
