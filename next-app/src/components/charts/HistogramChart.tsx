'use client';
import Plot from './PlotWrapper';
import { CHART, LAYOUT_BASE, CONFIG, SERIF_FONT, SANS_FONT } from '@/lib/constants';

interface Props {
  values: (number | null)[];
  district: string;
}

export default function HistogramChart({ values, district }: Props) {
  const clean = values.filter((v): v is number => v != null);
  if (!clean.length) return (
    <div style={{ height: 420, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontFamily: 'var(--serif)', fontStyle: 'italic', fontSize: '0.85rem' }}>
      No data for {district}
    </div>
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [{
    x: clean, type: 'histogram', nbinsx: 12,
    marker: { color: '#111a11', opacity: 0.78 },
    name: 'Rice Yield',
  }];

  return (
    <Plot
      data={data}
      layout={{
        ...LAYOUT_BASE,
        title: {
          text: `Yield distribution · ${district}`,
          font: { family: SERIF_FONT, size: 13, color: CHART.muted },
          x: 0.04,
        },
        xaxis: {
          title: { text: 'Rice Yield (Kg/ha)', font: { family: SERIF_FONT, size: 12 } },
          tickfont: { family: SANS_FONT, size: 11 },
          gridcolor: '#eae5dc', gridwidth: 1,
          linecolor: '#d6d0c4',
        },
        yaxis: {
          title: { text: 'Years', font: { family: SERIF_FONT, size: 12 } },
          tickfont: { family: SANS_FONT, size: 11 },
          gridcolor: '#eae5dc', gridwidth: 1,
          linecolor: '#d6d0c4',
        },
        height: 420,
        bargap: 0.06,
      }}
      config={CONFIG}
      style={{ width: '100%' }}
    />
  );
}
