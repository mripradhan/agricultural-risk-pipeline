'use client';
import Plot from './PlotWrapper';
import { CHART, LAYOUT_BASE, CONFIG, SERIF_FONT, SANS_FONT } from '@/lib/constants';
import type { SHAPData } from '@/lib/types';

interface Props {
  shapData:  SHAPData;
  district:  string;
}

function getBadgeClass(cat: string) {
  if (cat === 'Controllable')   return 'badge badge-ctrl';
  if (cat === 'Uncontrollable') return 'badge badge-unctrl';
  return 'badge badge-struct';
}

export default function SHAPWaterfall({ shapData, district }: Props) {
  const entry = shapData.districts[district];

  if (!entry) {
    return (
      <div className="panel" style={{ marginTop: 0 }}>
        No SHAP data for <strong>{district}</strong> — district may be in the training set.
      </div>
    );
  }

  const { features, categories, base } = shapData;
  const { values, year, actual, predicted } = entry;
  const sign = (v: number) => (v >= 0 ? '+' : '');

  const pairs = features
    .map((f, i) => ({ feature: f, value: values[i], cat: categories[i], abs: Math.abs(values[i]) }))
    .sort((a, b) => b.abs - a.abs)
    .slice(0, 15);

  const ctrlTotal   = pairs.filter(p => p.cat === 'Controllable').reduce((s, p) => s + p.value, 0);
  const unctrlTotal = pairs.filter(p => p.cat === 'Uncontrollable').reduce((s, p) => s + p.value, 0);

  const colors = pairs.map(p => p.value >= 0 ? CHART.pos : CHART.neg);
  const labels = pairs.map(p => p.feature.length > 44 ? p.feature.slice(0, 41) + '…' : p.feature);

  const ctrlTop = [...pairs]
    .filter(p => p.cat === 'Controllable')
    .sort((a, b) => b.abs - a.abs)[0];

  return (
    <>
      {/* Summary */}
      <div className="panel" style={{ marginBottom: 14 }}>
        <strong>{district} · {year}</strong> &nbsp;—&nbsp;
        Controllable: <strong>{sign(ctrlTotal)}{Math.round(ctrlTotal)} Kg/ha</strong> &nbsp;·&nbsp;
        Uncontrollable: <strong>{sign(unctrlTotal)}{Math.round(unctrlTotal)} Kg/ha</strong> &nbsp;·&nbsp;
        baseline {Math.round(base)} Kg/ha &nbsp;·&nbsp;
        actual <strong>{actual != null ? Math.round(actual) : '—'} Kg/ha</strong> &nbsp;·&nbsp;
        predicted <strong>{predicted != null ? Math.round(predicted) : '—'} Kg/ha</strong>
      </div>

      {/* Waterfall chart */}
      <div className="chart-wrap" style={{ marginBottom: 20 }}>
        <Plot
          data={[{
            type: 'bar', orientation: 'h',
            x: pairs.map(p => p.value),
            y: labels,
            marker: { color: colors, opacity: 0.85 },
            text: pairs.map(p => `${sign(p.value)}${p.value.toFixed(1)}`),
            textposition: 'outside',
            textfont: { family: SANS_FONT, size: 10, color: CHART.muted },
            hovertemplate: '<b>%{y}</b><br>SHAP: %{x:.2f} Kg/ha<extra></extra>',
          } as Plotly.Data]}
          layout={{
            ...LAYOUT_BASE,
            margin: { t: 52, r: 80, b: 60, l: 296 },
            height: 640,
            title: {
              text: `SHAP attribution · ${district} (${year}) · base ${Math.round(base)} Kg/ha`,
              font: { family: SERIF_FONT, size: 13, color: CHART.muted },
              x: 0.01,
            },
            xaxis: {
              title: { text: 'SHAP value (Kg/ha)', font: { family: SERIF_FONT, size: 12 } },
              tickfont: { family: SANS_FONT, size: 11 },
              gridcolor: '#eae5dc', gridwidth: 1,
              zeroline: true, zerolinewidth: 1.5, zerolinecolor: '#c8b898',
              linecolor: '#d6d0c4',
            },
            yaxis: {
              automargin: true,
              tickfont: { family: SERIF_FONT, size: 11 },
              gridcolor: '#eae5dc',
              linecolor: '#d6d0c4',
            },
          }}
          config={CONFIG}
          style={{ width: '100%' }}
        />
      </div>

      {/* Policy lever */}
      {ctrlTop && (
        <div className="panel amber" style={{ marginBottom: 20 }}>
          <strong>◆ &nbsp;Policy lever for {district}</strong> &nbsp;—&nbsp;
          <code style={{ background: 'rgba(0,0,0,.06)', padding: '1px 6px', fontSize: '.8rem' }}>
            {ctrlTop.feature}
          </code>
          &nbsp;has the largest controllable SHAP impact ({sign(ctrlTop.value)}{ctrlTop.value.toFixed(1)} Kg/ha).
          Currently <strong>{ctrlTop.value > 0 ? 'above' : 'below'} baseline</strong>
          &nbsp;— maintaining or improving it would {ctrlTop.value > 0 ? 'increase' : 'decrease'} yield.
        </div>
      )}

      {/* Breakdown table */}
      <div className="data-card">
        <div className="data-card-head">Feature breakdown · top 15 by |SHAP|</div>
        <div style={{ overflowX: 'auto' }}>
          <table className="breakdown-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th style={{ textAlign: 'center' }}>SHAP Impact (Kg/ha)</th>
                <th style={{ textAlign: 'center' }}>Category</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((p, i) => (
                <tr key={i}>
                  <td style={{ fontFamily: 'var(--sans)', fontSize: '.78rem', color: 'var(--text-mid)' }}>
                    {p.feature}
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <span className={p.value >= 0 ? 'shap-pos' : 'shap-neg'}>
                      {sign(p.value)}{p.value.toFixed(1)}
                    </span>
                  </td>
                  <td style={{ textAlign: 'center' }}>
                    <span className={getBadgeClass(p.cat)}>{p.cat}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
