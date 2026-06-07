'use client';
import dynamic from 'next/dynamic';
import type { PlotParams } from 'react-plotly.js';

// Single dynamic-imported Plotly component with full types
const Plot = dynamic<PlotParams>(
  () => import('react-plotly.js'),
  { ssr: false, loading: () => <div style={{ height: '100%', minHeight: 200 }} /> }
);

export default Plot;
