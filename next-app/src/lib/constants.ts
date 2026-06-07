export const CONTROLLABLE = new Set([
  'NITROGEN CONSUMPTION (tons)',
  'PHOSPHATE CONSUMPTION (tons)',
  'POTASH CONSUMPTION (tons)',
  'TOTAL FERTILISER CONSUMPTION (tons)',
  'GROSS IRRIGATED AREA (1000 ha)',
  'GROSS CROPPED AREA (1000 ha)',
  'RICE AREA (1000 ha)',
]);

export const CHART = {
  ink:    '#1c1c1c',
  muted:  '#7a7a6e',
  accent: '#c8922a',
  pos:    '#2d5a2d',
  neg:    '#7a3030',
  bg:     'rgba(0,0,0,0)',
} as const;

export const SERIF_FONT = "'Playfair Display',Georgia,serif";
export const SANS_FONT  = "'Source Sans 3','Helvetica Neue',sans-serif";

export const LAYOUT_BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font: { family: SANS_FONT, size: 12, color: '#3a3a3a' },
  margin: { t: 52, r: 24, b: 52, l: 64 },
  showlegend: false,
};

export const CONFIG = { responsive: true, displayModeBar: false } as const;
