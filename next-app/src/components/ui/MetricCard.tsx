interface Props {
  label: string;
  value: string | number;
  sub?:  string;
}

export default function MetricCard({ label, value, sub }: Props) {
  return (
    <div className="metric-card">
      <div className="mc-val">{value}</div>
      <div className="mc-lbl">{label}</div>
      {sub && <div className="mc-sub">{sub}</div>}
    </div>
  );
}
