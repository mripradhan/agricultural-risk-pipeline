'use client';
import { useDistrict } from '@/context/DistrictContext';

const TABS = [
  { id: 'eda',  label: '◈  Analysis' },
  { id: 'maps', label: '▦  Maps' },
  { id: 'shap', label: '◎  SHAP' },
];

export default function TopNav() {
  const { activeTab, setActiveTab } = useDistrict();

  return (
    <nav className="topnav">
      <div className="nav-brand">RootCause</div>
      <div className="nav-tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab-btn${activeTab === t.id ? ' active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
    </nav>
  );
}
