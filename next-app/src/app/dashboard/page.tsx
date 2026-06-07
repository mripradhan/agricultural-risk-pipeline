'use client';
import { useDistrict } from '@/context/DistrictContext';
import Sidebar from '@/components/Sidebar';
import TopNav from '@/components/TopNav';
import StatBar from '@/components/StatBar';
import AnalysisTab from '@/components/tabs/AnalysisTab';
import MapsTab from '@/components/tabs/MapsTab';
import SHAPTab from '@/components/tabs/SHAPTab';

function Loader() {
  return (
    <div className="loader">
      <div className="loader-ring" />
      <div className="loader-text">Loading observatory data</div>
    </div>
  );
}

export default function DashboardPage() {
  const { isLoading, activeTab } = useDistrict();

  if (isLoading) return <Loader />;

  return (
    <div className="app">
      <Sidebar />
      <div className="main">
        <TopNav />
        <StatBar />
        <div className="content">
          {activeTab === 'eda'  && <AnalysisTab />}
          {activeTab === 'maps' && <MapsTab />}
          {activeTab === 'shap' && <SHAPTab />}
        </div>
      </div>
    </div>
  );
}
