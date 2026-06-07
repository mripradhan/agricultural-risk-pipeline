import { DistrictProvider } from '@/context/DistrictContext';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return <DistrictProvider>{children}</DistrictProvider>;
}
