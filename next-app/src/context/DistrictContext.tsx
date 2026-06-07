'use client';

import { createContext, useContext, useState, useEffect, useMemo, ReactNode } from 'react';
import useSWR from 'swr';
import type { Meta, Districts, MapsData, TabNetData, SHAPData } from '@/lib/types';

const fetcher = (url: string) => fetch(url).then(r => { if (!r.ok) throw new Error(url); return r.json(); });

interface Ctx {
  meta:      Meta      | undefined;
  districts: Districts | undefined;
  mapsData:  MapsData  | undefined;
  tabnet:    TabNetData| undefined;
  shapData:  SHAPData  | undefined;
  isLoading: boolean;
  selState:       string; setSelState:       (v: string) => void;
  selDistrict:    string; setSelDistrict:    (v: string) => void;
  selCrop:        string; setSelCrop:        (v: string) => void;
  activeTab:      string; setActiveTab:      (v: string) => void;
  states:             string[];
  districtsForState:  string[];
}

const DistrictContext = createContext<Ctx | null>(null);

export function DistrictProvider({ children }: { children: ReactNode }) {
  const { data: meta      } = useSWR<Meta>      ('/data/meta.json',      fetcher);
  const { data: districts } = useSWR<Districts> ('/data/districts.json', fetcher);
  const { data: mapsData  } = useSWR<MapsData>  ('/data/maps.json',      fetcher);
  const { data: tabnet    } = useSWR<TabNetData> ('/data/tabnet.json',    fetcher);
  const { data: shapData  } = useSWR<SHAPData>  ('/data/shap.json',      fetcher);

  const [selState,    setSelState]    = useState('');
  const [selDistrict, setSelDistrict] = useState('');
  const [selCrop,     setSelCrop]     = useState('');
  const [activeTab,   setActiveTab]   = useState('eda');

  const states = useMemo(() => {
    if (!districts) return [];
    return [...new Set(Object.values(districts).map(d => d.state))].sort();
  }, [districts]);

  const districtsForState = useMemo(() => {
    if (!districts || !selState) return [];
    return Object.entries(districts)
      .filter(([, v]) => v.state === selState)
      .map(([k]) => k).sort();
  }, [districts, selState]);

  useEffect(() => { if (states.length && !selState) setSelState(states[0]); }, [states, selState]);
  useEffect(() => { if (districtsForState.length) setSelDistrict(districtsForState[0]); }, [selState]); // eslint-disable-line
  useEffect(() => { if (meta?.yield_cols?.length && !selCrop) setSelCrop(meta.yield_cols[0]); }, [meta, selCrop]);

  const isLoading = !meta || !districts || !mapsData || !tabnet || !shapData;

  return (
    <DistrictContext.Provider value={{
      meta, districts, mapsData, tabnet, shapData, isLoading,
      selState, setSelState,
      selDistrict, setSelDistrict,
      selCrop, setSelCrop,
      activeTab, setActiveTab,
      states, districtsForState,
    }}>
      {children}
    </DistrictContext.Provider>
  );
}

export function useDistrict() {
  const ctx = useContext(DistrictContext);
  if (!ctx) throw new Error('useDistrict must be inside DistrictProvider');
  return ctx;
}
