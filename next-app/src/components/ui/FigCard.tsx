'use client';
import { useState } from 'react';
import type { Figure } from '@/lib/types';

interface Props {
  fig: Figure;
  basePath: string;
}

export default function FigCard({ fig, basePath }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="fig-card">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={`${basePath}/${fig.file}`}
        alt={fig.label}
        loading="lazy"
        onError={e => { (e.currentTarget.parentElement as HTMLElement).style.display = 'none'; }}
      />
      <div className="fig-caption">
        <div className="fig-label">{fig.label}</div>
        <button className="fig-toggle" onClick={() => setOpen(o => !o)}>
          {open ? '↑ Hide' : '↳ What does this show?'}
        </button>
        {open && <div className="fig-body">{fig.caption}</div>}
      </div>
    </div>
  );
}
