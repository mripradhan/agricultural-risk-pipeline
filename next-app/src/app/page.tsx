'use client';
import { useRouter } from 'next/navigation';

export default function LandingPage() {
  const router = useRouter();

  return (
    <section className="landing">
      <div className="hero-left">
        <div className="hero-overline">ICRISAT · IMD · 560 Districts · 1990–2015</div>
        <h1 className="hero-headline">
          District-Level<br />
          Crop Yield<br />
          Intelligence
        </h1>
        <p className="hero-body">
          An explainability observatory for rice yield across 560 Indian districts.
          Integrates ICRISAT agronomy data, IMD rainfall records, and district-level
          fertilizer consumption to train a gradient-boosted model — then applies SHAP
          attribution to decompose each prediction into controllable and uncontrollable
          policy levers.
        </p>
        <div className="hero-rule" />
        <button className="hero-begin" onClick={() => router.push('/dashboard')}>
          Begin Analysis &nbsp; ▸
        </button>
      </div>

      <div className="hero-right">
        <svg className="hero-illustration" viewBox="0 0 380 460" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <defs>
            <pattern id="dot-grid" x="0" y="0" width="24" height="24" patternUnits="userSpaceOnUse">
              <circle cx="12" cy="12" r="1.3" fill="#c8922a" opacity="0.2" />
            </pattern>
          </defs>
          <rect width="380" height="460" fill="url(#dot-grid)" />
          <circle cx="190" cy="230" r="155" fill="none" stroke="#1c1c1c" strokeWidth="0.5" opacity="0.08" />
          <circle cx="190" cy="230" r="115" fill="none" stroke="#1c1c1c" strokeWidth="0.5" opacity="0.12" />
          <circle cx="190" cy="230" r="76"  fill="none" stroke="#1c1c1c" strokeWidth="0.7" opacity="0.16" />
          <circle cx="190" cy="230" r="38"  fill="none" stroke="#c8922a" strokeWidth="1"   opacity="0.45" />
          <circle cx="190" cy="230" r="4"   fill="#c8922a" opacity="0.6" />
          <line x1="190" y1="60"  x2="190" y2="400" stroke="#1c1c1c" strokeWidth="0.4" opacity="0.1" strokeDasharray="5 8" />
          <line x1="30"  y1="230" x2="350" y2="230" stroke="#1c1c1c" strokeWidth="0.4" opacity="0.1" strokeDasharray="5 8" />
          <circle cx="136" cy="162" r="5"   fill="#1c1c1c" opacity="0.2" />
          <circle cx="228" cy="148" r="7"   fill="#c8922a" opacity="0.45" />
          <circle cx="255" cy="278" r="4.5" fill="#1c1c1c" opacity="0.2" />
          <circle cx="145" cy="295" r="6"   fill="#c8922a" opacity="0.4" />
          <circle cx="200" cy="170" r="3"   fill="#1c1c1c" opacity="0.18" />
          <circle cx="162" cy="248" r="3.5" fill="#1c1c1c" opacity="0.18" />
          <circle cx="218" cy="306" r="3"   fill="#c8922a" opacity="0.3" />
          <circle cx="118" cy="210" r="4"   fill="#1c1c1c" opacity="0.15" />
          <text x="190" y="432" fontFamily="'Playfair Display',serif" fontSize="12" fill="#a8a89a" textAnchor="middle" fontStyle="italic">
            district yield observatory
          </text>
          <line x1="130" y1="442" x2="250" y2="442" stroke="#d6d0c4" strokeWidth="0.7" />
        </svg>
      </div>
    </section>
  );
}
