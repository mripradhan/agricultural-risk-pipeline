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
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/figures/agripix.png"
          alt="Agricultural field"
          className="hero-illustration"
        />
      </div>
    </section>
  );
}
