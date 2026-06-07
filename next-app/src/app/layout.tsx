import type { Metadata } from 'next';
import '@/styles/globals.css';

export const metadata: Metadata = {
  title: 'RootCause — District Yield Explainability',
  description: 'District-level crop yield intelligence for 560 Indian districts using XGBoost + SHAP',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600&family=Roboto+Slab:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;1,400&display=swap"
          rel="stylesheet"
        />
        <style>{`
          :root {
            --font-sans:    'Manrope', 'Source Sans 3', 'Helvetica Neue', Arial, sans-serif;
            --font-heading: 'Roboto Slab', 'Playfair Display', Georgia, serif;
          }
        `}</style>
      </head>
      <body>{children}</body>
    </html>
  );
}
