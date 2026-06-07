import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Allow serving large JSON files from public/data without size warnings
  // Allow iframes for Folium choropleth maps
  async headers() {
    return [
      {
        source: '/maps/:path*',
        headers: [{ key: 'X-Frame-Options', value: 'SAMEORIGIN' }],
      },
    ];
  },
};

export default nextConfig;
