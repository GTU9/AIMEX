/** @type {import('next').NextConfig} */
const nextConfig = {
  // Docker 배포 최적화: .next/standalone 으로 최소 런타임 산출 (이미지 경량화)
  output: 'standalone',
  images: {
    unoptimized: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb', // 10MB로 증가
    },
  },
}

export default nextConfig
