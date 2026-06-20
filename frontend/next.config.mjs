/** @type {import('next').NextConfig} */
const nextConfig = {
  // Docker 배포 최적화: .next/standalone 으로 최소 런타임 산출 (이미지 경량화)
  output: 'standalone',
  // 개발 모드 인디케이터(좌하단 'N Issues' 배지) 숨김 — 프로덕션엔 원래 없는 dev 전용 UI
  devIndicators: false,
  images: {
    unoptimized: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb', // 10MB로 증가
    },
  },
  // 상대경로 /api/v1/* 호출(갤러리 목록·삭제·상세, 이미지 정적서빙 src 등)을 백엔드로 프록시.
  // 배포 시 BACKEND_INTERNAL_URL(예: http://backend:8000)을 우선 사용.
  async rewrites() {
    const backend =
      process.env.BACKEND_INTERNAL_URL ||
      process.env.NEXT_PUBLIC_BACKEND_URL ||
      'http://localhost:8000'
    return [
      { source: '/api/v1/:path*', destination: `${backend}/api/v1/:path*` },
    ]
  },
}

export default nextConfig
