# AIMEX 프론트엔드 (Next.js 15, standalone 출력)
# 멀티스테이지: deps → build → runtime

# 1) 의존성 설치
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

# 2) 빌드
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* 는 빌드 시점에 번들에 박힌다 → build-arg 로 주입
ARG NEXT_PUBLIC_BACKEND_URL
ARG NEXT_PUBLIC_API_BASE_URL
ARG NEXT_PUBLIC_GOOGLE_CLIENT_ID
ARG NEXT_PUBLIC_GOOGLE_REDIRECT_URI
ARG NEXT_PUBLIC_NAVER_CLIENT_ID
ARG NEXT_PUBLIC_NAVER_REDIRECT_URI
ARG NEXT_PUBLIC_INSTAGRAM_APP_ID
ARG NEXT_PUBLIC_INSTAGRAM_REDIRECT_URI
ENV NEXT_PUBLIC_BACKEND_URL=$NEXT_PUBLIC_BACKEND_URL \
    NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL \
    NEXT_PUBLIC_GOOGLE_CLIENT_ID=$NEXT_PUBLIC_GOOGLE_CLIENT_ID \
    NEXT_PUBLIC_GOOGLE_REDIRECT_URI=$NEXT_PUBLIC_GOOGLE_REDIRECT_URI \
    NEXT_PUBLIC_NAVER_CLIENT_ID=$NEXT_PUBLIC_NAVER_CLIENT_ID \
    NEXT_PUBLIC_NAVER_REDIRECT_URI=$NEXT_PUBLIC_NAVER_REDIRECT_URI \
    NEXT_PUBLIC_INSTAGRAM_APP_ID=$NEXT_PUBLIC_INSTAGRAM_APP_ID \
    NEXT_PUBLIC_INSTAGRAM_REDIRECT_URI=$NEXT_PUBLIC_INSTAGRAM_REDIRECT_URI \
    NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# 3) 런타임 (standalone)
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

# NextAuth 서버 런타임 변수(NEXTAUTH_URL/SECRET, BACKEND_URL)는 env_file 로 주입
CMD ["node", "server.js"]
