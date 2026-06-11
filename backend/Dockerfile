# AIMEX 백엔드 (NAS 하이브리드 배포용)
# GPU 추론은 RunPod 에 위임하므로 경량 requirements-nas.txt 사용.
FROM python:3.10-slim

# 런타임/빌드 필수 시스템 패키지 (일부 휠 미제공 패키지 대비)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시)
COPY requirements-nas.txt .
RUN pip install --upgrade pip && pip install -r requirements-nas.txt

# 앱 소스 복사
COPY . .

EXPOSE 8000

# 헬스체크 (compose 에서도 사용 가능)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# 기본 실행. 마이그레이션은 docker-compose 의 command 에서 선행 처리.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
