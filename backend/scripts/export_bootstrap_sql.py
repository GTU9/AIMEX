"""
AIMEX_MAIN 부트스트랩 SQL 생성기 (MySQL 서버 불필요).

SQLAlchemy 모델 메타데이터에서 MySQL용 CREATE TABLE DDL을 뽑고, 로컬 검증용
더미 데이터(admin 팀 + 테스트 사용자 + MBTI/스타일프리셋 + 인플루언서 2명) INSERT를
덧붙여 단일 .sql 파일로 저장한다. 이 파일 하나로 빈 MySQL에 스키마+더미를 복원할 수 있다.

  - 실시간 mysqldump 백업이 아니라 "스키마 + 더미데이터" 부트스트랩이다.
    (현재 실행 중인 DB 상태가 아니라 모델 정의 기준 = 깨끗한 초기 상태)

실행:
    cd backend
    python -m scripts.export_bootstrap_sql

복원:
    mysql -u root -p < backups/aimex_main_20260611.sql
"""

import os

# 매퍼 관계 해소를 위해 전체 모델 등록 (app.models.__init__ 미import 모듈 포함)
import app.models  # noqa: F401
from app.models import (  # noqa: F401
    conversation,
    image_generation,
    content_enhancement,
    prompt_optimization,
    image_storage,
)
from app.models.base import Base
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import mysql

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "backups",
    "aimex_main_20260611.sql",
)

# 더미 식별자 (고정 — 멱등 복원)
USER_ID = "b70cf91a-9823-4a5f-adc9-05bec4a5eab4"
PRESET_ID = "dummy-style-preset-0001"
INF1 = "dummy-influencer-0001"
INF2 = "dummy-influencer-0002"

DUMMY_DATA = f"""
-- ============================================================
-- 더미 데이터 (로컬 검증용)
-- ============================================================
INSERT IGNORE INTO TEAM (group_id, group_name, group_description, created_at, updated_at)
VALUES (1, 'admin', 'Administrator group', NOW(), NOW());

INSERT IGNORE INTO USER (user_id, provider_id, provider, user_name, email, created_at, updated_at)
VALUES ('{USER_ID}', 'devtest-001', 'google', '개발 테스트 사용자', 'devtest@example.com', NOW(), NOW());

INSERT IGNORE INTO USER_GROUP (user_id, group_id) VALUES ('{USER_ID}', 1);

INSERT IGNORE INTO MODEL_MBTI (mbti_id, mbti_name, mbti_traits, mbti_speech)
VALUES (16, 'ENFP', '외향적, 직관적, 감정적, 인식형', '에너지 넘치고 공감하는 말투');

INSERT IGNORE INTO STYLE_PRESET
(style_preset_id, style_preset_name, influencer_type, influencer_gender, influencer_age_group,
 influencer_hairstyle, influencer_style, influencer_personality, influencer_speech, mbti_id,
 system_prompt, influencer_description, created_at, updated_at)
VALUES
('{PRESET_ID}', '더미 스타일 프리셋', 0, 1, 20,
 '긴 웨이브 헤어', '청순하고 발랄한 캐주얼', '활발하고 친근함', '친근한 반말체', 16,
 '당신은 밝고 친근한 AI 인플루언서입니다.', '로컬 검증용 더미 스타일 프리셋', NOW(), NOW());

INSERT IGNORE INTO AI_INFLUENCER
(influencer_id, user_id, group_id, style_preset_id, mbti_id, influencer_name, influencer_description,
 image_url, learning_status, influencer_model_repo, chatbot_option,
 influencer_personality, influencer_tone, influencer_age_group, voice_option, image_option, system_prompt,
 created_at, updated_at)
VALUES
('{INF1}', '{USER_ID}', 1, '{PRESET_ID}', 16, '테스트 인플루언서 - 지나',
 '로컬 검증용 더미 인플루언서입니다. 밝고 친근한 20대 여성 콘셉트.', '/placeholder-user.jpg',
 1, 'dummy/local-test-model', 1, '활발하고 긍정적이며 호기심이 많음', '친근한 반말과 이모지를 즐겨 사용',
 20, 1, 1, '당신은 로컬 검증용 더미 AI 인플루언서입니다.', NOW(), NOW()),
('{INF2}', '{USER_ID}', 1, '{PRESET_ID}', 16, '테스트 인플루언서 - 카이',
 '로컬 검증용 더미 인플루언서입니다. 차분하고 전문적인 30대 남성 콘셉트.', '/placeholder-user.jpg',
 1, 'dummy/local-test-model', 1, '침착하고 분석적이며 신뢰감을 줌', '정중한 존댓말, 군더더기 없는 설명체',
 30, 1, 1, '당신은 로컬 검증용 더미 AI 인플루언서입니다.', NOW(), NOW());
"""


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dialect = mysql.dialect()

    parts = [
        "-- ============================================================",
        "-- AIMEX_MAIN 부트스트랩 (스키마 + 더미 데이터)",
        "-- 생성: backend/scripts/export_bootstrap_sql.py",
        "-- 주의: 실시간 mysqldump 백업이 아니라 모델 기준 초기 스키마 + 더미.",
        "-- 복원: mysql -u root -p < aimex_main_20260611.sql",
        "-- ============================================================",
        "",
        "CREATE DATABASE IF NOT EXISTS AIMEX_MAIN CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
        "USE AIMEX_MAIN;",
        "SET FOREIGN_KEY_CHECKS=0;",
        "",
        "-- ============================================================",
        "-- 스키마",
        "-- ============================================================",
    ]

    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=dialect)).strip()
        ddl = ddl.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
        parts.append(ddl.rstrip().rstrip(";") + ";")
        parts.append("")

    parts.append(DUMMY_DATA)
    parts.append("SET FOREIGN_KEY_CHECKS=1;")
    parts.append("")

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts))

    print(f"OK written: {OUT}")
    print(f"tables: {len(Base.metadata.sorted_tables)}")


if __name__ == "__main__":
    main()
