-- ============================================================
-- AIMEX_MAIN 부트스트랩 (스키마 + 더미 데이터)
-- 생성: backend/scripts/export_bootstrap_sql.py
-- 주의: 실시간 mysqldump 백업이 아니라 모델 기준 초기 스키마 + 더미.
-- 복원: mysql -u root -p < aimex_main_20260611.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS AIMEX_MAIN CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE AIMEX_MAIN;
SET FOREIGN_KEY_CHECKS=0;

-- ============================================================
-- 스키마
-- ============================================================
CREATE TABLE IF NOT EXISTS `MCP_SERVER` (
	mcp_id INTEGER NOT NULL COMMENT 'MCP서버 고유식별자' AUTO_INCREMENT, 
	mcp_name VARCHAR(255) NOT NULL COMMENT 'MCP서버 고유 이름', 
	mcp_status INTEGER NOT NULL COMMENT '0: stdio, 1: SSE', 
	mcp_config TEXT NOT NULL COMMENT 'MCP 서버연결 설정값 (JSON 형식)', 
	description VARCHAR(255), 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (mcp_id), 
	UNIQUE (mcp_name)
);

CREATE TABLE IF NOT EXISTS `MODEL_MBTI` (
	mbti_id INTEGER NOT NULL COMMENT 'MBTI 성격 고유 식별자' AUTO_INCREMENT, 
	mbti_name VARCHAR(100) NOT NULL COMMENT 'MBTI 이름', 
	mbti_traits VARCHAR(255) NOT NULL COMMENT 'MBTI 별 성격, 특성', 
	mbti_speech TEXT NOT NULL COMMENT 'MBTI 말투 설명', 
	PRIMARY KEY (mbti_id)
);

CREATE TABLE IF NOT EXISTS `STYLE_PRESET` (
	style_preset_id VARCHAR(255) NOT NULL COMMENT '스타일 프리셋 고유 식별자', 
	style_preset_name VARCHAR(100) NOT NULL COMMENT '스타일 프리셋 이름', 
	influencer_type INTEGER NOT NULL COMMENT '인플루언서 유형', 
	influencer_gender INTEGER NOT NULL COMMENT '인플루언서 성별, 0:남성, 1:여성, 2:없음', 
	influencer_age_group INTEGER NOT NULL COMMENT '인플루언서 연령대, (20대,30대, ...)', 
	influencer_hairstyle VARCHAR(100) NOT NULL COMMENT '인플루언서 헤어 스타일', 
	influencer_style VARCHAR(255) NOT NULL COMMENT '인플루언서 전체 스타일(힙함, 청순 등)', 
	influencer_personality TEXT NOT NULL COMMENT '인플루언서 성격', 
	influencer_speech TEXT NOT NULL COMMENT '인플루언서 말투', 
	mbti_id INTEGER COMMENT 'MBTI 성격 고유 식별자', 
	system_prompt TEXT NOT NULL COMMENT '인플루언서 시스템 프롬프트', 
	influencer_description TEXT NOT NULL COMMENT '인플루언서 설명', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (style_preset_id)
);

CREATE TABLE IF NOT EXISTS `TEAM` (
	group_id INTEGER NOT NULL COMMENT '그룹 고유 식별자' AUTO_INCREMENT, 
	group_name VARCHAR(100) NOT NULL COMMENT '그룹명', 
	group_description TEXT COMMENT '그룹 설명', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (group_id)
);

CREATE TABLE IF NOT EXISTS `USER` (
	user_id VARCHAR(255) NOT NULL COMMENT '내부 사용자 고유 id', 
	provider_id VARCHAR(255) NOT NULL COMMENT '소셜 제공자의 고유 사용자 식별자', 
	provider VARCHAR(20) NOT NULL COMMENT '소셜 로그인 제공자', 
	user_name VARCHAR(20) NOT NULL COMMENT '사용자 이름', 
	email VARCHAR(50) NOT NULL COMMENT '사용자 이메일', 
	current_pod_id VARCHAR(100) COMMENT '현재 활성 RunPod ID', 
	pod_status VARCHAR(20) COMMENT 'Pod 상태: none, starting, ready, processing', 
	session_created_at DATETIME COMMENT '세션 생성 시간', 
	session_expires_at DATETIME COMMENT '세션 만료 시간 (15분)', 
	processing_expires_at DATETIME COMMENT '처리 만료 시간 (10분)', 
	total_generations INTEGER COMMENT '총 이미지 생성 횟수', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (user_id), 
	UNIQUE (provider_id), 
	UNIQUE (email)
);

CREATE TABLE IF NOT EXISTS content_enhancements (
	enhancement_id VARCHAR(36) NOT NULL, 
	user_id VARCHAR(36) NOT NULL, 
	original_content TEXT NOT NULL, 
	enhanced_content TEXT NOT NULL, 
	status VARCHAR(20), 
	openai_model VARCHAR(50), 
	openai_tokens_used INTEGER, 
	openai_cost FLOAT, 
	board_id VARCHAR(36), 
	influencer_id VARCHAR(36), 
	enhancement_prompt TEXT, 
	improvement_notes TEXT, 
	created_at DATETIME DEFAULT now(), 
	updated_at DATETIME, 
	approved_at DATETIME, 
	PRIMARY KEY (enhancement_id)
);

CREATE TABLE IF NOT EXISTS generated_images (
	id INTEGER NOT NULL AUTO_INCREMENT, 
	storage_id VARCHAR(100) NOT NULL, 
	team_id INTEGER NOT NULL, 
	user_id VARCHAR(50) NOT NULL, 
	prompt TEXT, 
	negative_prompt TEXT, 
	width INTEGER NOT NULL, 
	height INTEGER NOT NULL, 
	seed INTEGER, 
	workflow_name VARCHAR(200), 
	model_name VARCHAR(200), 
	extra_metadata JSON, 
	s3_url VARCHAR(500), 
	file_size INTEGER, 
	mime_type VARCHAR(50), 
	created_at DATETIME NOT NULL DEFAULT now(), 
	updated_at DATETIME, 
	PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS image_generation_requests (
	request_id VARCHAR(255) NOT NULL COMMENT '요청 고유 식별자', 
	user_id VARCHAR(255) NOT NULL COMMENT '사용자 ID', 
	prompt TEXT NOT NULL COMMENT '생성 프롬프트', 
	negative_prompt TEXT COMMENT '부정 프롬프트', 
	width INTEGER COMMENT '이미지 너비', 
	height INTEGER COMMENT '이미지 높이', 
	steps INTEGER COMMENT '생성 스텝', 
	cfg_scale FLOAT COMMENT 'CFG 스케일', 
	seed INTEGER COMMENT '시드값', 
	status VARCHAR(50) COMMENT '생성 상태', 
	result_url VARCHAR(1000) COMMENT '결과 이미지 URL', 
	error_message TEXT COMMENT '오류 메시지', 
	processing_time FLOAT COMMENT '처리 시간(초)', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (request_id)
);

CREATE TABLE IF NOT EXISTS pod_sessions (
	session_id VARCHAR(255) NOT NULL COMMENT 'Pod 세션 고유 식별자', 
	user_id VARCHAR(255) NOT NULL COMMENT '사용자 고유 식별자', 
	pod_id VARCHAR(255) NOT NULL COMMENT 'RunPod 인스턴스 ID', 
	pod_endpoint_url VARCHAR(2048) COMMENT 'Pod 엔드포인트 URL', 
	pod_status ENUM('STARTING','READY','PROCESSING','IDLE','TERMINATING','TERMINATED') NOT NULL COMMENT 'Pod 상태', 
	session_status ENUM('INPUT_WAITING','PROCESSING','IDLE','EXPIRED') NOT NULL COMMENT '세션 상태', 
	last_activity_at DATETIME NOT NULL COMMENT '마지막 활동 시간', 
	input_timeout_minutes INTEGER NOT NULL COMMENT '입력 대기 타임아웃 (분)', 
	processing_timeout_minutes INTEGER NOT NULL COMMENT '이미지 생성 타임아웃 (분)', 
	input_deadline DATETIME COMMENT '입력 마감 시간', 
	processing_deadline DATETIME COMMENT '처리 마감 시간', 
	total_generations INTEGER NOT NULL COMMENT '총 이미지 생성 횟수', 
	total_cost VARCHAR(20) NOT NULL COMMENT '총 사용 비용 (USD)', 
	error_message TEXT COMMENT '오류 메시지', 
	pod_config JSON COMMENT 'Pod 설정 정보', 
	terminated_at DATETIME COMMENT '종료 시간', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (session_id), 
	UNIQUE (pod_id)
)COMMENT='RunPod 세션 관리 테이블';

CREATE TABLE IF NOT EXISTS prompt_optimization_usage (
	id VARCHAR(50) NOT NULL, 
	date VARCHAR(10) NOT NULL COMMENT '날짜 (YYYY-MM-DD)', 
	user_id VARCHAR(50) COMMENT '사용자 ID', 
	total_requests INTEGER COMMENT '총 요청 수', 
	successful_requests INTEGER COMMENT '성공한 요청 수', 
	failed_requests INTEGER COMMENT '실패한 요청 수', 
	total_tokens INTEGER COMMENT '총 토큰 사용량', 
	avg_tokens_per_request FLOAT COMMENT '요청당 평균 토큰 수', 
	openai_requests INTEGER COMMENT 'OpenAI 사용 요청 수', 
	mock_requests INTEGER COMMENT 'Mock 사용 요청 수', 
	avg_optimization_time FLOAT COMMENT '평균 최적화 시간', 
	total_optimization_time FLOAT COMMENT '총 최적화 시간', 
	created_at DATETIME COMMENT '생성 시간', 
	updated_at DATETIME COMMENT '수정 시간', 
	PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS prompt_optimizations (
	id VARCHAR(50) NOT NULL, 
	original_prompt TEXT NOT NULL COMMENT '원본 프롬프트', 
	optimized_prompt TEXT NOT NULL COMMENT '최적화된 프롬프트', 
	negative_prompt TEXT COMMENT '네거티브 프롬프트', 
	style VARCHAR(50) COMMENT '스타일', 
	quality_level VARCHAR(20) COMMENT '품질 수준', 
	aspect_ratio VARCHAR(10) COMMENT '종횡비', 
	additional_tags TEXT COMMENT '추가 태그', 
	style_tags JSON COMMENT '스타일 태그 목록', 
	quality_tags JSON COMMENT '품질 태그 목록', 
	optimization_metadata JSON COMMENT '최적화 메타데이터', 
	optimization_method VARCHAR(50) COMMENT '최적화 방법 (openai, mock)', 
	model_used VARCHAR(100) COMMENT '사용된 모델', 
	tokens_used INTEGER COMMENT '사용된 토큰 수', 
	user_id VARCHAR(50) COMMENT '사용자 ID', 
	session_id VARCHAR(100) COMMENT '세션 ID', 
	optimization_time FLOAT COMMENT '최적화 소요 시간 (초)', 
	created_at DATETIME COMMENT '생성 시간', 
	updated_at DATETIME COMMENT '수정 시간', 
	PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS prompt_processing_pipelines (
	pipeline_id VARCHAR(255) NOT NULL COMMENT '파이프라인 고유 식별자', 
	user_id VARCHAR(255) NOT NULL COMMENT '사용자 고유 식별자', 
	session_id VARCHAR(255) COMMENT 'Pod 세션 ID', 
	original_prompt TEXT NOT NULL COMMENT '사용자 입력 원본 프롬프트', 
	style_preset VARCHAR(255) COMMENT '선택된 스타일 프리셋', 
	original_s3_key VARCHAR(1024) COMMENT '원본 프롬프트 S3 키', 
	original_s3_url VARCHAR(2048) COMMENT '원본 프롬프트 S3 URL', 
	original_saved_at DATETIME COMMENT '원본 S3 저장 시간', 
	openai_model_used VARCHAR(100) NOT NULL COMMENT '사용된 OpenAI 모델', 
	openai_request_id VARCHAR(255) COMMENT 'OpenAI 요청 ID', 
	optimized_prompt TEXT COMMENT '최적화된 영문 프롬프트', 
	optimization_status ENUM('PENDING','PROCESSING','COMPLETED','FAILED') NOT NULL COMMENT '최적화 상태', 
	openai_cost VARCHAR(20) NOT NULL COMMENT 'OpenAI 사용 비용 (USD)', 
	optimized_s3_key VARCHAR(1024) COMMENT '최적화된 프롬프트 S3 키', 
	optimized_s3_url VARCHAR(2048) COMMENT '최적화된 프롬프트 S3 URL', 
	optimized_saved_at DATETIME COMMENT '최적화된 프롬프트 S3 저장 시간', 
	pipeline_status ENUM('PENDING','S3_SAVING','OPENAI_PROCESSING','S3_RESAVING','COMPLETED','FAILED') NOT NULL COMMENT '파이프라인 상태', 
	error_message TEXT COMMENT '오류 메시지', 
	processing_metadata JSON COMMENT '처리 관련 추가 정보', 
	completed_at DATETIME COMMENT '완료 시간', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (pipeline_id)
)COMMENT='프롬프트 처리 파이프라인 테이블';

CREATE TABLE IF NOT EXISTS prompt_templates (
	id VARCHAR(50) NOT NULL, 
	name VARCHAR(200) NOT NULL COMMENT '템플릿 이름', 
	description TEXT COMMENT '템플릿 설명', 
	category VARCHAR(50) COMMENT '카테고리', 
	template_prompt TEXT NOT NULL COMMENT '프롬프트 템플릿', 
	template_negative TEXT COMMENT '네거티브 프롬프트 템플릿', 
	default_style VARCHAR(50) COMMENT '기본 스타일', 
	default_quality VARCHAR(20) COMMENT '기본 품질', 
	default_aspect_ratio VARCHAR(10) COMMENT '기본 종횡비', 
	tags JSON COMMENT '태그 목록', 
	variables JSON COMMENT '템플릿 변수 정의', 
	created_by VARCHAR(50) COMMENT '생성자 ID', 
	is_public VARCHAR(1) COMMENT '공개 여부', 
	is_active VARCHAR(1) COMMENT '활성 여부', 
	usage_count INTEGER COMMENT '사용 횟수', 
	created_at DATETIME COMMENT '생성 시간', 
	updated_at DATETIME COMMENT '수정 시간', 
	PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS `HF_TOKEN_MANAGE` (
	hf_manage_id VARCHAR(255) NOT NULL COMMENT '허깅페이스 토큰 관리 고유 식별자', 
	group_id INTEGER COMMENT '그룹 고유 식별자 (NULL 가능 - 할당되지 않은 토큰)', 
	hf_token_value TEXT NOT NULL COMMENT '허깅페이스 실제 토큰 값 (암호화)', 
	hf_token_nickname VARCHAR(100) NOT NULL COMMENT '사용자에게 보여지는 허깅페이스 토큰 별칭', 
	hf_user_name VARCHAR(50) NOT NULL COMMENT '허깅페이스 계정 사용자 이름', 
	is_default BOOL NOT NULL COMMENT '그룹의 기본 토큰 여부', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (hf_manage_id), 
	FOREIGN KEY(group_id) REFERENCES `TEAM` (group_id)
);

CREATE TABLE IF NOT EXISTS `IMAGE_STORAGE` (
	storage_id VARCHAR(255) NOT NULL COMMENT '이미지 저장 고유 식별자', 
	s3_url VARCHAR(1000) NOT NULL COMMENT 'S3 이미지 URL', 
	group_id INTEGER NOT NULL COMMENT '그룹 ID', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (storage_id), 
	FOREIGN KEY(group_id) REFERENCES `TEAM` (group_id)
);

CREATE TABLE IF NOT EXISTS `SYSTEM_LOG` (
	log_id VARCHAR(255) NOT NULL COMMENT '로그 고유 식별자', 
	user_id VARCHAR(255) NOT NULL COMMENT '내부 사용자 고유 식별자', 
	log_type SMALLINT NOT NULL COMMENT '0: API요청, 1: 시스템오류, 2: 인증관련', 
	log_content TEXT NOT NULL COMMENT 'API 요청 내용, 오류 메시지 등 상세한 로그 내용, JSON 형식으로 저장', 
	created_at TIMESTAMP NOT NULL COMMENT '로그 생성일', 
	PRIMARY KEY (log_id), 
	FOREIGN KEY(user_id) REFERENCES `USER` (user_id)
);

CREATE TABLE IF NOT EXISTS `USER_GROUP` (
	user_id VARCHAR(255) NOT NULL, 
	group_id INTEGER NOT NULL, 
	PRIMARY KEY (user_id, group_id), 
	FOREIGN KEY(user_id) REFERENCES `USER` (user_id), 
	FOREIGN KEY(group_id) REFERENCES `TEAM` (group_id)
);

CREATE TABLE IF NOT EXISTS `AI_INFLUENCER` (
	influencer_id VARCHAR(255) NOT NULL COMMENT '인플루언서 고유 식별자', 
	user_id VARCHAR(255) NOT NULL COMMENT '내부 사용자 고유 식별자', 
	group_id INTEGER NOT NULL COMMENT '그룹 고유 식별자', 
	hf_manage_id VARCHAR(255) COMMENT '허깅페이스 토큰 관리 고유 식별자', 
	style_preset_id VARCHAR(255) NOT NULL COMMENT '스타일 프리셋 고유 식별자', 
	mbti_id INTEGER COMMENT 'MBTI 성격 고유 식별자', 
	influencer_name VARCHAR(100) NOT NULL COMMENT 'AI 인플루언서 이름', 
	influencer_description TEXT COMMENT 'AI 인플루언서 설명', 
	image_url TEXT COMMENT '인플루언서 이미지를 받아오면 그대로 사용, 없다면 정보를 기반으로 만들어서 사용', 
	influencer_data_url VARCHAR(255) COMMENT '인플루언서 학습 데이터셋 URL 경로', 
	learning_status INTEGER NOT NULL COMMENT '인플루언서 학습 상태, 0: 학습 중, 1: 사용가능', 
	influencer_model_repo VARCHAR(255) NOT NULL COMMENT '허깅페이스 repo URL 경로', 
	chatbot_option BOOL NOT NULL COMMENT '챗봇 생성 여부', 
	instagram_id VARCHAR(255) COMMENT '연동된 인스타그램 계정 ID', 
	instagram_page_id VARCHAR(255) COMMENT '인스타그램 비즈니스 페이지 ID (웹훅에서 사용)', 
	instagram_username VARCHAR(100) COMMENT '인스타그램 사용자명', 
	instagram_account_type VARCHAR(50) COMMENT '인스타그램 계정 타입 (PERSONAL, BUSINESS, CREATOR)', 
	instagram_access_token TEXT COMMENT '인스타그램 액세스 토큰', 
	instagram_connected_at TIMESTAMP NULL COMMENT '인스타그램 계정 연동 일시', 
	instagram_is_active BOOL COMMENT '인스타그램 연동 활성화 여부', 
	instagram_token_expires_at TIMESTAMP NULL COMMENT '인스타그램 액세스 토큰 만료 일시', 
	influencer_personality TEXT COMMENT 'AI 인플루언서 성격', 
	influencer_tone TEXT COMMENT 'AI 인플루언서 말투/톤', 
	influencer_age_group INTEGER COMMENT 'AI 인플루언서 연령대', 
	voice_option BOOL COMMENT '음성 생성 옵션', 
	image_option BOOL COMMENT '이미지 생성 옵션', 
	system_prompt TEXT COMMENT 'AI 인플루언서 시스템 프롬프트', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	CONSTRAINT pk_ai_influencer PRIMARY KEY (influencer_id), 
	FOREIGN KEY(user_id) REFERENCES `USER` (user_id), 
	FOREIGN KEY(group_id) REFERENCES `TEAM` (group_id), 
	FOREIGN KEY(hf_manage_id) REFERENCES `HF_TOKEN_MANAGE` (hf_manage_id), 
	FOREIGN KEY(style_preset_id) REFERENCES `STYLE_PRESET` (style_preset_id), 
	FOREIGN KEY(mbti_id) REFERENCES `MODEL_MBTI` (mbti_id), 
	UNIQUE (influencer_name)
);

CREATE TABLE IF NOT EXISTS `AI_INFLUENCER_MCP_SERVER` (
	influencer_id VARCHAR(255) NOT NULL, 
	mcp_id INTEGER NOT NULL, 
	PRIMARY KEY (influencer_id, mcp_id), 
	FOREIGN KEY(influencer_id) REFERENCES `AI_INFLUENCER` (influencer_id), 
	FOREIGN KEY(mcp_id) REFERENCES `MCP_SERVER` (mcp_id)
);

CREATE TABLE IF NOT EXISTS `BATCH_KEY` (
	batch_key_id VARCHAR(255) NOT NULL COMMENT '배치키 고유 식별자', 
	influencer_id VARCHAR(255) NOT NULL COMMENT '인플루언서 고유 식별자', 
	batch_key VARCHAR(255) NOT NULL COMMENT '배치키 값', 
	task_id VARCHAR(255) COMMENT 'QA 생성 작업 ID', 
	openai_batch_id VARCHAR(255) COMMENT 'OpenAI 배치 ID', 
	status VARCHAR(50) COMMENT '배치 상태', 
	total_qa_pairs INTEGER COMMENT '총 QA 쌍 수', 
	generated_qa_pairs INTEGER COMMENT '생성된 QA 쌍 수', 
	input_file_id VARCHAR(255) COMMENT '입력 파일 ID', 
	output_file_id VARCHAR(255) COMMENT '출력 파일 ID', 
	error_message TEXT COMMENT '오류 메시지', 
	vllm_task_id VARCHAR(255) COMMENT 'VLLM/RunPod 파인튜닝 작업 ID', 
	s3_qa_file_url VARCHAR(500) COMMENT 'S3 QA 파일 URL', 
	s3_processed_file_url VARCHAR(500) COMMENT 'S3 처리된 파일 URL', 
	is_processed BOOL COMMENT '결과 처리 완료 여부', 
	is_uploaded_to_s3 BOOL COMMENT 'S3 업로드 완료 여부', 
	is_finetuning_started BOOL COMMENT '파인튜닝 시작 여부', 
	created_at DATETIME COMMENT '생성 시간', 
	updated_at DATETIME COMMENT '수정 시간', 
	completed_at DATETIME COMMENT '완료 시간', 
	PRIMARY KEY (batch_key_id), 
	FOREIGN KEY(influencer_id) REFERENCES `AI_INFLUENCER` (influencer_id)
);

CREATE TABLE IF NOT EXISTS `BOARD` (
	board_id VARCHAR(255) NOT NULL COMMENT '게시물 고유 식별자', 
	influencer_id VARCHAR(255) NOT NULL COMMENT '인플루언서 고유 식별자', 
	user_id VARCHAR(255) NOT NULL COMMENT '내부 사용자 고유 식별자', 
	team_id INTEGER NOT NULL COMMENT '팀 고유 식별자', 
	group_id INTEGER NOT NULL COMMENT '그룹 고유 식별자', 
	board_topic VARCHAR(255) NOT NULL COMMENT '게시글의 주제 또는 카테고리명', 
	board_description TEXT COMMENT '게시글의 상세 설명', 
	board_platform INTEGER NOT NULL COMMENT '0:인스타그램, 1:블로그, 2:페이스북', 
	board_hash_tag TEXT COMMENT '해시태그 리스트, JSON 형식으로 저장', 
	board_status INTEGER NOT NULL COMMENT '1:임시저장, 2:예약상태, 3:발행됨', 
	image_url TEXT NOT NULL COMMENT '게시글 썸네일 또는 대표 이미지 URL 경로', 
	reservation_at TIMESTAMP NULL COMMENT '예약 발행 시간', 
	published_at TIMESTAMP NULL COMMENT '실제 발행 시간', 
	platform_post_id VARCHAR(255) COMMENT '각 플랫폼에 업로드된 게시글의 post ID (인스타그램, 페이스북, 블로그 등)', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (board_id), 
	FOREIGN KEY(user_id, group_id) REFERENCES `USER_GROUP` (user_id, group_id) ON DELETE CASCADE ON UPDATE CASCADE, 
	FOREIGN KEY(influencer_id) REFERENCES `AI_INFLUENCER` (influencer_id)
);

CREATE TABLE IF NOT EXISTS `CHAT_MESSAGE` (
	chat_message_id VARCHAR(36) NOT NULL COMMENT '채팅 메시지 고유 식별자', 
	session_id VARCHAR(36) NOT NULL COMMENT '대화 세션 고유 식별자', 
	influencer_id VARCHAR(255) NOT NULL COMMENT '인플루언서 고유 식별자', 
	message_content TEXT NOT NULL COMMENT '대화 내용', 
	message_type VARCHAR(20) NOT NULL COMMENT '메시지 타입 (user/ai)', 
	created_at DATETIME NOT NULL COMMENT '메시지 생성 시각' DEFAULT now(), 
	end_at DATETIME COMMENT '세션 종료 시각', 
	PRIMARY KEY (chat_message_id), 
	FOREIGN KEY(influencer_id) REFERENCES `AI_INFLUENCER` (influencer_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS `GENERATED_TONE` (
	tone_id VARCHAR(255) NOT NULL COMMENT '생성된 어투 고유 식별자', 
	influencer_id VARCHAR(255) NOT NULL COMMENT '인플루언서 고유 식별자', 
	title VARCHAR(255) NOT NULL COMMENT '어투 제목 (예: 말투 1)', 
	example TEXT NOT NULL COMMENT '어투 예시 대화', 
	tone_description VARCHAR(255) NOT NULL COMMENT '어투 설명', 
	hashtags VARCHAR(255) COMMENT '어투 관련 해시태그', 
	system_prompt TEXT NOT NULL COMMENT '어투 생성에 사용된 시스템 프롬프트', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (tone_id), 
	FOREIGN KEY(influencer_id) REFERENCES `AI_INFLUENCER` (influencer_id)
);

CREATE TABLE IF NOT EXISTS `INFLUENCER_API` (
	api_id VARCHAR(255) NOT NULL COMMENT 'API 고유 식별자', 
	influencer_id VARCHAR(255) NOT NULL COMMENT '모델 고유 식별자', 
	api_value VARCHAR(255) NOT NULL COMMENT '발급된 API 값', 
	created_at DATETIME NOT NULL COMMENT '생성 시각', 
	updated_at DATETIME NOT NULL COMMENT '마지막 수정 시각', 
	PRIMARY KEY (api_id), 
	FOREIGN KEY(influencer_id) REFERENCES `AI_INFLUENCER` (influencer_id), 
	UNIQUE (api_value)
);

CREATE TABLE IF NOT EXISTS conversations (
	conversation_id VARCHAR(255) NOT NULL, 
	influencer_id VARCHAR(255) NOT NULL, 
	user_instagram_id VARCHAR(255) NOT NULL, 
	user_instagram_username VARCHAR(255), 
	started_at DATETIME NOT NULL DEFAULT now(), 
	last_message_at DATETIME NOT NULL DEFAULT now(), 
	is_active BOOL NOT NULL, 
	total_messages INTEGER NOT NULL, 
	PRIMARY KEY (conversation_id), 
	FOREIGN KEY(influencer_id) REFERENCES `AI_INFLUENCER` (influencer_id)
);

CREATE TABLE IF NOT EXISTS voice_base (
	id INTEGER NOT NULL AUTO_INCREMENT, 
	influencer_id VARCHAR(255) NOT NULL, 
	file_name VARCHAR(255) NOT NULL, 
	file_size INTEGER, 
	file_type VARCHAR(50), 
	s3_url TEXT NOT NULL, 
	s3_key VARCHAR(500) NOT NULL, 
	duration FLOAT, 
	created_at DATETIME DEFAULT now(), 
	updated_at DATETIME DEFAULT now(), 
	PRIMARY KEY (id), 
	UNIQUE (influencer_id), 
	FOREIGN KEY(influencer_id) REFERENCES `AI_INFLUENCER` (influencer_id)
);

CREATE TABLE IF NOT EXISTS `API_CALL_AGGREGATION` (
	api_call_id VARCHAR(255) NOT NULL COMMENT 'API호출 집계 고유 식별자', 
	api_id VARCHAR(255) NOT NULL COMMENT 'API 고유 식별자', 
	influencer_id VARCHAR(255) NOT NULL COMMENT '모델 고유 식별자', 
	daily_call_count INTEGER NOT NULL COMMENT '일일 API 호출 횟수', 
	created_at TIMESTAMP NOT NULL COMMENT '일일 API 집계 데이터 생성일', 
	updated_at TIMESTAMP NOT NULL COMMENT '일일 API 집계 데이터 수정일', 
	PRIMARY KEY (api_call_id), 
	FOREIGN KEY(api_id) REFERENCES `INFLUENCER_API` (api_id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS conversation_messages (
	message_id VARCHAR(255) NOT NULL, 
	conversation_id VARCHAR(255) NOT NULL, 
	sender_type VARCHAR(20) NOT NULL, 
	sender_instagram_id VARCHAR(255) NOT NULL, 
	message_text TEXT NOT NULL, 
	sent_at DATETIME NOT NULL DEFAULT now(), 
	instagram_message_id VARCHAR(255), 
	is_echo BOOL NOT NULL, 
	generation_time_ms INTEGER, 
	model_used VARCHAR(255), 
	system_prompt_used TEXT, 
	PRIMARY KEY (message_id), 
	FOREIGN KEY(conversation_id) REFERENCES conversations (conversation_id)
);

CREATE TABLE IF NOT EXISTS generated_voice (
	id INTEGER NOT NULL AUTO_INCREMENT, 
	influencer_id VARCHAR(255) NOT NULL, 
	base_voice_id INTEGER NOT NULL, 
	text TEXT NOT NULL, 
	task_id VARCHAR(255), 
	status VARCHAR(50), 
	s3_url TEXT, 
	s3_key VARCHAR(500), 
	duration FLOAT, 
	file_size INTEGER, 
	is_deleted BOOL, 
	created_at DATETIME DEFAULT now(), 
	updated_at DATETIME, 
	PRIMARY KEY (id), 
	FOREIGN KEY(influencer_id) REFERENCES `AI_INFLUENCER` (influencer_id), 
	FOREIGN KEY(base_voice_id) REFERENCES voice_base (id)
);


-- ============================================================
-- 더미 데이터 (로컬 검증용)
-- ============================================================
INSERT IGNORE INTO TEAM (group_id, group_name, group_description, created_at, updated_at)
VALUES (1, 'admin', 'Administrator group', NOW(), NOW());

INSERT IGNORE INTO USER (user_id, provider_id, provider, user_name, email, created_at, updated_at)
VALUES ('b70cf91a-9823-4a5f-adc9-05bec4a5eab4', 'devtest-001', 'google', '개발 테스트 사용자', 'devtest@example.com', NOW(), NOW());

INSERT IGNORE INTO USER_GROUP (user_id, group_id) VALUES ('b70cf91a-9823-4a5f-adc9-05bec4a5eab4', 1);

INSERT IGNORE INTO MODEL_MBTI (mbti_id, mbti_name, mbti_traits, mbti_speech)
VALUES (16, 'ENFP', '외향적, 직관적, 감정적, 인식형', '에너지 넘치고 공감하는 말투');

INSERT IGNORE INTO STYLE_PRESET
(style_preset_id, style_preset_name, influencer_type, influencer_gender, influencer_age_group,
 influencer_hairstyle, influencer_style, influencer_personality, influencer_speech, mbti_id,
 system_prompt, influencer_description, created_at, updated_at)
VALUES
('dummy-style-preset-0001', '더미 스타일 프리셋', 0, 1, 20,
 '긴 웨이브 헤어', '청순하고 발랄한 캐주얼', '활발하고 친근함', '친근한 반말체', 16,
 '당신은 밝고 친근한 AI 인플루언서입니다.', '로컬 검증용 더미 스타일 프리셋', NOW(), NOW());

INSERT IGNORE INTO AI_INFLUENCER
(influencer_id, user_id, group_id, style_preset_id, mbti_id, influencer_name, influencer_description,
 image_url, learning_status, influencer_model_repo, chatbot_option,
 influencer_personality, influencer_tone, influencer_age_group, voice_option, image_option, system_prompt,
 created_at, updated_at)
VALUES
('dummy-influencer-0001', 'b70cf91a-9823-4a5f-adc9-05bec4a5eab4', 1, 'dummy-style-preset-0001', 16, '테스트 인플루언서 - 지나',
 '로컬 검증용 더미 인플루언서입니다. 밝고 친근한 20대 여성 콘셉트.', '/placeholder-user.jpg',
 1, 'dummy/local-test-model', 1, '활발하고 긍정적이며 호기심이 많음', '친근한 반말과 이모지를 즐겨 사용',
 20, 1, 1, '당신은 로컬 검증용 더미 AI 인플루언서입니다.', NOW(), NOW()),
('dummy-influencer-0002', 'b70cf91a-9823-4a5f-adc9-05bec4a5eab4', 1, 'dummy-style-preset-0001', 16, '테스트 인플루언서 - 카이',
 '로컬 검증용 더미 인플루언서입니다. 차분하고 전문적인 30대 남성 콘셉트.', '/placeholder-user.jpg',
 1, 'dummy/local-test-model', 1, '침착하고 분석적이며 신뢰감을 줌', '정중한 존댓말, 군더더기 없는 설명체',
 30, 1, 1, '당신은 로컬 검증용 더미 AI 인플루언서입니다.', NOW(), NOW());

SET FOREIGN_KEY_CHECKS=1;
