from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    Query,
    Depends,
    HTTPException,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import HFTokenManage
from app.services.runpod_manager import get_vllm_manager, get_tts_manager
from app.services.s3_service import S3Service
from app.core.encryption import decrypt_sensitive_data
from app.services.hf_token_resolver import get_token_by_group
from app.services.chat_message_service import ChatMessageService
import json
import logging
import base64
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from app.core.config import settings
import os
import httpx
from app.services.s3_image_service import get_s3_image_service

router = APIRouter()
logger = logging.getLogger(__name__)


async def _process_tts_async(websocket: WebSocket, text: str, influencer_id: str):
    """비동기로 TTS 처리하고 완료되면 base64 오디오 데이터 전송"""
    from app.models.influencer import AIInfluencer
    from sqlalchemy.orm import Session
    from app.database import get_db
    import asyncio
    
    import base64 as _b64
    import os as _os

    try:
        # DB에서 influencer 의 베이스 음성(로컬 파일) 읽기 → base64 (클로닝 참조)
        db: Session = next(get_db())
        voice_b64 = None
        try:
            influencer = db.query(AIInfluencer).filter(
                AIInfluencer.influencer_id == influencer_id
            ).first()
            if influencer and influencer.voice_base and influencer.voice_base.s3_url:
                vpath = influencer.voice_base.s3_url  # 로컬 경로
                try:
                    if _os.path.exists(vpath):
                        with open(vpath, "rb") as vf:
                            voice_b64 = _b64.b64encode(vf.read()).decode()
                        logger.info(f"[WS] 베이스 음성 로드(클로닝): {vpath}")
                    else:
                        logger.warning(f"[WS] 베이스 음성 파일 없음: {vpath}")
                except Exception as ve:
                    logger.warning(f"[WS] 베이스 음성 읽기 실패: {ve}")
            else:
                logger.info(f"[WS] 베이스 음성 미설정 → 기본 음색으로 합성")
        finally:
            db.close()

        tts_manager = get_tts_manager()
        logger.info(f"[WS] TTS 생성 요청: {text[:50]}...")
        job_input = {
            "text": text,
            "influencer_id": influencer_id,
            "voice_data_base64": voice_b64,  # 있으면 클로닝, 없으면 기본 음색
        }

        tts_result = await tts_manager.runsync(job_input)

        # CosyVoice/Modal 응답 계약: {"output": {audio_base64, sample_rate, duration}}
        output = (tts_result or {}).get("output", {}) or {}
        audio_base64 = output.get("audio_base64") or output.get("audio_data") or output.get("audio")

        if not audio_base64:
            err = (tts_result or {}).get("error")
            logger.warning(f"[WS] TTS 오디오 없음 (error={err}, keys={list(output.keys())})")
            return

        try:
            await websocket.send_text(
                json.dumps({
                    "type": "audio",
                    "audio_base64": audio_base64,
                    "duration": output.get("duration"),
                    "format": output.get("format", "wav"),
                    "message": "음성이 생성되었습니다.",
                })
            )
            logger.info(f"[WS] TTS base64 오디오 전송 완료 (크기: {len(audio_base64)})")
        except Exception as send_error:
            logger.error(f"[WS] TTS 오디오 전송 실패(연결 끊김?): {send_error}")

    except Exception as e:
        logger.error(f"[WS] TTS 처리 중 오류: {e}")
        # TTS 오류는 무시하고 채팅은 계속 진행


# 메모리 히스토리 클래스 제거 - 데이터베이스만 사용


class ModelLoadRequest(BaseModel):
    lora_repo: str
    group_id: int


# 전역 히스토리 저장소 제거 - 데이터베이스만 사용


@router.websocket("/chatbot/{lora_repo}")
async def chatbot(
    websocket: WebSocket,
    lora_repo: str,
):
    # 매우 상세한 연결 정보 로그
    client_host = websocket.client.host if websocket.client else "unknown"
    client_port = websocket.client.port if websocket.client else "unknown"
    
    logger.info(f"🔗 [WS] WebSocket 연결 요청 시작")
    logger.info(f"🔗 [WS] Client: {client_host}:{client_port}")
    logger.info(f"🔗 [WS] Path: {websocket.scope.get('path', 'unknown')}")
    logger.info(f"🔗 [WS] Method: {websocket.scope.get('method', 'unknown')}")
    logger.info(f"🔗 [WS] Scheme: {websocket.scope.get('scheme', 'unknown')}")
    logger.info(f"🔗 [WS] Full URL: {websocket.url}")
    logger.info(f"🔗 [WS] Scope keys: {list(websocket.scope.keys())}")
    
    # Headers 로깅 (보안상 민감한 정보 제외)
    headers = dict(websocket.scope.get("headers", []))
    safe_headers = {}
    for header_name, header_value in headers.items():
        header_name_str = header_name.decode() if isinstance(header_name, bytes) else str(header_name)
        header_value_str = header_value.decode() if isinstance(header_value, bytes) else str(header_value)
        
        # 민감한 헤더는 마스킹
        if header_name_str.lower() in ['authorization', 'cookie', 'token']:
            safe_headers[header_name_str] = f"{header_value_str[:10]}..." if len(header_value_str) > 10 else "***"
        else:
            safe_headers[header_name_str] = header_value_str
    
    logger.info(f"🔗 [WS] Headers: {safe_headers}")
    
    # WebSocket query 파라미터 수동 파싱
    try:
        from urllib.parse import parse_qs, urlparse
        query_string = str(websocket.scope.get("query_string", b""), "utf-8")
        query_params = parse_qs(query_string)
        
        logger.info(f"[WS] Raw query string: {query_string}")
        logger.info(f"[WS] Parsed query params: {query_params}")
        
        # 필수 파라미터 추출
        group_id = query_params.get("group_id", [None])[0]
        influencer_id = query_params.get("influencer_id", [None])[0]
        token = query_params.get("token", [None])[0]
        
        logger.info(f"[WS] 요청 파라미터: lora_repo={lora_repo}, group_id={group_id}, influencer_id={influencer_id}")
        
        # influencer_id만 필수로 체크 (group_id는 선택적)
        if not influencer_id:
            logger.error(f"[WS] influencer_id 파라미터가 없음")
            await websocket.close(code=1003, reason="Missing influencer_id parameter")
            return
            
        if not token:
            logger.error(f"[WS] token 파라미터가 없음")
            await websocket.close(code=1003, reason="Missing token parameter")
            return
        
        # group_id가 없으면 influencer_id로 조회
        if not group_id:
            from app.models.influencer import AIInfluencer
            db_temp = next(get_db())
            try:
                influencer = db_temp.query(AIInfluencer).filter(
                    AIInfluencer.influencer_id == influencer_id
                ).first()
                if influencer:
                    group_id = str(influencer.group_id)
                    logger.info(f"[WS] DB에서 group_id 조회 성공: {group_id}")
                else:
                    logger.error(f"[WS] 인플루언서 {influencer_id}를 찾을 수 없음")
                    await websocket.close(code=1003, reason="Influencer not found")
                    return
            finally:
                db_temp.close()
        
        try:
            group_id = int(group_id)
        except (ValueError, TypeError):
            logger.error(f"[WS] group_id가 유효한 정수가 아님: {group_id}")
            await websocket.close(code=1003, reason="Invalid group_id parameter")
            return
        
        logger.info(f"[WS] 토큰 길이: {len(token)}자")
        logger.info(f"[WS] 토큰 앞 50자: {token[:50]}..." if len(token) > 50 else f"[WS] 토큰 전체: {token}")
        
    except Exception as e:
        logger.error(f"[WS] Query 파라미터 파싱 실패: {e}")
        logger.error(f"[WS] Exception type: {type(e).__name__}")
        logger.error(f"[WS] Full scope: {websocket.scope}")
        import traceback
        logger.error(f"[WS] Traceback: {traceback.format_exc()}")
        await websocket.close(code=1003, reason="Parameter parsing failed")
        return
    
    # WebSocket 연결을 먼저 수락
    try:
        await websocket.accept()
        logger.info(f"[WS] WebSocket 연결 수락 완료")
        logger.info(f"[WS] Connection state: {websocket.client_state}")
        logger.info(f"[WS] Application state: {websocket.application_state}")
    except Exception as e:
        logger.error(f"[WS] WebSocket 연결 수락 실패: {e}")
        logger.error(f"[WS] Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"[WS] Traceback: {traceback.format_exc()}")
        return
    
    # JWT 토큰 검증 (연결 후)
    try:
        from app.core.security import verify_token
        
        logger.info(f"[WS] JWT 토큰 검증 시작...")
        payload = verify_token(token)
        
        if not payload:
            logger.error(f"[WS] JWT 토큰 검증 실패: payload가 None")
            await websocket.send_text(
                json.dumps({
                    "error_code": "INVALID_TOKEN",
                    "message": "유효하지 않은 토큰입니다."
                })
            )
            await websocket.close()
            return
        
        user_id = payload.get("sub")
        user_email = payload.get("email")
        user_name = payload.get("name")
        groups = payload.get("groups", [])
        permissions = payload.get("permissions", [])
        
        logger.info(f"[WS] ✅ 토큰 검증 성공!")
        logger.info(f"[WS] 사용자 정보: user_id={user_id}, email={user_email}, name={user_name}")
        logger.info(f"[WS] 권한 정보: groups={groups}, permissions={permissions}")
        
    except Exception as e:
        logger.error(f"[WS] ❌ 토큰 검증 중 예외 발생: {type(e).__name__}: {str(e)}")
        logger.error(f"[WS] 토큰 디버그 정보:")
        logger.error(f"[WS] - 토큰 타입: {type(token)}")
        logger.error(f"[WS] - 토큰 길이: {len(token) if token else 'None'}")
        logger.error(f"[WS] - 첫 10자: {token[:10] if token else 'None'}")
        
        import traceback
        logger.error(f"[WS] 상세 스택 트레이스: {traceback.format_exc()}")
        
        await websocket.send_text(
            json.dumps({
                "error_code": "TOKEN_VERIFICATION_FAILED",
                "message": f"토큰 검증에 실패했습니다: {str(e)}"
            })
        )
        await websocket.close()
        return

    # 데이터베이스 연결 수동 생성
    try:
        from app.database import SessionLocal
        db = SessionLocal()
        logger.info(f"[WS] 데이터베이스 연결 생성 완료")
    except Exception as e:
        logger.error(f"[WS] 데이터베이스 연결 실패: {e}")
        await websocket.send_text(
            json.dumps({
                "error_code": "DATABASE_CONNECTION_FAILED",
                "message": "데이터베이스 연결에 실패했습니다."
            })
        )
        await websocket.close()
        return

    # lora_repo는 base64로 인코딩되어 있으므로 디코딩
    try:
        lora_repo_decoded = base64.b64decode(lora_repo).decode()
    except Exception as e:
        await websocket.send_text(
            json.dumps(
                {
                    "error_code": "LORA_REPO_DECODE_ERROR",
                    "message": f"lora_repo 디코딩 실패: {e}",
                }
            )
        )
        await websocket.close()
        return

    # 데이터베이스 히스토리 서비스 초기화
    chat_message_service = ChatMessageService(db)
    # 세션별 히스토리 초기화
    session_id = f"{lora_repo_decoded}_{group_id}_{influencer_id or 'default'}"
    current_session_id = None  # 현재 세션 ID 초기화

    try:
        # RunPod 서버 상태 확인 (상세 로그 포함)
        logger.info(f"[WS] ========== RunPod 서버 상태 확인 시작 ==========")
        logger.info(f"[WS] Session ID: {session_id}")
        logger.info(f"[WS] Model/LoRA repo: {lora_repo_decoded}")
        logger.info(f"[WS] Group ID: {group_id}")
        logger.info(f"[WS] Influencer ID: {influencer_id}")
        
        # 인플루언서 정보 조회 및 전송
        if influencer_id:
            from app.models.influencer import AIInfluencer
            influencer_info = db.query(AIInfluencer).filter(
                AIInfluencer.influencer_id == influencer_id
            ).first()
            s3_service = get_s3_image_service()
            if influencer_info:
                # 인플루언서 정보를 클라이언트에 전송
                logger.info(f"[WS] 인플루언서 정보 - 이름: {influencer_info.influencer_name}")
                logger.info(f"[WS] 인플루언서 정보 - 설명: {influencer_info.influencer_description}")
                logger.info(f"[WS] 인플루언서 정보 - 이미지 URL: {influencer_info.image_url}")
                
                
                image_url_value = s3_service.generate_presigned_url(
                    influencer_info.image_url, expiration=3600
                )
                logger.info(f"[WS] 인플루언서 이미지 URL 값: {image_url_value}")
                
                await websocket.send_text(json.dumps({
                    "type": "influencer_info",
                    "data": {
                        "name": influencer_info.influencer_name,
                        "description": influencer_info.influencer_description,
                        "image_url": image_url_value  # 명시적으로 값 전달
                    }
                }))
                logger.info(f"[WS] 인플루언서 정보 전송 완료: {influencer_info.influencer_name}")
        
        # 환경변수 확인 (settings 사용)
        from app.core.config import settings
        runpod_api_key = settings.RUNPOD_API_KEY
        logger.info(f"[WS] RUNPOD_API_KEY 설정됨: {'Yes' if runpod_api_key else 'No'}")
        if runpod_api_key:
            logger.info(f"[WS] RUNPOD_API_KEY 길이: {len(runpod_api_key)}자")
            logger.info(f"[WS] RUNPOD_API_KEY 앞 10자: {runpod_api_key[:10]}...")
        
        # vLLM 매니저 정보 확인
        vllm_manager = get_vllm_manager()
        logger.info(f"[WS] vLLM Manager 생성됨: {type(vllm_manager)}")
        
        health_status = await vllm_manager.health_check()
        logger.info(f"[WS] vLLM health check 결과: {health_status}")
        
        if not health_status:
            logger.error(f"[WS] ❌ RunPod 서버 연결 실패")
            await websocket.send_text(
                json.dumps(
                    {
                        "error_code": "RUNPOD_SERVER_UNAVAILABLE", 
                        "message": "RunPod 서버에 연결할 수 없습니다. API 키를 확인해주세요.",
                    }
                )
            )
            await websocket.close()
            return
        
        logger.info(f"[WS] ✅ RunPod 서버 상태 확인 완료")
        
        logger.info(f"[WS] ✅ vLLM Manager 준비됨 (RunPod Serverless)")

        logger.info(
            f"[WS] RunPod WebSocket 연결 시작: lora_repo={lora_repo_decoded}, group_id={group_id}, session_id={session_id}"
        )

        # HF 토큰 가져오기 (필요시)
        hf_token = await _get_hf_token_by_group(group_id, db)

        # 인플루언서 정보 가져오기
        influencer = None
        hf_repo = None
        if influencer_id:
            from app.models.influencer import AIInfluencer

            influencer = (
                db.query(AIInfluencer)
                .filter(AIInfluencer.influencer_id == influencer_id)
                .first()
            )
            if influencer and influencer.system_prompt is not None:
                system_prompt = str(influencer.system_prompt)
                logger.info(
                    f"[WS] ✅ 저장된 시스템 프롬프트 사용: {influencer.influencer_name}"
                )
            else:
                logger.info(
                    f"[WS] ⚠️ 저장된 시스템 프롬프트가 없어 기본 시스템 프롬프트 사용"
                )
            
            # HuggingFace repository 경로 가져오기
            if influencer and influencer.influencer_model_repo:
                hf_repo = str(influencer.influencer_model_repo)
                logger.info(f"[WS] 🔧 HF Repository: {hf_repo}")

        # RunPod는 어댑터 사전 로드가 필요하지 않음 (요청 시 지정)
        logger.info(f"[WS] RunPod LoRA 어댑터 준비: {lora_repo_decoded}")

        # WebSocket 프록시 모드
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"[WS] 메시지 수신: {data[:100]}...")

                # 메시지 파싱 (JSON 또는 일반 텍스트)
                logger.info(f"[WS] 메시지 타입 분석 시작")
                try:
                    message_data = json.loads(data)
                    message_type = message_data.get("type", "chat")
                    user_message = message_data.get("message", data)
                    logger.info(f"[WS] JSON 메시지 파싱 성공: type={message_type}")
                    
                    # 히스토리 관련 명령 처리
                    if message_type == "get_history":
                        # 현재 세션의 히스토리 조회
                        if current_session_id:
                            session_messages = (
                                chat_message_service.get_session_messages(
                                    current_session_id
                                )
                            )

                            # 사용자/AI 메시지를 쌍으로 구성하여 히스토리 생성 (message_type 기반)
                            history_data = []
                            current_user_msg = None
                            current_timestamp = None

                            for msg in session_messages:
                                if msg.message_type == "user":
                                    current_user_msg = msg.message_content
                                    current_timestamp = (
                                        msg.created_at.isoformat()
                                        if msg.created_at
                                        else None
                                    )
                                elif msg.message_type == "ai" and current_user_msg:
                                    ai_response = msg.message_content
                                    history_data.append(
                                        {
                                            "query": current_user_msg,
                                            "response": ai_response,
                                            "timestamp": current_timestamp,
                                            "source": "session",
                                            "session_id": msg.session_id,
                                        }
                                    )
                                    current_user_msg = None
                                    current_timestamp = None

                            await websocket.send_text(
                                json.dumps({"type": "history", "data": history_data})
                            )
                        else:
                            await websocket.send_text(
                                json.dumps({"type": "history", "data": []})
                            )
                        continue
                    elif message_type == "clear_history":
                        # 현재 세션 종료 (새 세션 시작)
                        if current_session_id:
                            chat_message_service.end_session(current_session_id)
                            logger.info(
                                f"[WS] 세션 종료 (히스토리 초기화): session_id={current_session_id}"
                            )

                        # 새 세션 생성
                        current_session_id = chat_message_service.create_session(
                            influencer_id or "default"
                        )
                        logger.info(
                            f"[WS] 새 세션 생성 (히스토리 초기화): session_id={current_session_id}"
                        )

                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "history_cleared",
                                    "message": "채팅 히스토리가 초기화되었습니다.",
                                }
                            )
                        )
                        continue

                except json.JSONDecodeError:
                    # 일반 텍스트 메시지로 처리
                    message_type = "chat"
                    user_message = data
                    logger.info(f"[WS] 일반 텍스트 메시지로 처리")

                # chat 메시지 처리 (일반 대화)
                if message_type == "chat":
                    # 세션 관리
                    if current_session_id is None:
                        # 새 세션 생성
                        current_session_id = chat_message_service.create_session(
                            influencer_id or "default"
                        )
                    logger.info(f"[WS] 새 세션 생성: session_id={current_session_id}")

                # 의도 기반 히스토리 컨텍스트 추가
                enhanced_message = user_message

                # 현재 세션의 이전 메시지들 조회
                session_messages = chat_message_service.get_session_messages(
                    current_session_id
                )

                if session_messages:
                    # 빈 메시지 제외하고 유효한 메시지만 필터링
                    valid_messages = [
                        msg for msg in session_messages if msg.message_content.strip()
                    ]

                    if len(valid_messages) >= 2:  # 최소 1턴(사용자+AI) 이상
                        try:
                            # 사용자/AI 메시지를 쌍으로 구성 (message_type 기반)
                            conversation_pairs = []
                            current_user_msg = None

                            for msg in valid_messages:
                                if msg.message_type == "user":
                                    current_user_msg = msg.message_content
                                elif msg.message_type == "ai" and current_user_msg:
                                    ai_response = msg.message_content
                                    conversation_pairs.append(
                                        {
                                            "query": current_user_msg,
                                            "response": ai_response,
                                        }
                                    )
                                    current_user_msg = None  # 다음 쌍을 위해 초기화

                            if conversation_pairs:
                                # 의도 분석을 통한 히스토리 프롬프트 생성
                                intent_based_context = await analyze_user_intent(
                                    user_message, conversation_pairs
                                )

                                if (
                                    intent_based_context
                                    and len(intent_based_context) > 10
                                ):
                                    enhanced_message = f"{intent_based_context}\n\n현재 질문: {user_message}"
                                    logger.info(
                                        f"[WS] 의도 기반 히스토리 사용 (세션: {current_session_id}, 대화쌍: {len(conversation_pairs)}개)"
                                    )
                                else:
                                    # 의도 분석 실패 시 최근 대화만 사용
                                    latest_pair = conversation_pairs[-1]
                                    enhanced_message = f"이전 질문: {latest_pair['query'][:50]}...\n\n현재 질문: {user_message}"
                                    logger.info(
                                        f"[WS] 최근 대화 사용 (세션: {current_session_id})"
                                    )
                        except Exception as e:
                            logger.warning(
                                f"[WS] 히스토리 처리 실패, 요약 없이 진행: {e}"
                            )
                            enhanced_message = user_message

                # 메시지 수신 즉시 활동 표시 (RAG/MCP/Modal 콜드스타트 동안 프론트 타임아웃 방지)
                await websocket.send_text(
                    json.dumps({"type": "thinking", "message": "생각중..."}, ensure_ascii=False)
                )

                # === RAG: 문서 우선, 없으면 MCP 검색으로 보완 ===
                rag_context, rag_sources = await build_rag_context(
                    influencer_id or "", user_message
                )
                context_kind = "doc"
                if not rag_context:
                    # 문서에 근거가 없으면 할당된 MCP(웹검색 등) 서버로 보완
                    from app.services.mcp_search_service import search_via_assigned_mcp

                    mcp_ctx, mcp_sources = await search_via_assigned_mcp(
                        influencer_id or "", user_message, db
                    )
                    if mcp_ctx:
                        rag_context, rag_sources = mcp_ctx, mcp_sources
                        context_kind = "search"
                rag_suffix = (
                    f"\n\n다음 {'참고문서' if context_kind == 'doc' else '웹 검색 결과'}의 "
                    f"사실에 근거해 답하라:\n{rag_context}"
                    if rag_context
                    else ""
                )
                if rag_context:
                    await websocket.send_text(
                        json.dumps({"type": "sources", "data": rag_sources}, ensure_ascii=False)
                    )

                # MCP 처리 로직 추가
                mcp_result = None
                tools_used = []
                try:
                    # MCP 처리를 위한 API 엔드포인트 모듈 가져오기
                    from app.api.v1.endpoints.mcp import process_with_mcp_tools
                    from app.services.mcp_server_service import MCPServerService
                    
                    logger.info(f"[WS] MCP 처리 시작: {user_message[:50]}...")
                    
                    # 인플루언서에게 할당된 MCP 서버 목록 가져오기
                    mcp_service = MCPServerService(db)
                    assigned_servers = mcp_service.get_influencer_mcp_servers(influencer_id or "")
                    selected_servers = [server.mcp_name for server in assigned_servers]
                    
                    if selected_servers:
                        logger.info(f"[WS] 할당된 MCP 서버: {selected_servers}")
                        # MCP 도구를 사용하여 메시지 처리
                        mcp_response, tools_used = await process_with_mcp_tools(user_message, selected_servers)
                        
                        if mcp_response:
                            mcp_result = mcp_response
                            logger.info(f"[WS] ✅ MCP 처리 성공, 사용된 도구: {tools_used}")
                            # MCP 사용 정보를 클라이언트에 전송
                            await websocket.send_text(
                                json.dumps({
                                    "type": "mcp_used",
                                    "tools": tools_used,
                                    "message": "MCP 도구를 사용하여 답변을 생성 중입니다."
                                })
                            )
                        else:
                            logger.info(f"[WS] MCP 처리 없음 - 일반 LLM으로 처리")
                    else:
                        logger.info(f"[WS] 할당된 MCP 서버가 없음 - 일반 LLM으로 처리")
                except Exception as e:
                    logger.error(f"[WS] MCP 처리 중 오류: {e}")
                    import traceback
                    logger.error(f"[WS] MCP 오류 상세: {traceback.format_exc()}")
                
                # 최종 메시지 구성
                if mcp_result:
                    payload = {
                        "input": {
                            "hf_token": hf_token,
                            "hf_repo": hf_repo,
                            "system_message": system_prompt + rag_suffix,
                            "prompt": mcp_result,
                            "temperature": 1,
                            "max_tokens": 2048
                        }
                    }
                    
                    # runsync로 전체 응답 받기
                    result = await vllm_manager.runsync(payload)
                    
                    # 응답 처리
                    full_response = result
                    
                    # 타이핑 시작 상태 전송 
                    await websocket.send_text(
                        json.dumps({"type": "typing", "message": "답변 입력중..."}, ensure_ascii=False)
                    )
                    
                    # 받은 응답을 단어 단위로 분할해서 스트리밍
                    words = full_response.split(' ')
                    chunk_size = 2  # 2단어씩 전송
                    token_count = 0
                    
                    for i in range(0, len(words), chunk_size):
                        chunk_words = words[i:i + chunk_size]
                        chunk_text = ' '.join(chunk_words)
                        
                        # 마지막 청크가 아니면 공백 추가
                        if i + chunk_size < len(words):
                            chunk_text += ' '
                        
                        # 클라이언트에 청크 전송
                        await websocket.send_text(
                            json.dumps({"type": "token", "content": chunk_text}, ensure_ascii=False)
                        )
                        token_count += 1
                        
                        # 스트리밍 효과를 위한 딜레이
                        await asyncio.sleep(0.1)

                    # 스트리밍 완료 신호
                    await websocket.send_text(
                        json.dumps({"type": "complete", "content": ""}, ensure_ascii=False)
                    )
                    
                    # 사용자 메시지와 AI 응답 저장
                    if current_session_id:
                        try:
                            # 사용자 메시지 저장
                            chat_message_service.add_message_to_session(
                                session_id=current_session_id,
                                influencer_id=influencer_id or "default",
                                message_content=user_message,
                                message_type="user",
                            )
                            # AI 응답 저장  
                            chat_message_service.add_message_to_session(
                                session_id=current_session_id,
                                influencer_id=influencer_id or "default",
                                message_content=full_response,
                                message_type="ai",
                            )
                            logger.info(
                                f"[WS] MCP 응답 세션 저장 완료: session_id={current_session_id}"
                            )
                        except Exception as e:
                            logger.error(f"[WS] MCP 응답 세션 저장 실패: {e}")
                    
                    # TTS 생성 (MCP 응답에 대해서도)
                    if full_response.strip():
                        asyncio.create_task(
                            _process_tts_async(
                                websocket, 
                                full_response, 
                                influencer_id if influencer_id else "default"
                            )
                        )
                    
                    # MCP 응답은 여기서 처리 완료이므로 continue
                    continue
                else:
                    # MCP 결과가 없으면 기존 로직대로 enhanced_message 사용
                    final_prompt = enhanced_message
                
                system_prompt = (
                    str(influencer.system_prompt)
                    if influencer and influencer.system_prompt
                    else "당신은 도움이 되는 AI 어시스턴트입니다."
                )

                # 필수 파라미터 검증
                if not hf_token:
                    logger.error("[WS] HF 토큰이 없습니다")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": "HuggingFace 토큰이 설정되지 않았습니다. 관리자에게 문의하세요."
                    }))
                    continue
                
                if not hf_repo:
                    logger.error("[WS] HF repository가 없습니다")
                    await websocket.send_text(json.dumps({
                        "type": "error", 
                        "content": "모델 저장소가 설정되지 않았습니다. 관리자에게 문의하세요."
                    }))
                    continue

                # 생각중 상태 전송
                await websocket.send_text(
                    json.dumps({"type": "thinking", "message": "생각중..."}, ensure_ascii=False)
                )
                
                # runsync로 응답 생성 후 클라이언트에 스트리밍으로 전달
                payload = {
                    "input": {
                        "hf_token": hf_token,
                        "hf_repo": hf_repo,
                        "system_message": system_prompt + rag_suffix,
                        "prompt": final_prompt,
                        "temperature": 1,
                        "max_tokens": 2048
                    }
                }
                
                # runsync로 전체 응답 받기
                result = await vllm_manager.runsync(payload)
                
                # 응답 처리
                full_response = result
                
                
                # 타이핑 시작 상태 전송 
                await websocket.send_text(
                    json.dumps({"type": "typing", "message": "답변 입력중..."}, ensure_ascii=False)
                )
                
                # 받은 응답을 단어 단위로 분할해서 스트리밍
                words = full_response.split(' ')
                chunk_size = 2  # 2단어씩 전송
                token_count = 0
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    # 마지막 청크가 아니면 공백 추가
                    if i + chunk_size < len(words):
                        chunk_text += ' '
                    
                    # 클라이언트에 청크 전송
                    await websocket.send_text(
                        json.dumps({"type": "token", "content": chunk_text}, ensure_ascii=False)
                    )
                    token_count += 1
                    
                    # 스트리밍 효과를 위한 딜레이
                    await asyncio.sleep(0.1)

                # 스트리밍 완료 신호
                await websocket.send_text(
                    json.dumps({"type": "complete", "content": ""}, ensure_ascii=False)
                )

                # RAG 사용 시 대화 이력 저장 (실패해도 채팅에는 영향 없음)
                if rag_context:
                    try:
                        from app.models.rag import RAGChatHistory

                        db.add(
                            RAGChatHistory(
                                user_id=(influencer_id or "")[:36],
                                group_id=group_id or 0,
                                question=user_message,
                                answer=full_response,
                                sources=rag_sources,
                            )
                        )
                        db.commit()
                    except Exception as e:
                        logger.warning(f"[WS] rag_chat_history 저장 실패: {e}")
                        db.rollback()

                # TTS 생성 시작 (비동기로 처리)
                if full_response.strip():
                    # 비동기 태스크로 TTS 처리
                    asyncio.create_task(
                        _process_tts_async(
                            websocket, 
                            full_response, 
                            influencer_id if influencer_id else "default"
                        )
                    )

                    # 세션에 대화 저장 (메시지 타입 구분)
                    if full_response.strip():
                        try:
                            # 사용자 메시지 저장
                            chat_message_service.add_message_to_session(
                                session_id=current_session_id,
                                influencer_id=influencer_id or "default",
                                message_content=user_message,
                                message_type="user",
                            )

                            # AI 응답 저장
                            chat_message_service.add_message_to_session(
                                session_id=current_session_id,
                                influencer_id=influencer_id or "default",
                                message_content=full_response,
                                message_type="ai",
                            )
                            logger.info(
                                f"[WS] 세션에 대화 저장 완료: session_id={current_session_id}"
                            )
                        except Exception as e:
                            logger.error(f"[WS] 세션 저장 실패: {e}")
                        
                    logger.info(
                        f"[WS] RunPod 스트리밍 응답 전송 완료 (토큰 수: {token_count})"
                    )

            except WebSocketDisconnect:
                # 세션 종료
                if current_session_id:
                    chat_message_service.end_session(current_session_id)
                    logger.info(f"[WS] 세션 종료: session_id={current_session_id}")

                logger.info(f"[WS] WebSocket 연결 종료: lora_repo={lora_repo_decoded}")
                break
            except Exception as e:
                logger.error(f"[WS] WebSocket 처리 중 오류: {e}")
                logger.error(f"[WS] Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"[WS] Full traceback: {traceback.format_exc()}")
                logger.error(f"[WS] Current message: {data[:200]}..." if len(data) > 200 else f"[WS] Current message: {data}")
                await websocket.send_text(
                    json.dumps({"error_code": "WEBSOCKET_ERROR", "message": str(e)})
                )
                break

    except Exception as e:
        logger.error(f"[WS] ========== WebSocket 연결 처리 중 심각한 오류 ==========")
        logger.error(f"[WS] Error: {e}")
        logger.error(f"[WS] Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"[WS] Full traceback:\n{traceback.format_exc()}")
        logger.error(f"[WS] Session ID: {session_id if 'session_id' in locals() else 'Not created'}")
        logger.error(f"[WS] ====================================================")
        try:
            await websocket.send_text(
                json.dumps({"error_code": "CONNECTION_ERROR", "message": str(e)})
            )
        except:
            logger.error(f"[WS] Failed to send error message to client")
    finally:
        # 데이터베이스 연결 정리
        try:
            if 'db' in locals():
                db.close()
                logger.info(f"[WS] 데이터베이스 연결 정리 완료")
        except Exception as e:
            logger.error(f"[WS] 데이터베이스 연결 정리 실패: {e}")


async def build_rag_context(influencer_id: str, query: str) -> tuple[str, list]:
    """인플루언서의 벡터화 문서에서 query 관련 컨텍스트를 검색.

    반환: (context_str, sources). Chroma/임베딩 실패 또는 0건이면 ('', []) 로
    graceful 폴백하여 챗봇이 절대 중단되지 않게 한다.
    """
    if not influencer_id or not query:
        return "", []
    try:
        from app.services.embedding_client import embed_texts
        from app.services.rag_vector_store import get_vector_store

        qvec = (await embed_texts([query]))[0]
        hits = get_vector_store().search(
            qvec, influencer_id, settings.RAG_TOP_K, settings.RAG_SCORE_THRESHOLD
        )
        if not hits:
            return "", []
        ctx = "\n\n".join(f"[참고문서] {h['text']}" for h in hits)
        sources = [
            {"text": h["text"][:200], "score": round(h.get("score", 0.0), 3), "source": h.get("source")}
            for h in hits
        ]
        logger.info(f"[WS] RAG 컨텍스트 {len(hits)}건 주입 (influencer={influencer_id})")
        return ctx, sources
    except Exception as e:
        logger.warning(f"[WS] RAG 컨텍스트 생성 실패(폴백): {e}")
        return "", []


async def _get_hf_token_by_group(group_id: int, db: Session) -> str | None:
    """그룹 ID로 HF 토큰 가져오기"""
    try:
        hf_token_manage = (
            db.query(HFTokenManage)
            .filter(HFTokenManage.group_id == group_id)
            .order_by(HFTokenManage.created_at.desc())
            .first()
        )

        if hf_token_manage:
            return decrypt_sensitive_data(str(hf_token_manage.hf_token_value))
        else:
            logger.warning(f"그룹 {group_id}에 등록된 HF 토큰이 없습니다.")
            return None

    except Exception as e:
        logger.error(f"HF 토큰 조회 실패: {e}")
        return None


# OpenAI API 클라이언트 추가
async def get_openai_client():
    """OpenAI API 클라이언트 생성"""
    return httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        timeout=30.0,
    )


async def analyze_user_intent(user_message: str, history: List[Dict]) -> str:
    """사용자 질문의 의도를 분석하여 히스토리 프롬프트 생성"""
    if not history:
        return ""

    try:
        # 최근 3개 대화만 사용
        recent_history = history[-3:] if len(history) > 3 else history

        # 히스토리 텍스트 구성
        history_text = ""
        for i, chat in enumerate(recent_history, 1):
            history_text += (
                f"이전 질문{i}: {chat['query']}\n이전 답변{i}: {chat['response']}\n\n"
            )

        # OpenAI API 호출 (의도 분석용 시스템 프롬프트)
        client = await get_openai_client()

        system_prompt = """당신은 사용자의 질문 의도를 분석하는 전문가입니다.

주어진 이전 대화 히스토리와 현재 질문을 바탕으로, 현재 질문과 관련된 이전 대화만을 선별하여 컨텍스트를 제공하세요.

규칙:
1. 현재 질문과 직접적으로 관련된 이전 대화만 포함
2. 관련 없는 대화는 제외
3. 간결하고 명확하게 요약
4. 현재 질문에 도움이 되는 정보만 포함

형식: "이전에 [관련 내용]에 대해 이야기했는데, [현재 질문과의 연관성]"
예시: "이전에 날씨 API 사용법에 대해 이야기했는데, 이번에는 다른 API 사용법을 문의하시는군요."

현재 질문: {user_message}

이전 대화:
{history_text}

분석 결과:"""

        response = await client.post(
            "/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt.format(
                            user_message=user_message, history_text=history_text
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"현재 질문: {user_message}\n\n이전 대화:\n{history_text}",
                    },
                ],
                "max_tokens": 150,
                "temperature": 0.3,
            },
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            logger.info(f"✅ 의도 분석 완료: {len(content)}자")
            return content
        else:
            logger.warning(f"⚠️ 의도 분석 실패: {response.status_code}")
            return ""

    except Exception as e:
        logger.error(f"❌ 의도 분석 중 오류: {e}")
        return ""


async def summarize_chat_history(history: List[Dict], max_tokens: int = 80) -> str:
    """OpenAI를 사용해서 채팅 히스토리를 요약 (완전한 요약 보장)"""
    if not history:
        return ""

    try:
        # 히스토리를 텍스트로 변환
        history_text = ""
        for i, chat in enumerate(history[-5:], 1):  # 최근 5개 대화만
            history_text += f"Q{i}: {chat['query']}\nA{i}: {chat['response']}\n\n"

        # OpenAI API 호출 (개선된 시스템 프롬프트)
        openai_client = await get_openai_client()
        response = await openai_client.post(
            "/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "다음 대화를 80토큰 이하로 완전히 요약하세요. 핵심 정보만 포함하고, 문장을 중간에 끊지 마세요. 반드시 완전한 문장으로 마무리하세요.",
                    },
                    {
                        "role": "user",
                        "content": f"다음 대화를 요약해주세요:\n\n{history_text}",
                    },
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,  # 더 일관된 요약을 위해 낮춤
                "stop": None,  # 중간에 끊기지 않도록 stop 토큰 제거
            },
        )

        if response.status_code == 200:
            result = response.json()
            summary = result["choices"][0]["message"]["content"].strip()
            logger.info(f"[WS] 히스토리 요약 완료: {len(summary)}자")
            return summary
        else:
            logger.warning(f"[WS] OpenAI 요약 실패: {response.status_code}")
            return ""

    except Exception as e:
        logger.error(f"[WS] 히스토리 요약 중 오류: {e}")
        return ""


@router.post("/load_model")
async def model_load(req: ModelLoadRequest, db: Session = Depends(get_db)):
    """모델 로드 (RunPod 서버리스 사용)"""
    try:
        # HF 토큰 가져오기
        hf_token = await _get_hf_token_by_group(req.group_id, db)
        if not hf_token:
            raise HTTPException(status_code=400, detail="HF 토큰이 없습니다.")

        # vLLM 매니저 가져오기
        vllm_manager = get_vllm_manager()
        
        # RunPod 서버 상태 확인
        if not await vllm_manager.health_check():
            raise HTTPException(
                status_code=503, detail="RunPod 서버에 연결할 수 없습니다."
            )

        # RunPod 서버리스는 어댑터 사전 로드가 필요하지 않음
        # 요청 시 동적으로 로드되므로 성공으로 반환
        logger.info(f"[MODEL LOAD API] RunPod 어댑터 준비 완료: {req.lora_repo}")
        return {
            "success": True,
            "message": "RunPod 서버에서 모델이 준비되었습니다. 요청 시 동적으로 로드됩니다.",
            "server_type": "runpod",
            "adapter_repo": req.lora_repo
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MODEL LOAD API] 모델 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
