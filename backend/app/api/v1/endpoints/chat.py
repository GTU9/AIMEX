from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import logging
import json
import uuid
from pydantic import BaseModel
from app.database import get_db
from app.models.influencer import (
    AIInfluencer,
    InfluencerAPI,
    APICallAggregation,
)
from app.models.chat_message import ChatMessage
from app.models.user import User
from app.core.security import get_current_user
from app.utils.timezone_utils import get_current_kst
from app.core.security import get_current_user, get_current_user_by_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


# 챗봇 API에 대한 CORS 설정
@router.options("/chatbot")
async def chatbot_options():
    """챗봇 API CORS preflight 요청 처리"""
    return {"message": "OK"}

@router.options("/chatbot/user")
async def chatbot_user_options():
    """사용자 챗봇 API CORS preflight 요청 처리"""
    return {"message": "OK"}


# API 키로 접근 가능한 챗봇 요청 스키마
class ChatbotRequest(BaseModel):
    message: str
    session_id: str | None = None

# JWT 토큰으로 접근 가능한 챗봇 요청 스키마
class ChatbotWithInfluencerRequest(BaseModel):
    message: str
    influencer_id: str
    session_id: str | None = None


class ChatbotResponse(BaseModel):
    response: str
    session_id: str
    influencer_name: str


# 채팅 메시지 스키마
class ChatMessageSchema(BaseModel):
    session_id: str
    influencer_id: str
    message_content: str
    created_at: str
    end_at: str | None = None

    class Config:
        from_attributes = True


class ChatMessageCreate(BaseModel):
    influencer_id: str
    message_content: str
    message_type: str = "user"  # user 또는 ai
    end_at: str | None = None


# 비스트리밍 챗봇 엔드포인트 (기존)
@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot_chat(
    request: ChatbotRequest,
    influencer: AIInfluencer = Depends(get_current_user_by_api_key),
    db: Session = Depends(get_db),
):
    """
    API 키로 접근 가능한 비스트리밍 챗봇 엔드포인트
    인플루언서와 대화할 수 있습니다. (완전한 응답을 한 번에 반환)
    """
    try:
        # API 사용량 추적
        await track_api_usage(db, str(influencer.influencer_id))

        # RunPod 서비스 호출
        try:
            from app.services.runpod_manager import get_vllm_manager
            
            # vLLM 매니저 가져오기
            vllm_manager = get_vllm_manager()

            # RunPod 서버 상태 확인
            if not await vllm_manager.health_check():
                logger.warning("RunPod 서버에 연결할 수 없어 기본 응답을 사용합니다.")
                response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. '{request.message}'에 대한 답변을 드리겠습니다."
            else:
                # 시스템 프롬프트 구성
                system_message = (
                    str(influencer.system_prompt)
                    if influencer.system_prompt is not None
                    else f"당신은 {influencer.influencer_name}입니다. 도움이 되는 답변을 해주세요."
                )

                # RunPod 서버에서 응답 생성
                lora_adapter = None
                hf_repo = None
                hf_token = None
                
                logger.info(f"🔍 Influencer 정보: id={influencer.influencer_id}, model_repo={influencer.influencer_model_repo}")
                
                if influencer.influencer_id:
                    # LoRA 어댑터 이름 설정 (인플루언서 ID 사용)
                    lora_adapter = str(influencer.influencer_id)
                    
                    if influencer.influencer_model_repo:
                        # DB에 저장된 HF 레포지토리 경로 사용
                        hf_repo = str(influencer.influencer_model_repo)
                        logger.info(f"🔧 LoRA 어댑터 사용: {lora_adapter}, HF repo: {hf_repo}")
                    else:
                        # model_repo가 없으면 기본 경로 패턴 사용 (임시)
                        # 예: eb4f7078-e069-4e05-845f-6b052ef8739c -> username/model-eb4f7078
                        # 실제로는 데이터베이스에 정확한 HF repo 경로가 있어야 함
                        logger.warning(f"⚠️ Influencer model_repo가 없음: id={influencer.influencer_id}")
                        logger.warning(f"⚠️ 데이터베이스에 HuggingFace repository 경로를 설정해야 합니다!")
                        # HF repo 없이는 작동하지 않으므로 None으로 설정
                        lora_adapter = None
                
                # HF 토큰 가져오기
                if hf_repo:
                    try:
                        from app.services.hf_token_resolver import get_token_for_influencer
                        hf_token, hf_username = await get_token_for_influencer(influencer, db)
                        if hf_token:
                            logger.info(f"🔑 HF 토큰 사용 (user: {hf_username})")
                    except Exception as e:
                        logger.warning(f"⚠️ HF 토큰 가져오기 실패: {e}")
                
                # 필수 파라미터 검증
                if not hf_token or not hf_repo:
                    logger.error(f"필수 파라미터 누락 - hf_token: {'있음' if hf_token else '없음'}, hf_repo: {'있음' if hf_repo else '없음'}")
                    response_text = f"모델 설정이 완료되지 않았습니다. 관리자에게 문의하세요."
                else:
                    # RunPod 텍스트 생성 요청 (새로운 방식)
                    payload = {
                        "input": {
                            "hf_token": hf_token,
                            "hf_repo": hf_repo,
                            "system_message": system_message,
                            "prompt": request.message,
                            "temperature": 1,
                            "max_tokens": 2048
                        }
                    }
                    
                    result = await vllm_manager.runsync(payload)
                
                # 응답 전체 로깅
                    logger.info(f"🔍 RunPod 응답 전체: {json.dumps(result, indent=2, ensure_ascii=False)}")
                    
                    # RunPod 응답 처리 (새로운 형식)
                    if result.get("status") == "completed":
                        # output 내의 generated_text 확인
                        output = result.get("output", {})
                        response_text = output.get("generated_text", "")
                        
                        if response_text:
                            logger.info(f"✅ 생성된 텍스트: {response_text[:100]}...")
                        else:
                            logger.warning(f"⚠️ 응답에 generated_text가 없음: {result}")
                            response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. 응답 생성 중 문제가 발생했습니다."
                    elif result.get("status") == "failed":
                        # 실패한 경우
                        logger.error(f"❌ RunPod 요청 실패: {result.get('error', 'Unknown error')}")
                        response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. '{request.message}'에 대한 답변을 드리겠습니다."
                    else:
                        # 예상하지 못한 응답 형식
                        logger.warning(f"⚠️ 예상하지 못한 RunPod 응답 형식: {result}")
                        response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. '{request.message}'에 대한 답변을 드리겠습니다."

                logger.info(f"✅ RunPod 응답 생성 성공: {influencer.influencer_name}")

        except Exception as e:
            logger.error(f"❌ RunPod 응답 생성 실패: {e}")
            # RunPod 실패 시 기본 응답 사용
            response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. '{request.message}'에 대한 답변을 드리겠습니다."

        # 세션 ID 생성 (실제로는 더 복잡한 로직 필요)
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"

        return ChatbotResponse(
            response=response_text,
            session_id=session_id,
            influencer_name=str(influencer.influencer_name),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot error: {str(e)}",
        )


@router.post("/chatbot/user", response_model=ChatbotResponse)
async def chatbot_for_user(
    request: ChatbotWithInfluencerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    JWT 토큰으로 접근 가능한 챗봇 엔드포인트
    사용자가 influencer_id를 지정하여 인플루언서와 대화할 수 있습니다.
    """
    try:
        # 인플루언서 조회
        influencer = (
            db.query(AIInfluencer)
            .filter(AIInfluencer.influencer_id == request.influencer_id)
            .first()
        )
        
        if not influencer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Influencer not found"
            )
        
        # 사용자가 인플루언서에 접근할 수 있는지 확인 (같은 그룹)
        # current_user 는 get_current_user 가 반환하는 JWT payload dict 이므로
        # 사용자의 소속 그룹(들)을 DB 에서 조회해 비교한다.
        from app.models.user import User as _User
        _uid = current_user.get("sub") if isinstance(current_user, dict) else getattr(current_user, "user_id", None)
        _db_user = db.query(_User).filter(_User.user_id == _uid).first()
        _user_group_ids = [t.group_id for t in _db_user.teams] if _db_user else []
        if influencer.group_id not in _user_group_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to chat with this influencer"
            )
        
        # 챗봇 옵션 확인
        if not influencer.chatbot_option:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This influencer's chatbot is not enabled"
            )
        
        # 학습 상태 확인
        if influencer.learning_status != 1:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Influencer is not ready for chat"
            )
        
        logger.info(f"🔍 사용자 챗봇 요청 - Influencer: id={influencer.influencer_id}, name={influencer.influencer_name}, model_repo={influencer.influencer_model_repo}")
        
        # RunPod 서비스 호출
        try:
            from app.services.runpod_manager import get_vllm_manager
            
            # vLLM 매니저 가져오기
            vllm_manager = get_vllm_manager()

            # RunPod 서버 상태 확인
            if not await vllm_manager.health_check():
                logger.warning("RunPod 서버에 연결할 수 없어 기본 응답을 사용합니다.")
                response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. '{request.message}'에 대한 답변을 드리겠습니다."
            else:
                # 시스템 프롬프트 구성
                system_message = (
                    str(influencer.system_prompt)
                    if influencer.system_prompt is not None
                    else f"당신은 {influencer.influencer_name}입니다. 도움이 되는 답변을 해주세요."
                )

                # RunPod 서버에서 응답 생성
                lora_adapter = None
                hf_repo = None
                hf_token = None
                
                if influencer.influencer_id:
                    # LoRA 어댑터 이름 설정 (인플루언서 ID 사용)
                    lora_adapter = str(influencer.influencer_id)
                    
                    if influencer.influencer_model_repo:
                        # DB에 저장된 HF 레포지토리 경로 사용
                        hf_repo = str(influencer.influencer_model_repo)
                        logger.info(f"🔧 LoRA 어댑터 사용: {lora_adapter}, HF repo: {hf_repo}")
                    else:
                        logger.warning(f"⚠️ Influencer model_repo가 없음: id={influencer.influencer_id}")
                        logger.warning(f"⚠️ 데이터베이스에 HuggingFace repository 경로를 설정해야 합니다!")
                        # HF repo 없이는 작동하지 않으므로 None으로 설정
                        lora_adapter = None
                
                # HF 토큰 가져오기
                if hf_repo:
                    try:
                        from app.services.hf_token_resolver import get_token_for_influencer
                        hf_token, hf_username = await get_token_for_influencer(influencer, db)
                        if hf_token:
                            logger.info(f"🔑 HF 토큰 사용 (user: {hf_username})")
                    except Exception as e:
                        logger.warning(f"⚠️ HF 토큰 가져오기 실패: {e}")
                
                # 필수 파라미터 검증
                if not hf_token or not hf_repo:
                    logger.error(f"필수 파라미터 누락 - hf_token: {'있음' if hf_token else '없음'}, hf_repo: {'있음' if hf_repo else '없음'}")
                    response_text = f"모델 설정이 완료되지 않았습니다. 관리자에게 문의하세요."
                else:
                    # RunPod 텍스트 생성 요청 (새로운 방식)
                    payload = {
                        "input": {
                            "hf_token": hf_token,
                            "hf_repo": hf_repo,
                            "system_message": system_message,
                            "prompt": request.message,
                            "temperature": 1,
                            "max_tokens": 2048
                        }
                    }
                    
                    result = await vllm_manager.runsync(payload)
                    logger.info(f"🔍 [User] 추론 응답(raw): {str(result)[:300]}")

                    # runsync 는 generated_text 문자열(권장) 또는 dict 를 반환할 수 있음
                    if isinstance(result, str):
                        response_text = result.strip()
                    elif isinstance(result, dict):
                        _out = result.get("output")
                        response_text = (
                            result.get("generated_text")
                            or (_out.get("generated_text") if isinstance(_out, dict) else None)
                            or ""
                        ).strip()
                    else:
                        response_text = ""

                    if response_text:
                        logger.info(f"✅ 생성된 텍스트: {response_text[:100]}...")
                    else:
                        logger.warning(f"⚠️ 빈/예상외 추론 응답: {result}")
                        response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. 응답 생성 중 문제가 발생했습니다."

                logger.info(f"✅ RunPod 응답 생성 성공: {influencer.influencer_name}")

        except Exception as e:
            logger.error(f"❌ RunPod 응답 생성 실패: {e}")
            response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. '{request.message}'에 대한 답변을 드리겠습니다."

        # 세션 ID 생성
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"

        return ChatbotResponse(
            response=response_text,
            session_id=session_id,
            influencer_name=str(influencer.influencer_name),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 챗봇 처리 중 오류: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot error: {str(e)}",
        )


# 스트리밍 챗봇 엔드포인트 (새로 추가)
@router.post("/chatbot/stream")
async def chatbot_chat_stream(
    request: ChatbotRequest,
    influencer: AIInfluencer = Depends(get_current_user_by_api_key),
    db: Session = Depends(get_db),
):
    """
    API 키로 접근 가능한 스트리밍 챗봇 엔드포인트
    인플루언서와 대화할 수 있습니다. (실시간으로 토큰을 스트리밍)
    """
    try:
        # API 사용량 추적
        await track_api_usage(db, str(influencer.influencer_id))

        async def generate_stream():
            try:
                # RunPod 서비스 호출
                from app.services.runpod_manager import get_vllm_manager
                
                # vLLM 매니저 가져오기
                vllm_manager = get_vllm_manager()

                # RunPod 서버 상태 확인
                if not await vllm_manager.health_check():
                    logger.warning("RunPod 서버에 연결할 수 없어 기본 응답을 사용합니다.")
                    error_response = f"안녕하세요! 저는 {influencer.influencer_name}입니다. '{request.message}'에 대한 답변을 드리겠습니다."
                    yield f"data: {json.dumps({'text': error_response})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return

                # 시스템 프롬프트 구성
                system_message = (
                    str(influencer.system_prompt)
                    if influencer.system_prompt is not None
                    else f"당신은 {influencer.influencer_name}입니다. 도움이 되는 답변을 해주세요."
                )

                # RunPod 서버에서 스트리밍 응답 생성
                lora_adapter = None
                hf_repo = None
                hf_token = None
                
                logger.info(f"🔍 [Stream] Influencer 정보: id={influencer.influencer_id}, model_repo={influencer.influencer_model_repo}")
                
                if influencer.influencer_id:
                    # LoRA 어댑터 이름 설정 (인플루언서 ID 사용)
                    lora_adapter = str(influencer.influencer_id)
                    
                    if influencer.influencer_model_repo:
                        # DB에 저장된 HF 레포지토리 경로 사용
                        hf_repo = str(influencer.influencer_model_repo)
                        logger.info(f"🔧 LoRA 어댑터 사용: {lora_adapter}, HF repo: {hf_repo}")
                    else:
                        # model_repo가 없으면 기본 경로 패턴 사용 (임시)
                        logger.warning(f"⚠️ [Stream] Influencer model_repo가 없음: id={influencer.influencer_id}")
                        logger.warning(f"⚠️ [Stream] 데이터베이스에 HuggingFace repository 경로를 설정해야 합니다!")
                        # HF repo 없이는 작동하지 않으므로 None으로 설정
                        lora_adapter = None
                
                # HF 토큰 가져오기
                if hf_repo:
                    try:
                        from app.services.hf_token_resolver import get_token_for_influencer
                        hf_token, hf_username = await get_token_for_influencer(influencer, db)
                        if hf_token:
                            logger.info(f"🔑 HF 토큰 사용 (user: {hf_username})")
                    except Exception as e:
                        logger.warning(f"⚠️ HF 토큰 가져오기 실패: {e}")
                
                # 필수 파라미터 검증
                if not hf_token or not hf_repo:
                    logger.error(f"필수 파라미터 누락 - hf_token: {'있음' if hf_token else '없음'}, hf_repo: {'있음' if hf_repo else '없음'}")
                    # 스트리밍 에러 응답
                    error_response = "모델 설정이 완료되지 않았습니다. 관리자에게 문의하세요."
                    yield f"data: {json.dumps({'text': error_response})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    return
                
                # 생각중 상태 전송
                yield f"data: {json.dumps({'status': 'thinking', 'message': '생각중...'}, ensure_ascii=False)}\n\n"
                
                # runsync로 응답을 받고 클라이언트에 스트리밍으로 전달
                payload = {
                    "input": {
                        "hf_token": hf_token,
                        "hf_repo": hf_repo,
                        "system_message": system_message,
                        "prompt": request.message,
                        "temperature": 1,
                        "max_tokens": 2048
                    }
                }
                
                # runsync로 전체 응답 받기
                result = await vllm_manager.runsync(payload)
                
                # 응답 처리
                response_text = ""
                if result.get("status") == "completed":
                    response_text = result.get("generated_text", "")
                    if not response_text:
                        # 이전 형식 호환성
                        output = result.get("output", {})
                        response_text = output.get("generated_text", "")
                    
                    if not response_text:
                        response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. 응답 생성 중 문제가 발생했습니다."
                        
                    logger.info(f"✅ 생성된 텍스트: {response_text[:100]}...")
                else:
                    logger.error(f"❌ RunPod 요청 실패: {result.get('error', 'Unknown error')}")
                    response_text = f"안녕하세요! 저는 {influencer.influencer_name}입니다. '{request.message}'에 대한 답변을 드리겠습니다."
                
                # 타이핑 시작 상태 전송
                yield f"data: {json.dumps({'status': 'typing', 'message': '답변 입력중...'}, ensure_ascii=False)}\n\n"
                
                # 받은 응답을 단어 단위로 분할해서 스트리밍
                import asyncio
                words = response_text.split()
                chunk_size = 2  # 2단어씩 전송
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    # 마지막 청크가 아니면 공백 추가
                    if i + chunk_size < len(words):
                        chunk_text += ' '
                    
                    # 클라이언트에 청크 전송
                    yield f"data: {json.dumps({'text': chunk_text}, ensure_ascii=False)}\n\n"
                    
                    # 스트리밍 효과를 위한 딜레이
                    await asyncio.sleep(0.1)
                
                # 스트리밍 완료 신호
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
                logger.info(f"✅ RunPod 응답을 스트리밍으로 전달 완료: {influencer.influencer_name}")

            except Exception as e:
                logger.error(f"❌ RunPod 스트리밍 응답 생성 실패: {e}")
                # RunPod 실패 시 기본 응답 사용
                error_response = f"안녕하세요! 저는 {influencer.influencer_name}입니다. '{request.message}'에 대한 답변을 드리겠습니다."
                yield f"data: {json.dumps({'text': error_response}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8",
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot streaming error: {str(e)}",
        )


async def track_api_usage(db: Session, influencer_id: str):
    """API 사용량 추적"""
    try:
        logger.info(f"📊 API 사용량 추적 시작 - influencer_id: {influencer_id}")
        
        # API 키 조회
        api_key = (
            db.query(InfluencerAPI)
            .filter(InfluencerAPI.influencer_id == influencer_id)
            .first()
        )
        
        if not api_key:
            logger.warning(f"⚠️ API 키를 찾을 수 없음 - influencer_id: {influencer_id}")
            return
        
        logger.info(f"🔑 API 키 조회 성공 - api_id: {api_key.api_id}")
        
        today = datetime.now().date()

        # 오늘 날짜의 API 호출 집계 조회
        aggregation = (
            db.query(APICallAggregation)
            .filter(
                APICallAggregation.api_id == api_key.api_id,
                APICallAggregation.created_at >= today,
            )
            .first()
        )

        if aggregation:
            # 기존 집계 업데이트
            old_count = aggregation.daily_call_count
            aggregation.daily_call_count += 1
            aggregation.updated_at = datetime.now()
            logger.info(f"✅ 기존 집계 업데이트 - api_id: {api_key.api_id}, 이전: {old_count}, 현재: {aggregation.daily_call_count}")
        else:
            # 새로운 집계 생성
            aggregation = APICallAggregation(
                api_id=api_key.api_id,
                influencer_id=influencer_id,
                daily_call_count=1,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            db.add(aggregation)
            logger.info(f"🆕 새로운 집계 생성 - api_id: {api_key.api_id}, influencer_id: {influencer_id}")

        db.commit()
        logger.info(f"💾 API 사용량 추적 완료 - influencer_id: {influencer_id}, api_id: {api_key.api_id}")

    except Exception as e:
        # API 사용량 추적 실패는 로그만 남기고 계속 진행
        logger.error(f"❌ API usage tracking failed: {e}")
        db.rollback()


# 기존 사용자 인증 기반 엔드포인트들 (관리용)
@router.get("", response_model=List[ChatMessageSchema])
async def get_chat_messages(
    influencer_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """채팅 메시지 목록 조회"""
    # 인플루언서 소유권 확인
    influencer = (
        db.query(AIInfluencer)
        .filter(
            AIInfluencer.influencer_id == influencer_id,
            AIInfluencer.user_id == current_user.user_id,
        )
        .first()
    )

    if influencer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Influencer not found"
        )

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.influencer_id == influencer_id)
        .offset(skip)
        .limit(limit)
        .all()
    )

    return messages


@router.post("", response_model=ChatMessageSchema)
async def create_chat_message(
    message_data: ChatMessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """새 채팅 메시지 생성"""
    # 인플루언서 소유권 확인
    influencer = (
        db.query(AIInfluencer)
        .filter(
            AIInfluencer.influencer_id == message_data.influencer_id,
            AIInfluencer.user_id == current_user.user_id,
        )
        .first()
    )

    if influencer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Influencer not found"
        )

    message = ChatMessage(
        chat_message_id=str(uuid.uuid4()),
        session_id=message_data.session_id if hasattr(message_data, 'session_id') else str(uuid.uuid4()),
        influencer_id=message_data.influencer_id,
        message_content=message_data.message_content,
        message_type=message_data.message_type,
        created_at=get_current_kst(),
        end_at=message_data.end_at,
    )

    db.add(message)
    db.commit()
    db.refresh(message)

    return message


@router.get("/{session_id}", response_model=ChatMessageSchema)
async def get_chat_message(
    session_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """특정 채팅 메시지 조회"""
    message = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).first()

    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat message not found"
        )

    # 인플루언서 소유권 확인
    influencer = (
        db.query(AIInfluencer)
        .filter(
            AIInfluencer.influencer_id == message.influencer_id,
            AIInfluencer.user_id == current_user.user_id,
        )
        .first()
    )

    if influencer is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this chat message",
        )

    return message
