from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
import json
import logging
import os
from app.database import get_db
from app.models.influencer import AIInfluencer
from app.models.voice import VoiceBase, GeneratedVoice
from app.services.runpod_manager import get_tts_manager
from app.services.s3_service import get_s3_service
from app.core.security import get_current_user
from app.schemas.tts import TTSResultRequest, TTSResultResponse, TTSResultMetadata
from app.core.config import settings
import base64

logger = logging.getLogger(__name__)

router = APIRouter()




class VoiceGenerationRequest(BaseModel):
    text: str
    influencer_id: str
    base_voice_url: Optional[str] = None


@router.post("/generate_voice")
async def generate_voice(
    request: VoiceGenerationRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
    s3_service = Depends(get_s3_service),
):
    """텍스트를 음성으로 변환"""
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    # 인플루언서 확인
    influencer = db.query(AIInfluencer).filter(
        AIInfluencer.influencer_id == request.influencer_id
    ).first()
    
    if not influencer:
        raise HTTPException(status_code=404, detail="인플루언서를 찾을 수 없습니다")
    
    # 베이스 음성 확인
    base_voice = db.query(VoiceBase).filter(
        VoiceBase.influencer_id == influencer.influencer_id
    ).first()
    
    if not base_voice:
        raise HTTPException(status_code=400, detail="베이스 음성이 설정되지 않았습니다")
    
    # 텍스트 길이 검증
    if len(request.text) > 500:
        raise HTTPException(status_code=400, detail="텍스트는 500자 이하여야 합니다")
    
    try:
        import base64
        import os

        tts_manager = get_tts_manager()

        # 베이스 음성(로컬 파일) → base64 (음성 클로닝 참조)
        voice_b64 = None
        bpath = base_voice.s3_url
        if bpath and os.path.exists(bpath):
            with open(bpath, "rb") as bf:
                voice_b64 = base64.b64encode(bf.read()).decode()
        else:
            logger.warning(f"베이스 음성 파일 없음(기본 음색 사용): {bpath}")

        # DB 레코드 생성 (pending)
        generated_voice = GeneratedVoice(
            influencer_id=influencer.influencer_id,
            base_voice_id=base_voice.id,
            text=request.text,
            task_id=None,
            status="pending",
            s3_url=None,
            s3_key=None,
            duration=None,
            file_size=None,
        )
        db.add(generated_voice)
        db.commit()
        db.refresh(generated_voice)
        voice_id = generated_voice.id
        logger.info(f"음성 생성 레코드 생성: voice_id={voice_id}")

        # Modal XTTS 동기 호출 (응답: {output:{audio_base64, duration, ...}})
        job_input = {
            "text": request.text,
            "influencer_id": request.influencer_id,
            "voice_data_base64": voice_b64,
        }
        result = await tts_manager.runsync(job_input)
        output = (result or {}).get("output", {}) or {}
        audio_b64 = output.get("audio_base64") or output.get("audio_data") or output.get("audio")

        if not audio_b64:
            generated_voice.status = "failed"
            db.commit()
            err = (result or {}).get("error", "응답에 오디오가 없음")
            logger.error(f"음성 생성 실패: {err}")
            raise HTTPException(status_code=500, detail=f"음성 생성에 실패했습니다: {err}")

        # 로컬 저장: uploads/voices/{influencer}/generated/{voice_id}.wav (외부 볼륨)
        audio_bytes = base64.b64decode(audio_b64)
        rel_path = f"{request.influencer_id}/generated/{voice_id}.wav"
        save_dir = os.path.join("uploads", "voices", request.influencer_id, "generated")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{voice_id}.wav")
        with open(save_path, "wb") as f:
            f.write(audio_bytes)
        play_url = f"/api/v1/voices/{rel_path}"

        generated_voice.status = "completed"
        generated_voice.s3_url = play_url       # 프론트 재생 URL (StaticFiles 서빙)
        generated_voice.s3_key = save_path      # 로컬 파일 경로 (삭제용)
        generated_voice.duration = output.get("duration")
        generated_voice.file_size = len(audio_bytes)
        db.commit()

        logger.info(f"✅ 음성 생성 완료: voice_id={voice_id}, {len(audio_bytes)} bytes -> {save_path}")
        return {
            "voice_id": voice_id,
            "url": play_url,
            "s3_url": play_url,
            "status": "completed",
            "duration": output.get("duration"),
            "file_size": len(audio_bytes),
            "text": request.text,
            "created_at": generated_voice.created_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"음성 생성 실패: {str(e)}")
        logger.error(f"상세 에러: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e) or "음성 생성 중 오류가 발생했습니다")




@router.get("/status/{task_id}")
async def get_voice_generation_status(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """음성 생성 작업 상태 조회"""
    user_id = current_user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    # 작업 조회
    voice = db.query(GeneratedVoice).filter(
        GeneratedVoice.task_id == task_id
    ).first()
    
    if not voice:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    
    # 권한 확인 (인플루언서 소유자인지 확인)
    influencer = db.query(AIInfluencer).filter(
        AIInfluencer.influencer_id == voice.influencer_id
    ).first()
    
    if not influencer or influencer.user_id != user_id:
        raise HTTPException(status_code=403, detail="접근 권한이 없습니다")
    
    return {
        "task_id": task_id,
        "status": voice.status,
        "text": voice.text,
        "s3_url": voice.s3_url,
        "duration": voice.duration,
        "file_size": voice.file_size,
        "created_at": voice.created_at.isoformat() if voice.created_at else None,
        "completed_at": voice.updated_at.isoformat() if voice.status == "completed" and voice.updated_at else None
    }


@router.post("/result", response_model=TTSResultResponse)
async def receive_tts_result(
    request: TTSResultRequest,
    db: Session = Depends(get_db),
    s3_service = Depends(get_s3_service),
):
    """TTS Worker로부터 음성 생성 결과 수신"""
    logger.info("TTS 결과 수신 시작")
    
    try:
        # 메타데이터 파싱
        metadata = request.metadata
        job_id = metadata.get("job_id", "unknown")
        
        # Base64 디코딩
        try:
            audio_data = base64.b64decode(request.audio_base64)
            logger.info(f"음성 데이터 디코딩 성공: {len(audio_data)} bytes")
        except Exception as e:
            logger.error(f"Base64 디코딩 실패: {e}")
            return TTSResultResponse(
                success=False,
                message="Invalid base64 audio data",
                error=str(e)
            )
        
        # S3에 업로드
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"tts/generated/{timestamp}_{job_id[:8]}.wav"
        
        try:
            s3_result = await s3_service.upload_file_from_bytes(
                file_bytes=audio_data,
                key=s3_key,
                content_type="audio/wav"
            )
            
            logger.info(f"S3 업로드 성공: {s3_result['url']}")
            
            # 필수 필드 검증
            influencer_id = metadata.get("influencer_id")
            base_voice_id = metadata.get("base_voice_id")
            
            if not influencer_id:
                logger.error(f"influencer_id가 없습니다. metadata: {metadata}")
                return TTSResultResponse(
                    success=False,
                    message="Missing influencer_id in metadata",
                    error="influencer_id is required"
                )
            
            if not base_voice_id:
                logger.error(f"base_voice_id가 없습니다. metadata: {metadata}")
                return TTSResultResponse(
                    success=False,
                    message="Missing base_voice_id in metadata",
                    error="base_voice_id is required"
                )
            
            # 메타데이터에서 voice_id 가져오기
            voice_id = metadata.get("voice_id")
            
            if not voice_id:
                logger.error(f"voice_id가 없습니다. metadata: {metadata}")
                return TTSResultResponse(
                    success=False,
                    message="Missing voice_id in metadata",
                    error="voice_id is required"
                )
            
            # 기존 레코드 조회 (voice_id로 검색)
            existing_voice = db.query(GeneratedVoice).filter(
                GeneratedVoice.id == voice_id
            ).first()
            
            if existing_voice:
                # 기존 레코드 업데이트
                existing_voice.status = "completed"
                existing_voice.s3_url = s3_result["url"]
                existing_voice.s3_key = s3_key
                existing_voice.duration = metadata.get("duration")
                existing_voice.file_size = metadata.get("file_size")
                existing_voice.metadata = json.dumps(metadata)
                existing_voice.updated_at = datetime.now()
                
                logger.info(f"기존 TTS 레코드 업데이트: task_id={job_id}")
            else:
                # 기존 레코드가 없으면 새로 생성
                generated_voice = GeneratedVoice(
                    influencer_id=influencer_id,  # 검증된 값 사용
                    base_voice_id=base_voice_id,  # 검증된 값 사용
                    text=metadata.get("text", ""),
                    task_id=job_id,
                    status="completed",
                    s3_url=s3_result["url"],
                    s3_key=s3_key,
                    duration=metadata.get("duration"),
                    file_size=metadata.get("file_size"),
                    metadata=json.dumps(metadata)  # 전체 메타데이터 저장
                )
                
                db.add(generated_voice)
                logger.info(f"새로운 TTS 레코드 생성: task_id={job_id}")
            
            db.commit()
            
            return TTSResultResponse(
                success=True,
                message="TTS result saved successfully",
                s3_url=s3_result["url"],
                task_id=job_id
            )
            
        except Exception as e:
            logger.error(f"S3 업로드 실패: {e}")
            return TTSResultResponse(
                success=False,
                message="Failed to upload to S3",
                error=str(e)
            )
            
    except Exception as e:
        logger.error(f"TTS 결과 처리 중 오류: {str(e)}")
        import traceback
        logger.error(f"상세 에러: {traceback.format_exc()}")
        
        return TTSResultResponse(
            success=False,
            message="Failed to process TTS result",
            error=str(e)
        )