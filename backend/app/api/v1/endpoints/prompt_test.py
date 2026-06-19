"""
프롬프트 최적화 테스트 API 엔드포인트

ComfyUI 실행 없이 프롬프트 최적화 결과만 확인할 수 있는 테스트용 API
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging

from app.core.security import get_current_user
from app.services.prompt_optimization_service import get_prompt_optimization_service

logger = logging.getLogger(__name__)

router = APIRouter()


# 요청/응답 스키마
class PromptTestRequest(BaseModel):
    """프롬프트 테스트 요청"""
    prompt: str
    selected_styles: Optional[Dict[str, str]] = {}


class PromptTestResponse(BaseModel):
    """프롬프트 테스트 응답"""
    success: bool
    original_prompt: str
    selected_styles: Dict[str, str]
    optimized_prompt: str
    character_count: int
    estimated_tokens: int
    style_keywords_applied: List[str]
    optimization_method: str
    message: str


@router.post("/test-prompt", response_model=PromptTestResponse)
async def test_prompt_optimization(
    request: PromptTestRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    프롬프트 최적화 테스트
    
    한국어 프롬프트 + 스타일 선택이 어떻게 SDXL-Turbo 최적화
    영문 프롬프트로 변환되는지 미리 확인 (실제 생성과 동일 경로)
    """
    try:
        user_id = current_user.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="사용자 ID를 찾을 수 없습니다.")
        
        logger.info(f"🧪 프롬프트 최적화 테스트 시작 - 사용자: {user_id}")
        logger.info(f"📝 원본 프롬프트: '{request.prompt}'")
        logger.info(f"🎨 선택된 스타일: {request.selected_styles}")
        
        # 프롬프트 최적화 서비스 호출
        prompt_service = get_prompt_optimization_service()
        
        # 스타일 키워드 수집 (내부 메서드 접근)
        style_keywords_dict = prompt_service._collect_flux_style_keywords(request.selected_styles)
        style_keywords_applied = []
        for category, keywords in style_keywords_dict.items():
            style_keywords_applied.extend(keywords.split(", "))
        
        # 최적화 실행 (실제 생성 모델 = SDXL-Turbo 와 동일 경로)
        try:
            optimized_prompt = await prompt_service.optimize_sdxl_prompt(
                user_prompt=request.prompt,
                selected_styles=request.selected_styles
            )
            optimization_method = "openai_sdxl_optimization"

        except Exception as optimization_error:
            logger.warning(f"⚠️ OpenAI 최적화 실패, 폴백 모드 사용: {optimization_error}")
            # 폴백 모드 시뮬레이션
            optimized_prompt = prompt_service._flux_fallback_prompt(
                request.prompt, 
                request.selected_styles
            )
            optimization_method = "fallback_translation"
        
        # 토큰 수 추정 (대략적으로 4글자 = 1토큰)
        estimated_tokens = len(optimized_prompt) // 4
        
        # 최적화 품질 평가
        character_count = len(optimized_prompt)
        
        if character_count <= 350 and estimated_tokens <= 77:
            quality_message = "✅ SDXL-Turbo에 최적화된 프롬프트입니다."
        elif character_count > 350:
            quality_message = "⚠️ 프롬프트가 다소 긴 편입니다. 더 간결하게 조정을 권장합니다."
        else:
            quality_message = "🔍 프롬프트가 생성되었습니다."
        
        logger.info(f"✅ 프롬프트 최적화 완료")
        logger.info(f"   최적화 결과: '{optimized_prompt[:50]}...'")
        logger.info(f"   길이: {character_count}자, 추정 토큰: {estimated_tokens}")
        
        return PromptTestResponse(
            success=True,
            original_prompt=request.prompt,
            selected_styles=request.selected_styles,
            optimized_prompt=optimized_prompt,
            character_count=character_count,
            estimated_tokens=estimated_tokens,
            style_keywords_applied=style_keywords_applied,
            optimization_method=optimization_method,
            message=quality_message
        )
        
    except Exception as e:
        logger.error(f"❌ 프롬프트 테스트 실패: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"프롬프트 최적화 테스트 실패: {str(e)}"
        )


@router.post("/test-multiple-prompts")
async def test_multiple_prompts(
    prompts_data: List[PromptTestRequest],
    current_user: Dict = Depends(get_current_user)
):
    """
    여러 프롬프트 일괄 테스트
    
    다양한 프롬프트와 스타일 조합을 한 번에 테스트
    """
    try:
        user_id = current_user.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="사용자 ID를 찾을 수 없습니다.")
        
        logger.info(f"🧪 일괄 프롬프트 테스트 시작 - {len(prompts_data)}개 프롬프트")
        
        results = []
        prompt_service = get_prompt_optimization_service()
        
        for i, prompt_data in enumerate(prompts_data, 1):
            try:
                logger.info(f"📝 테스트 {i}/{len(prompts_data)}: '{prompt_data.prompt}'")
                
                # 개별 프롬프트 최적화
                optimized_prompt = await prompt_service.optimize_flux_prompt(
                    user_prompt=prompt_data.prompt,
                    selected_styles=prompt_data.selected_styles
                )
                
                # 스타일 키워드 수집
                style_keywords_dict = prompt_service._collect_flux_style_keywords(prompt_data.selected_styles)
                style_keywords_applied = []
                for keywords in style_keywords_dict.values():
                    style_keywords_applied.extend(keywords.split(", "))
                
                results.append({
                    "test_number": i,
                    "original_prompt": prompt_data.prompt,
                    "selected_styles": prompt_data.selected_styles,
                    "optimized_prompt": optimized_prompt,
                    "character_count": len(optimized_prompt),
                    "estimated_tokens": len(optimized_prompt) // 4,
                    "style_keywords_applied": style_keywords_applied,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"❌ 프롬프트 {i} 최적화 실패: {e}")
                results.append({
                    "test_number": i,
                    "original_prompt": prompt_data.prompt,
                    "selected_styles": prompt_data.selected_styles,
                    "optimized_prompt": "",
                    "character_count": 0,
                    "estimated_tokens": 0,
                    "style_keywords_applied": [],
                    "success": False,
                    "error": str(e)
                })
        
        successful_tests = sum(1 for r in results if r["success"])
        
        logger.info(f"✅ 일괄 테스트 완료: {successful_tests}/{len(prompts_data)} 성공")
        
        return {
            "success": True,
            "total_tests": len(prompts_data),
            "successful_tests": successful_tests,
            "failed_tests": len(prompts_data) - successful_tests,
            "results": results,
            "message": f"{successful_tests}/{len(prompts_data)}개 프롬프트 최적화 완료"
        }
        
    except Exception as e:
        logger.error(f"❌ 일괄 프롬프트 테스트 실패: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"일괄 프롬프트 테스트 실패: {str(e)}"
        )


@router.get("/test-examples")
async def get_test_examples():
    """
    프롬프트 테스트용 예시 데이터 제공
    """
    examples = [
        {
            "prompt": "아름다운 여성의 초상화",
            "selected_styles": {
                "대분류": "사람",
                "세부스타일": "사실적", 
                "분위기": "따뜻한"
            },
            "description": "사실적인 여성 초상화 (따뜻한 분위기)"
        },
        {
            "prompt": "고양이가 창가에서 햇살을 받으며 잠자는 모습",
            "selected_styles": {
                "대분류": "동물",
                "세부스타일": "디지털아트",
                "분위기": "밝은"
            },
            "description": "디지털아트 스타일의 고양이 (밝은 분위기)"
        },
        {
            "prompt": "미래도시의 네온사인이 빛나는 야경",
            "selected_styles": {
                "대분류": "건물",
                "세부스타일": "판타지",
                "분위기": "어두운"
            },
            "description": "판타지 스타일의 미래도시 (어두운 분위기)"
        },
        {
            "prompt": "산속의 작은 오두막과 주변 자연 풍경",
            "selected_styles": {
                "대분류": "풍경",
                "세부스타일": "수채화",
                "분위기": "신비로운"
            },
            "description": "수채화 스타일의 자연 풍경 (신비로운 분위기)"
        },
        {
            "prompt": "빈티지한 책상 위의 오래된 책과 촛불",
            "selected_styles": {
                "대분류": "사물",
                "세부스타일": "유화",
                "분위기": "차가운"
            },
            "description": "유화 스타일의 정물화 (차가운 분위기)"
        }
    ]
    
    return {
        "success": True,
        "examples": examples,
        "message": f"{len(examples)}개의 테스트 예시를 제공합니다."
    }