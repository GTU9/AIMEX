"""
QA 생성 서비스
인플루언서 파인튜닝을 위한 QA 데이터셋 생성
"""

import os
import re
import time
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import tempfile
from openai import OpenAI, AsyncOpenAI

from app.schemas.qa_generation import CharacterProfile, Gender
from app.core.encryption import decrypt_sensitive_data

logger = logging.getLogger(__name__)

# 도메인별 QA 생성을 위한 카테고리 정의
DOMAIN_CATEGORIES = [
    "일상생활",
    "과학기술", 
    "사회이슈",
    "인문학",
    "스포츠",
    "역사문화"
]

# 도메인별 특성 설명
DOMAIN_DESCRIPTIONS = {
    "일상생활": "일상의 소소한 일들, 취미, 습관, 음식, 주말 활동 등",
    "과학기술": "AI, 기술 트렌드, 스마트폰, 미래 기술, 과학의 발전",
    "사회이슈": "사회 문제, 환경, 불평등, 세대 간 차이, 미래 사회",
    "인문학": "인생의 가치, 책, 예술, 철학, 역사의 교훈",
    "스포츠": "운동, 건강관리, 스포츠 경기, 운동의 즐거움",
    "역사문화": "전통문화, 역사적 장소, 문화의 다양성, 역사 인물"
}


class QAGenerationService:
    """QA 생성 서비스"""
    
    def __init__(self):
        """서비스 초기화"""
        self.openai_client = None
        self._initialize_openai()
        
    def _initialize_openai(self):
        """OpenAI 클라이언트 초기화"""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
            self.async_openai_client = AsyncOpenAI(api_key=api_key)
            logger.info("✅ OpenAI 클라이언트 초기화 완료")
        else:
            self.async_openai_client = None
            logger.warning("⚠️ OPENAI_API_KEY가 설정되지 않았습니다")
    
    async def generate_qa_for_influencer(
        self,
        character_name: str,
        character_description: str,
        personality: str,
        num_qa_pairs: int = 2000,
        domains: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        인플루언서용 대량 QA 생성
        
        Args:
            character_name: 캐릭터 이름
            character_description: 캐릭터 설명
            personality: 성격
            num_qa_pairs: 생성할 QA 쌍 개수
            domains: 도메인 리스트
            system_prompt: 시스템 프롬프트
            
        Returns:
            배치 요청 데이터
        """
        if not self.openai_client:
            raise Exception("OpenAI 클라이언트가 초기화되지 않았습니다")
        
        # 도메인 설정
        if not domains:
            domains = DOMAIN_CATEGORIES
        
        # 도메인별 QA 개수 계산 (균등 분배)
        qa_per_domain = num_qa_pairs // len(domains)
        remaining = num_qa_pairs % len(domains)
        
        logger.info(f"🎯 QA 생성 시작: {character_name}, 총 {num_qa_pairs}개")
        logger.info(f"📊 도메인별 할당: {qa_per_domain}개 (나머지: {remaining})")
        
        # 배치 요청 생성
        batch_requests = []
        request_count = 0
        
        for domain_idx, domain in enumerate(domains):
            # 마지막 도메인에 나머지 할당
            domain_qa_count = qa_per_domain
            if domain_idx == len(domains) - 1:
                domain_qa_count += remaining
            
            domain_desc = DOMAIN_DESCRIPTIONS.get(domain, domain)
            
            # 각 도메인에 대해 여러 요청 생성 (각 요청당 7개 QA)
            requests_for_domain = domain_qa_count // 7
            remaining_qa = domain_qa_count % 7
            
            for req_idx in range(requests_for_domain):
                custom_id = f"influencer_qa_{character_name}_{domain}_{request_count}"
                
                batch_request = self._create_batch_request(
                    custom_id=custom_id,
                    character_name=character_name,
                    character_description=character_description,
                    personality=personality,
                    domain=domain,
                    domain_desc=domain_desc,
                    system_prompt=system_prompt
                )
                
                batch_requests.append(batch_request)
                request_count += 1
            
            # 나머지 QA 처리
            if remaining_qa > 0:
                custom_id = f"influencer_qa_{character_name}_{domain}_{request_count}_partial"
                
                batch_request = self._create_batch_request(
                    custom_id=custom_id,
                    character_name=character_name,
                    character_description=character_description,
                    personality=personality,
                    domain=domain,
                    domain_desc=domain_desc,
                    system_prompt=system_prompt,
                    num_qa=remaining_qa
                )
                
                batch_requests.append(batch_request)
                request_count += 1
        
        logger.info(f"✅ 배치 요청 생성 완료: 총 {len(batch_requests)}개 요청")
        
        return {
            "batch_requests": batch_requests,
            "total_requests": len(batch_requests),
            "domains": domains,
            "qa_per_domain": qa_per_domain,
            "character_name": character_name
        }
    
    def _create_batch_request(
        self,
        custom_id: str,
        character_name: str,
        character_description: str,
        personality: str,
        domain: str,
        domain_desc: str,
        system_prompt: Optional[str] = None,
        num_qa: int = 7
    ) -> Dict[str, Any]:
        """
        OpenAI Batch API 형식의 요청 생성
        
        Args:
            custom_id: 요청 ID
            character_name: 캐릭터 이름
            character_description: 캐릭터 설명
            personality: 성격
            domain: 도메인
            domain_desc: 도메인 설명
            system_prompt: 시스템 프롬프트
            num_qa: QA 개수 (기본 7개)
            
        Returns:
            배치 요청 딕셔너리
        """
        if not system_prompt:
            system_prompt = f"당신은 {character_name}라는 캐릭터입니다. {personality} 성격을 가지고 있습니다."
        
        user_prompt = f"""{domain}({domain_desc})에 관한 {num_qa}턴의 자연스러운 멀티턴 대화를 만들어주세요.

{character_name}의 성격과 특성에 맞는 흥미롭고 깊이 있는 대화를 만들어주세요.
대화는 하나의 주제로 시작해서 자연스럽게 연결되며 점차 깊어지는 형태여야 합니다.

반드시 다음 JSON 배열 형식으로 답변해주세요:
[
  {{"q": "첫 번째 질문", "a": "첫 번째 답변"}},
  {{"q": "이전 답변을 바탕으로 한 후속 질문", "a": "더 구체적인 답변"}},
  {{"q": "더 깊이 파고드는 질문", "a": "개인적 경험이나 의견이 담긴 답변"}},
  {{"q": "다른 관점에서의 질문", "a": "새로운 시각을 제시하는 답변"}},
  {{"q": "실용적인 조언을 구하는 질문", "a": "구체적인 제안이 담긴 답변"}},
  {{"q": "감정이나 가치관에 대한 질문", "a": "캐릭터의 철학이 드러나는 답변"}},
  {{"q": "마무리하며 정리를 요청하는 질문", "a": "대화를 종합하며 핵심을 정리하는 답변"}}
]"""
        
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": user_prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7,
                "response_format": {"type": "json_object"}  # JSON 형식 강제
            }
        }
    
    async def create_batch_file(
        self,
        batch_requests: List[Dict[str, Any]]
    ) -> str:
        """
        배치 요청을 JSONL 파일로 생성하고 OpenAI에 업로드
        
        Args:
            batch_requests: 배치 요청 리스트
            
        Returns:
            업로드된 파일 ID
        """
        if not self.openai_client:
            raise Exception("OpenAI 클라이언트가 초기화되지 않았습니다")
        
        # 임시 JSONL 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            for request in batch_requests:
                tmp_file.write(json.dumps(request) + '\n')
            tmp_file_path = tmp_file.name
        
        try:
            # OpenAI에 파일 업로드
            with open(tmp_file_path, 'rb') as file:
                response = self.openai_client.files.create(
                    file=file,
                    purpose='batch'
                )
            
            file_id = response.id
            logger.info(f"✅ 배치 파일 업로드 완료: {file_id}")
            
            return file_id
            
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_file_path)
    
    async def submit_batch(
        self,
        file_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        OpenAI Batch API에 작업 제출
        
        Args:
            file_id: 업로드된 파일 ID
            metadata: 메타데이터
            
        Returns:
            배치 ID
        """
        if not self.openai_client:
            raise Exception("OpenAI 클라이언트가 초기화되지 않았습니다")
        
        # 배치 작업 생성
        batch = self.openai_client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=metadata or {}
        )
        
        logger.info(f"✅ 배치 작업 제출 완료: {batch.id}")
        
        return batch.id
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        배치 작업 상태 조회
        
        Args:
            batch_id: 배치 ID
            
        Returns:
            배치 상태 정보
        """
        if not self.openai_client:
            raise Exception("OpenAI 클라이언트가 초기화되지 않았습니다")
        
        batch = self.openai_client.batches.retrieve(batch_id)
        
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "request_counts": batch.request_counts,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id
        }
    
    async def process_batch_results(
        self,
        batch_id: str,
        output_file_id: str
    ) -> Dict[str, Any]:
        """
        배치 결과 처리
        
        Args:
            batch_id: 배치 ID
            output_file_id: 출력 파일 ID
            
        Returns:
            처리된 QA 데이터
        """
        if not self.openai_client:
            raise Exception("OpenAI 클라이언트가 초기화되지 않았습니다")
        
        # 결과 파일 다운로드
        file_response = self.openai_client.files.content(output_file_id)
        
        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_file:
            tmp_file.write(file_response.text)
            tmp_file_path = tmp_file.name
        
        try:
            # 결과 파싱
            qa_pairs = []
            errors = []
            
            with open(tmp_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        result = json.loads(line)
                        
                        # 성공적인 응답 처리
                        if result.get("response") and result["response"].get("body"):
                            choices = result["response"]["body"].get("choices", [])
                            if choices and choices[0].get("message"):
                                content = choices[0]["message"]["content"]
                                
                                # JSON 파싱 (모델 출력: list 또는 {"대화":[...]} dict 모두 허용)
                                try:
                                    raw = content.strip()
                                    if raw.startswith("```"):
                                        # ```json ... ``` 코드펜스 제거
                                        raw = raw.split("```", 2)[1]
                                        if raw.lstrip().lower().startswith("json"):
                                            raw = raw.lstrip()[4:]
                                    qa_data = json.loads(raw)
                                    # dict 면 내부의 첫 리스트 값을 사용 (예: {"대화":[...]})
                                    if isinstance(qa_data, dict):
                                        qa_data = next(
                                            (v for v in qa_data.values() if isinstance(v, list)),
                                            [],
                                        )
                                    if isinstance(qa_data, list):
                                        for item in qa_data:
                                            if not isinstance(item, dict):
                                                continue
                                            q = item.get("question") or item.get("q") or item.get("user")
                                            a = item.get("answer") or item.get("a") or item.get("assistant")
                                            if q and a:
                                                qa_pairs.append({"question": q, "answer": a})
                                except json.JSONDecodeError:
                                    logger.warning(f"JSON 파싱 실패: {content[:100]}...")
                        
                        # 에러 처리
                        elif result.get("error"):
                            errors.append({
                                "custom_id": result.get("custom_id"),
                                "error": result["error"]
                            })
                    
                    except json.JSONDecodeError:
                        logger.error(f"라인 파싱 실패: {line[:100]}...")
            
            logger.info(f"✅ 배치 결과 처리 완료: {len(qa_pairs)}개 QA, {len(errors)}개 에러")
            
            return {
                "qa_pairs": qa_pairs,
                "total_count": len(qa_pairs),
                "errors": errors,
                "error_count": len(errors)
            }
            
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_file_path)
    
    async def generate_tone_variations(self, character_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        캐릭터 기반 어투 변형 생성 (통합 버전)
        vLLM의 generate_qa_fast 로직을 백엔드에 통합
        
        Args:
            character_profile: 캐릭터 프로필 정보
            
        Returns:
            생성된 어투 응답
        """
        if not self.async_openai_client:
            raise Exception("OpenAI 클라이언트가 초기화되지 않았습니다")
        
        start_time = time.time()
        
        # 캐릭터 이름과 성격 추출
        character_name = character_profile.get('name', '캐릭터')
        personality = character_profile.get('personality', '친근하고 활발한 성격')
        description = character_profile.get('description', '')
        age_range = character_profile.get('age_range', '알 수 없음')
        gender = character_profile.get('gender', 'NON_BINARY')
        mbti = character_profile.get('mbti')
        
        logger.info(f"🚀 어투 생성 시작: {character_name}")
        
        # 1. 질문 생성
        question = await self._generate_question_for_character(character_profile)
        logger.info(f"📝 생성된 질문: {question}")
        
        # 2. 3개의 서로 다른 시스템 프롬프트 생성
        try:
            system_prompts = await self._create_three_distinct_system_prompts(character_profile)
            logger.info("✅ 3개 시스템 프롬프트 생성 성공")
        except Exception as e:
            logger.warning(f"시스템 프롬프트 생성 실패, 폴백 방식 사용: {e}")
            # 폴백: 기존 방식으로 3개 생성
            system_prompts = await asyncio.gather(*[
                self._create_character_prompt_for_tone(character_profile, i + 1)
                for i in range(3)
            ])
        
        # 3. 3가지 어투로 응답 생성 (병렬 처리)
        try:
            # 병렬로 3개 어투 생성
            response_tasks = [
                self._generate_single_tone_response(
                    character_profile=character_profile,
                    question=question,
                    system_prompt=system_prompt,
                    tone_num=i+1
                )
                for i, system_prompt in enumerate(system_prompts)
            ]
            tone_results = await asyncio.gather(*response_tasks, return_exceptions=True)
            
            # 결과 정리
            responses = {}
            for i, result in enumerate(tone_results):
                tone_name = f"말투{i+1}"
                if isinstance(result, Exception):
                    logger.error(f"말투 {i+1} 생성 실패: {result}")
                    responses[tone_name] = [{
                        "text": f"죄송합니다. 말투{i+1} 응답 생성에 실패했습니다.",
                        "hashtags": f"#오류 #말투{i+1}",
                        "description": f"생성 실패한 말투{i+1}",
                        "system_prompt": system_prompts[i]
                    }]
                else:
                    result['system_prompt'] = system_prompts[i]
                    responses[tone_name] = [result]
            
            generation_time = time.time() - start_time
            logger.info(f"✅ 어투 생성 완료: {generation_time:.2f}초")
            
            return {
                "question": question,
                "responses": responses,
                "generation_time_seconds": generation_time,
                "method": "integrated_backend"
            }
            
        except Exception as e:
            logger.error(f"❌ 어투 생성 실패: {e}")
            raise Exception(f"어투 생성 중 오류가 발생했습니다: {str(e)}")
    
    async def _generate_question_for_character(self, character_profile: Dict[str, Any]) -> str:
        """캐릭터에 맞는 질문 생성"""
        prompt = f"""
        {character_profile['name']}라는 캐릭터에게 물어볼 만한 흥미로운 질문을 하나 만들어주세요.
        
        캐릭터 정보:
        - 이름: {character_profile['name']}
        - 성격: {character_profile.get('personality', '알 수 없음')}
        - 설명: {character_profile.get('description', '없음')}
        
        요구사항:
        - 캐릭터의 개성이 드러날 수 있는 질문
        - 일상적이면서도 다양한 답변이 가능한 질문
        - 20자 이내의 간단한 질문
        
        질문만 반환하세요.
        """
        
        response = await self.async_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=50
        )
        
        return response.choices[0].message.content.strip()
    
    async def _create_three_distinct_system_prompts(self, character_profile: Dict[str, Any]) -> List[str]:
        """3개의 서로 다른 시스템 프롬프트를 한 번에 생성"""
        prompt = f"""
        {character_profile['name']}라는 캐릭터의 3가지 서로 다른 말투 스타일을 만들어주세요.
        
        캐릭터 정보:
        - 이름: {character_profile['name']}
        - 성격: {character_profile.get('personality', '친근하고 활발한 성격')}
        - 나이대: {character_profile.get('age_range', '알 수 없음')}
        - 성별: {character_profile.get('gender', '없음')}
        - MBTI: {character_profile.get('mbti', '알 수 없음')}
        - 설명: {character_profile.get('description', '없음')}
        
        각 말투는 다음 형식으로 작성해주세요:
        [말투1]
        구체적인 말투 지시사항...
        
        [말투2]
        구체적인 말투 지시사항...
        
        [말투3]
        구체적인 말투 지시사항...
        
        요구사항:
        - 각 말투는 서로 확연히 구별되어야 함
        - 캐릭터의 개성이 잘 드러나야 함
        - 구체적인 어미, 말투, 특징을 포함
        """
        
        response = await self.async_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=800
        )
        
        content = response.choices[0].message.content
        
        # 말투 파싱
        prompts = []
        pattern = r'\[말투\d+\]\s*([^\[]+)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches[:3]:
            system_prompt = f"당신은 {character_profile['name']}입니다. {match.strip()}"
            prompts.append(system_prompt)
        
        # 3개가 안 되면 기본값 추가
        while len(prompts) < 3:
            prompts.append(f"당신은 {character_profile['name']}입니다. 친근하고 활발한 성격으로 대화하세요.")
        
        return prompts
    
    async def _create_character_prompt_for_tone(self, character_profile: Dict[str, Any], tone_num: int) -> str:
        """특정 톤 번호에 대한 캐릭터 프롬프트 생성 (폴백용)"""
        base_prompt = f"당신은 {character_profile['name']}입니다."
        
        tone_variations = [
            "친근하고 활발한 말투로 대화하세요.",
            "차분하고 지적인 말투로 대화하세요.",
            "유머러스하고 재치있는 말투로 대화하세요."
        ]
        
        if 1 <= tone_num <= 3:
            return f"{base_prompt} {tone_variations[tone_num-1]}"
        else:
            return f"{base_prompt} 자유롭게 대화하세요."
    
    async def _generate_single_tone_response(
        self,
        character_profile: Dict[str, Any],
        question: str,
        system_prompt: str,
        tone_num: int
    ) -> Dict[str, Any]:
        """단일 어투로 응답 생성"""
        try:
            # 응답 생성
            response = await self.async_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.8,
                max_tokens=150
            )
            
            text = response.choices[0].message.content.strip()
            
            # 요약 정보 생성
            summary_prompt = f"""
            다음 말투 지시사항을 분석해서 JSON 형식으로 요약해주세요:
            
            말투 지시사항:
            {system_prompt}
            
            다음 형식으로 답변하세요:
            {{
                "hashtags": "#해시태그1 #해시태그2 #해시태그3",
                "description": "이 말투의 특징을 한 문장으로 설명"
            }}
            """
            
            summary_response = await self.async_openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.7,
                max_tokens=100
            )
            
            summary_content = summary_response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            try:
                summary_data = json.loads(summary_content)
                hashtags = summary_data.get('hashtags', f'#말투{tone_num}')
                description = summary_data.get('description', f'말투{tone_num}의 특징')
            except:
                # 파싱 실패 시 기본값
                hashtags = f'#말투{tone_num} #개성있는 #캐릭터'
                description = f'{character_profile["name"]}의 말투{tone_num}'
            
            return {
                "text": text,
                "hashtags": hashtags,
                "description": description
            }
            
        except Exception as e:
            logger.error(f"말투 {tone_num} 생성 중 오류: {e}")
            raise e


# 전역 서비스 인스턴스
qa_generation_service = QAGenerationService()


def get_qa_generation_service() -> QAGenerationService:
    """QA 생성 서비스 의존성 주입용 함수"""
    return qa_generation_service