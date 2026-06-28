# AI slop cleanup change log

## 2026-06-28

### 삭제

| 파일 | 이유 |
|---|---|
| `backend/app/application/use_cases/base.py` | 실제 유스케이스에서 참조되지 않는 범용 CQRS/Decorator 골격 |
| `backend/app/domain/entities/base.py` | 실제 도메인 모델에서 사용되지 않는 DDD 기반 클래스 모음 |
| `backend/app/presentation/dependencies/container.py` | 등록 서비스 없이 남아 있던 독자 DI 컨테이너 |
| `backend/app/utils/api_responses.py` | 호출처가 없는 두 번째 API 응답 추상화 |
| `backend/debug_workflow_example.py` | 제품 실행 경로와 분리된 디버그 예제 |
| `frontend-websocket-tts-example.tsx` | 프런트 앱에서 import되지 않는 루트 예제 |
| `frontend/components/debug/MBTIDebugPanel.tsx` | import되지 않는 디버그 패널 |
| `frontend/lib/debug/mbti-debug-utils.ts` | import되지 않는 디버그 유틸리티 |
| `frontend/components/instagram-debug-helper.tsx` | import되지 않는 디버그 컴포넌트 |
| `frontend/components/instagram-business-guide.tsx` | import되지 않는 독립 가이드 컴포넌트 |

### 테스트 보강

- `backend/tests/test_data_mapping.py`
  - 성별·나이·모델 유형 매핑과 캐릭터 데이터 생성의 현재 동작을 고정했다.
  - silent default는 이번 정리에서 변경하지 않고 후속 판단 대상으로 남겼다.

### 보존한 항목

- `backups/`, `data/`, `system_architecture/`: 사용자 데이터·설계 산출물로 판단해 삭제하지 않았다.
- `frontend/personal.md`: 개인정보 처리방침이므로 미사용 여부만으로 삭제하지 않았다.
- 인증 우회, DB sync fallback, 외부 서비스 fallback: 동작·보안 계약 변경이 필요해 후속 이슈로 분리했다.

### 검증

- `python -m unittest tests.test_data_mapping -v`
- `python -m compileall -q app`
- `npx.cmd tsc --noEmit --pretty false --incremental false`
- 삭제된 심볼·파일명에 대한 잔여 참조 검색

백엔드 전체 pytest는 현재 실행 Python에 `pytest`와 FastAPI 의존성이 없어 수행하지 못했다.
