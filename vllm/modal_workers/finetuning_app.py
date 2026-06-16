"""
Modal Serverless GPU Worker - QLoRA Fine-tuning
EXAONE-3.5-2.4B-Instruct 인플루언서별 LoRA 파인튜닝 후 HuggingFace 업로드.

기존 RunPod 워커(vllm/runpod_workers/finetuning/finetuning_worker.py)를 Modal 로 포팅.
멀티잡 스케줄러는 Modal 의 컨테이너 오토스케일링이 대체하므로 제거하고
핵심 학습 로직만 이식한다.

배포:
    modal deploy vllm/modal_workers/finetuning_app.py
배포 후 발급되는 URL을 backend .env 의 MODAL_FINETUNING_URL 에 설정.

입출력 계약 (백엔드 클라이언트와 일치해야 함):
  입력:  {"input": {"influencer_id": str,
                    "dataset_url": str,     # QA 데이터 JSON URL 또는 인라인 qa_data
                    "base_model": str,
                    "hf_token": str,
                    "hf_repo_id": str|null, # 미지정 시 influencer_id 기반 생성
                    "system_message": str|null}}
  출력:  {"output": {"status": str, "adapter_repo": str}}
"""
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import modal

# ---------------------------------------------------------------------------
# 설정 상수
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_SYSTEM_MESSAGE = "당신은 도움이 되는 AI 어시스턴트입니다."
MODELS_DIR = "/models"
HF_CACHE_DIR = f"{MODELS_DIR}/hf_cache"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Modal App / Image / Volume
# ---------------------------------------------------------------------------
app = modal.App("aimex-finetuning")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # EXAONE-3.5 trust_remote_code 호환 창:
        #   - transformers 5.0.0+ : get_input_embeddings/_tied_weights_keys 비호환(너무 높음)
        #   - transformers 4.48 미만 : 원격코드가 쓰는 RopeParameters 없음(너무 낮음)
        #   → 4.48~4.56 중간 버전(4.53.1)으로 고정.
        "torch==2.5.1",
        "transformers==4.53.1",
        "accelerate==1.8.1",
        "peft==0.14.0",
        "bitsandbytes==0.45.0",
        "datasets==3.1.0",
        "numpy<2",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "scipy",
        "scikit-learn",
        "requests",
        "fastapi[standard]",
    )
    .env({"HF_HOME": HF_CACHE_DIR})
)

volume = modal.Volume.from_name("aimex-models", create_if_missing=True)


# ---------------------------------------------------------------------------
# 데이터 전처리 (기존 RunPod 워커 로직 이식)
# ---------------------------------------------------------------------------
def _find_all_linear_names(model) -> List[str]:
    import torch

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            lora_module_names.add(name.split(".")[-1])
    lora_module_names.discard("lm_head")
    lora_module_names.discard("embed_tokens")
    return list(lora_module_names)


def _create_chat_format(tokenizer, instruction: str, output: str, system_msg: str) -> str:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def _normalize_qa_data(qa_data: List[Dict]) -> List[Dict]:
    """다양한 QA 입력 포맷을 학습용 단일턴 {"question","answer"} 리스트로 정규화.

    torch/transformers 등 무거운 의존 없이 단독 호출 가능한 순수 함수
    (단위 테스트용). 수용 포맷:
      (a) {"question": str, "answer": str}            - 단일 QA
      (b) {"q": str, "a": str}                        - OpenAI 배치 생성 단일 QA
      (c) {"user": str, "assistant": str}             - user/assistant 단일 턴
      (d) {"messages": [{"role","content"}, ...]}     - 멀티턴 세션
                                                        (각 assistant 턴을 직전
                                                         user 발화와 묶어 펼침)
      (e) {"conversation_history": [...], "user","assistant"}
                                                        - 멀티턴 컨텍스트의 최종 턴
                                                          ((c)와 동일 처리)
    system 역할 메시지는 system_message 로 별도 전달되므로 무시한다.
    """
    normalized: List[Dict] = []

    for item in qa_data:
        if not isinstance(item, dict):
            logger.warning("QA 항목이 dict 가 아님, 건너뜀: %r", type(item))
            continue

        # (d) 멀티턴 세션: messages 배열을 각 assistant 턴 기준으로 펼침
        if isinstance(item.get("messages"), list):
            last_user: Optional[str] = None
            for msg in item["messages"]:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    last_user = content
                elif role == "assistant" and last_user:
                    normalized.append({"question": last_user, "answer": content})
                    last_user = None
            continue

        # (a)/(b)/(c)/(e): 키 별칭 정규화
        question = (
            item.get("question")
            or item.get("q")
            or item.get("user")
            or ""
        ).strip()
        answer = (
            item.get("answer")
            or item.get("a")
            or item.get("assistant")
            or ""
        ).strip()

        if not question or not answer:
            logger.warning("빈 질문/답변 건너뜀: q=%r a=%r", question[:30], answer[:30])
            continue

        normalized.append({"question": question, "answer": answer})

    if not normalized:
        raise ValueError("정규화 후 유효한 QA 샘플이 없습니다.")

    return normalized


def _prepare_dataset(qa_data: List[Dict], system_message: str, tokenizer, max_length: int = 2048):
    from datasets import Dataset

    qa_data = _normalize_qa_data(qa_data)
    formatted = [
        {"text": _create_chat_format(tokenizer, item["question"], item["answer"], system_message)}
        for item in qa_data
    ]
    dataset = Dataset.from_list(formatted)

    def _tok(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(_tok, batched=True, remove_columns=dataset.column_names)


def _load_qa_data(dataset_url: str, inline: Optional[List[Dict]]) -> List[Dict]:
    """qa_data 확보: 인라인 우선, 없으면 dataset_url(http json) 다운로드."""
    if inline:
        return inline
    if not dataset_url:
        raise ValueError("dataset_url 또는 qa_data 가 필요합니다.")

    import requests

    logger.info("데이터셋 다운로드: %s", dataset_url)
    resp = requests.get(dataset_url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # {"qa_data": [...]} 또는 [...] 형태 모두 허용
    qa = data.get("qa_data", data) if isinstance(data, dict) else data
    if not isinstance(qa, list) or not qa:
        raise ValueError("데이터셋이 비어있거나 형식이 올바르지 않습니다.")
    return qa


# ---------------------------------------------------------------------------
# 파인튜닝 함수
# ---------------------------------------------------------------------------
@app.function(
    gpu="A10G",
    image=image,
    volumes={MODELS_DIR: volume},
    timeout=7200,  # 학습은 길어질 수 있음 (최대 2시간)
)
def run_finetuning(payload: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    from datasets import Dataset  # noqa: F401  (간접 의존, 명시)
    from huggingface_hub import HfApi, create_repo
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    influencer_id = payload.get("influencer_id", "")
    hf_token = payload.get("hf_token")
    if not hf_token:
        raise ValueError("필수 필드 누락: hf_token")

    base_model = payload.get("base_model") or DEFAULT_MODEL
    system_message = payload.get("system_message") or DEFAULT_SYSTEM_MESSAGE
    qa_data = _load_qa_data(payload.get("dataset_url", ""), payload.get("qa_data"))

    # adapter_repo 결정: 명시값 우선, 아니면 HF 계정 + influencer_id 로 생성
    hf_repo_id = payload.get("hf_repo_id")
    if not hf_repo_id:
        api = HfApi(token=hf_token)
        who = api.whoami()
        user = who.get("name") or who.get("email", "user").split("@")[0]
        hf_repo_id = f"{user}/aimex-lora-{influencer_id or 'adapter'}"

    epochs = int(payload.get("training_epochs", 3))
    batch_size = int(payload.get("batch_size", 1))
    learning_rate = float(payload.get("learning_rate", 3e-4))
    lora_r = int(payload.get("lora_r", 32))
    lora_alpha = int(payload.get("lora_alpha", 64))
    lora_dropout = float(payload.get("lora_dropout", 0.0))
    grad_accum = int(payload.get("gradient_accumulation_steps", 8))
    warmup_steps = int(payload.get("warmup_steps", 10))
    save_steps = int(payload.get("save_steps", 50))
    logging_steps = int(payload.get("logging_steps", 10))
    max_grad_norm = float(payload.get("max_grad_norm", 0.3))

    logger.info("파인튜닝 시작: base=%s repo=%s qa=%d", base_model, hf_repo_id, len(qa_data))

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "finetuned_model")
        os.makedirs(output_dir, exist_ok=True)

        # 1) 모델/토크나이저
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 2) LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=_find_all_linear_names(model),
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        # 3) 데이터셋 (90:10 split)
        tokenized = _prepare_dataset(qa_data, system_message, tokenizer)
        split = tokenized.train_test_split(test_size=0.1, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]

        # 4) TrainingArguments (eval_strategy 버전 호환)
        training_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "warmup_steps": warmup_steps,
            "save_steps": save_steps,
            "eval_steps": save_steps,
            "logging_steps": logging_steps,
            "learning_rate": learning_rate,
            "weight_decay": 0.001,
            "fp16": False,
            "bf16": True,
            "max_grad_norm": max_grad_norm,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "loss",
            "greater_is_better": False,
            "group_by_length": True,
            "report_to": ["none"],
            "remove_unused_columns": False,
        }
        try:
            training_kwargs["evaluation_strategy"] = "steps"
            training_args = TrainingArguments(**training_kwargs)
        except TypeError:
            training_kwargs.pop("evaluation_strategy", None)
            training_kwargs["eval_strategy"] = "steps"
            training_args = TrainingArguments(**training_kwargs)

        # 5) Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # 6) 학습 + 저장
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        # 7) HF 업로드
        api = HfApi()
        create_repo(repo_id=hf_repo_id, token=hf_token, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=hf_repo_id,
            token=hf_token,
            commit_message="LoRA fine-tuning via Modal",
        )

    logger.info("파인튜닝 완료: %s", hf_repo_id)
    return {"output": {"status": "completed", "adapter_repo": hf_repo_id}}


# ---------------------------------------------------------------------------
# HTTP 엔드포인트
# ---------------------------------------------------------------------------
@app.function(image=image, timeout=7200)
@modal.fastapi_endpoint(method="POST")
def finetune(item: Dict[str, Any]) -> Dict[str, Any]:
    body = item.get("input", item)
    try:
        # .remote() 는 학습 완료까지 블로킹. 장시간 작업은
        # run_finetuning.spawn(body) 로 비동기 전환 가능(README 참고).
        return run_finetuning.remote(body)
    except Exception as e:  # noqa: BLE001
        logger.error("파인튜닝 실패: %s", e)
        return {"output": {"status": "failed", "adapter_repo": ""}, "error": str(e)}


# ---------------------------------------------------------------------------
# 로컬 테스트: modal run vllm/modal_workers/finetuning_app.py
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    # QA_DATA_FILE 환경변수가 있으면 로컬 JSON 파일에서 qa_data 로드,
    # 없으면 데모 2개 사용. (실제 캐릭터 파인튜닝: build_character_qa.py 결과 사용)
    qa_file = os.environ.get("QA_DATA_FILE")
    if qa_file:
        with open(qa_file, encoding="utf-8") as f:
            qa_data = json.load(f)
    else:
        qa_data = [
            {"question": "안녕?", "answer": "안녕하세요! 반가워요."},
            {"question": "이름이 뭐야?", "answer": "저는 데모 인플루언서예요."},
        ]

    payload = {
        "influencer_id": os.environ.get("INFLUENCER_ID", "demo"),
        "hf_token": os.environ.get("HF_TOKEN", ""),
        "hf_repo_id": os.environ.get("HF_REPO_ID"),
        "base_model": DEFAULT_MODEL,
        "system_message": os.environ.get("SYSTEM_MESSAGE", DEFAULT_SYSTEM_MESSAGE),
        "qa_data": qa_data,
        "training_epochs": int(os.environ.get("TRAINING_EPOCHS", "3")),
    }
    print(f"파인튜닝 시작: qa={len(qa_data)}개, repo={payload['hf_repo_id']}")
    result = run_finetuning.remote(payload)
    print(result)
