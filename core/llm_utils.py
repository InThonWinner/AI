# core/llm_utils.py

"""
LLM 유틸리티 함수
- Gemini 기반 임베딩 생성
- 답변 생성 (RAG)
- Guard LLM (보안/필터링용)
"""

import os
import time
import logging
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types  # 필요 없으면 삭제해도 됨

logger = logging.getLogger(__name__)

# ===== 환경변수 & Gemini 클라이언트 설정 =====
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY가 설정되어 있지 않습니다. .env를 확인하세요.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")          # 하드코딩: 기본 답변 생성 모델
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")  # 하드코딩: 기본 임베딩 모델
GUARD_MODEL = os.getenv("GUARD_MODEL", "gemini-2.0-flash-lite")       # 하드코딩: Guard LLM용 경량 모델
EMBEDDING_DIM = 768  # 하드코딩: text-embedding-004의 벡터 차원 수

# Gemini 클라이언트 생성
client = genai.Client(api_key=API_KEY)
GEMINI_AVAILABLE = True


def get_embedding(text: str) -> List[float]:
    """
    주어진 문자열을 임베딩 벡터(리스트[float])로 변환.
    RAG에서 벡터 DB에 넣거나, 유사도 계산할 때 사용.
    """
    # 빈 텍스트면 zero vector 반환 (벡터 차원은 EMBEDDING_DIM으로 고정)
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM

    try:
        resp = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        # (단일 텍스트 입력이므로 embeddings[0] 사용)
        embedding = resp.embeddings[0].values  # type: ignore[attr-defined]
        return list(embedding)
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        # 에러 시 zero vector 반환
        return [0.0] * EMBEDDING_DIM


def generate_answer(question: str, context: str = "", max_retries: int = 3) -> str:
    """
    메인 LLM으로 답변 생성 (RAG + Gemini).
    - question: 사용자의 질문
    - context: 벡터 검색 등으로 찾은 관련 문맥 텍스트
    """
    if not GEMINI_AVAILABLE:
        return "LLM API가 설정되지 않았습니다. 환경변수를 확인해주세요."

    system_instruction = (
        "You are a helpful mentor for university CS students. "
        "Use the given context to answer the question. "
        "If the context is not enough, say you are not sure instead of making things up."
    )

    contents = [
        f"[SYSTEM]\n{system_instruction}",
        f"[CONTEXT]\n{context}",
        f"[QUESTION]\n{question}",
    ]

    # 재시도 로직 (Rate limit / 일시적인 오류 대응)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
            )
            return resp.text or ""
        except Exception as e:
            logger.error(f"LLM generation error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
            else:
                return "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    return "답변 생성 중 오류가 발생했습니다."


def call_guard_llm(prompt: str) -> str:
    """
    Guard LLM 호출 (보안 검증/필터링용).
    - 가능한 한 JSON 문자열을 반환하도록 프롬프트 설계.
    - 예: {"is_malicious": false, "reason": "...", "confidence": 0.0}
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Guard LLM not available, skipping check")
        return '{"is_malicious": false, "reason": "Guard LLM unavailable", "confidence": 0.0}'

    guard_instruction = (
        "You are a security expert. "
        "Analyze the given text and respond ONLY in strict JSON with keys: "
        '"is_malicious" (boolean), "reason" (string), "confidence" (float between 0 and 1).'
    )

    contents = [
        f"[SYSTEM]\n{guard_instruction}",
        f"[INPUT]\n{prompt}",
    ]

    try:
        resp = client.models.generate_content(
            model=GUARD_MODEL,
            contents=contents,
        )
        # 모델이 JSON 문자열을 돌려준다는 가정
        return resp.text or '{"is_malicious": false, "reason": "Empty response", "confidence": 0.0}'
    except Exception as e:
        logger.error(f"Guard LLM error: {e}")
        # 에러 시 보수적으로 통과 (서비스 가용성 우선)
        return '{"is_malicious": false, "reason": "Guard LLM error", "confidence": 0.0}'


def check_api_status() -> dict:
    """
    API 연결 상태 확인용 헬퍼.
    헬스체크 엔드포인트 등에서 사용 가능.
    """
    status = {
        "gemini_available": GEMINI_AVAILABLE,
        "embedding_ready": GEMINI_AVAILABLE,
        "generation_ready": GEMINI_AVAILABLE,
        "guard_ready": GEMINI_AVAILABLE,
        "gemini_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "guard_model": GUARD_MODEL,
    }
    return status
import os
print("DEBUG >>> GOOGLE_API_KEY 존재?", bool(os.getenv("GOOGLE_API_KEY")))
