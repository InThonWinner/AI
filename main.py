# main.py

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from core.rag_pipeline import (
    generate_rag_response,
    get_context_from_db,
    CHROMA_PATH,
)
from core.llm_utils import client as gemini_client

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 로직
    print("--- FastAPI 애플리케이션 시작 ---")

    # LLM 클라이언트 준비 상태 확인
    if gemini_client is None:
        print("Gemini 키 로드 실패")
    else:
        print("Gemini 클라이언트 로드 성공")

    # ChromaDB 경로 확인
    if not os.path.exists(CHROMA_PATH):
        print(f"ChromaDB 데이터 폴더('{CHROMA_PATH}')가 없습니다.")
        print("RAG 기능을 사용하려면 'python ingest_db.py'를 먼저 실행하세요.")

    yield

    # 종료 로직
    print("--- FastAPI 애플리케이션 종료 ---")
    # TODO: DB 연결 해제 등 정리 작업이 필요하다면 여기에 추가


app = FastAPI(
    title="Gemini RAG Microservice",
    description="NestJS에서 호출할 LLM 처리 백엔드 마이크로서비스입니다.",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """
    간단한 브라우저용 UI. 입력한 질문을 /chat 엔드포인트로 전송한다.
    """
    return """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8" />
        <title>Gemini RAG Chat</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            #log { border: 1px solid #ccc; padding: 16px; height: 320px; overflow-y: auto; margin-bottom: 16px; background: #fafafa; }
            textarea { width: 100%; padding: 8px; margin-bottom: 8px; }
            button { padding: 8px 16px; cursor: pointer; }
            .msg { margin-bottom: 10px; }
            .role { font-weight: bold; margin-right: 4px; }
        </style>
    </head>
    <body>
        <h1>Gemini RAG Chat</h1>
        <div id="log"></div>
        <textarea id="question" rows="3" placeholder="질문을 입력하세요"></textarea>
        <button onclick="sendQuestion()">전송</button>

        <script>
            async function sendQuestion() {
                const textarea = document.getElementById('question');
                const question = textarea.value.trim();
                if (!question) return;
                textarea.value = '';
                addMessage('사용자', question);
                try {
                    const resp = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });
                    if (!resp.ok) {
                        const err = await resp.json().catch(() => ({}));
                        throw new Error(err.detail || '서버 오류');
                    }
                    const data = await resp.json();
                    addMessage('챗봇', data.answer || '(응답 없음)');
                } catch (err) {
                    addMessage('시스템', '오류: ' + err.message);
                }
            }

            function addMessage(role, text) {
                const log = document.getElementById('log');
                const div = document.createElement('div');
                div.className = 'msg';
                div.innerHTML = `<span class="role">${role}:</span> ${text}`;
                log.appendChild(div);
                log.scrollTop = log.scrollHeight;
            }
        </script>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """API 헬스 체크 엔드포인트."""
    return {"status": "OK"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """브라우저가 요청하는 기본 favicon을 204로 응답."""
    return Response(status_code=204)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    사용자의 질문을 받아 RAG 파이프라인을 수행하고 답변을 반환한다.
    """
    try:
        context = get_context_from_db(req.question)
        answer_text = generate_rag_response(req.question, context)

        if context.strip():
            sources = ["ChromaDB 검색 결과"]
        else:
            sources = []

        return ChatResponse(answer=answer_text, sources=sources)

    except Exception as e:
        print(f"RAG 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail="RAG가 정상적으로 동작하지 않습니다.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
