# Gemini RAG Microservice

간단한 FastAPI 기반 마이크로서비스로, NestJS 백엔드나 다른 클라이언트가 호출해 질문을 전달하면 Google Gemini 기반 RAG 파이프라인이 답변을 만들어 줍니다. 핵심 기능은 `main.py` FastAPI 앱과 `core/rag_pipeline.py`에 정의된 보안 필터/벡터 검색 파이프라인입니다.

## 구성 요소
- `main.py` – `/chat`, `/health`, 루트 HTML UI 등을 제공하는 FastAPI 진입점.
- `core/rag_pipeline.py` – 질문 검증, 벡터 검색, Gemini 호출을 포함한 RAG 로직.
- `core/llm_utils.py` – Google Generative AI SDK 클라이언트, 임베딩/답변 생성 유틸.
- `core/db.py`, `core/models.py`, `core/db_loader.py` – SQLAlchemy 세션과 포트폴리오/게시글 모델, DB → 문서 변환 헬퍼.
- `init_post_embeddings.py` – 기존 게시글(Post) 레코드를 순회하며 `PostEmbedding` 테이블을 채우는 스크립트.
- `Procfile` – 배포 시 `uvicorn` 실행 명령 예시.

## 필수 환경 변수
`.env` 파일에 아래 키를 정의해야 합니다.

| 변수 | 설명 |
| ---- | ---- |
| `GOOGLE_API_KEY` | Google Generative AI API 키. |
| `GEMINI_MODEL` (선택) | 사용할 Gemini 모델명. 기본값 `gemini-2.0-flash`. |
| `EMBEDDING_MODEL` (선택) | 임베딩 모델명. 기본값 `text-embedding-004`. |
| `DATABASE_URL` | PostgreSQL 등 SQLAlchemy가 접근할 DB URL. |

`.gitignore`에 `.env`가 등록되어 있으므로 실수로 커밋되지 않습니다.

## 설치 및 실행
```bash
python -m venv venv
venv/Scripts/activate        # Windows PowerShell
pip install -r requirements.txt
python main.py
```

서버는 기본적으로 `http://127.0.0.1:8000`에서 실행되며:
- `GET /` – 간단한 HTML 채팅 UI.
- `GET /health` – 헬스 체크.
- `POST /chat` – JSON `{ "question": "..." }`를 보내면 `answer`/`sources`가 반환됩니다.

## 데이터 로딩 / 임베딩
`core/rag_pipeline.py`의 벡터 스토어는 기본적으로 예제 문서를 하드코딩해 초기화합니다. 실데이터를 사용하려면 아래 단계를 고려하세요.

1. `core/db_loader.py`의 `load_portfolios`, `load_posts`를 참고해 DB 연결을 준비합니다.
2. `init_post_embeddings.py`를 실행하면 `PostEmbedding` 테이블에 기존 게시글 임베딩을 채울 수 있습니다.
   ```bash
   python init_post_embeddings.py
   ```
3. 이후 RAG 초기화 구간에서 DB에서 불러온 문서를 `vector_store.add_document(...)`로 넣어주면 됩니다.

## 배포 참고
`Procfile`에는 `web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}` 형식의 명령이 정의되어 있어 Render/Heroku류 플랫폼에 바로 사용할 수 있습니다. 배포 환경에서도 `.env` 값과 DB 접근 권한을 잊지 말고 설정하세요.
