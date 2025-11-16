# core/db_loader.py

"""
Post + Portfolio를 읽어서 RAG용 임베딩(PostEmbedding)에 저장하는 스크립트.
처음 한 번 전체 인덱싱할 때 또는 주기적으로 돌리면 됨.
"""

from typing import Optional

from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import Post, Portfolio, PostEmbedding
from .llm_utils import get_embedding


def build_doc_text(post: Post, portfolio: Optional[Portfolio]) -> str:
    """
    Post 한 개 + (선택) Portfolio를 합쳐서 RAG용 문서 텍스트로 만든다.
    Portfolio 공개 플래그(showXXX)에 따라 붙일지 말지 결정.
    """
    lines: list[str] = []

    lines.append(f"[카테고리] {post.category}")
    if post.title:
        lines.append(f"[제목] {post.title}")
    lines.append("[본문]")
    lines.append((post.content or "").strip())
    lines.append("")

    if portfolio is not None:
        lines.append("[작성자 포트폴리오]")

        if portfolio.showTechStack and portfolio.techStack:
            lines.append(f"- 기술 스택: {portfolio.techStack}")

        if portfolio.showCareer and portfolio.career:
            lines.append(f"- 커리어/진로: {portfolio.career}")

        if portfolio.showProjects and portfolio.projects:
            lines.append(f"- 프로젝트: {portfolio.projects}")

        if portfolio.showActivitiesAwards and portfolio.activitiesAwards:
            lines.append(f"- 활동/수상: {portfolio.activitiesAwards}")

        if portfolio.showAffiliation and portfolio.affiliation:
            lines.append(f"- 소속: {portfolio.affiliation}")

        if portfolio.showContact and portfolio.contact:
            lines.append(f"- 연락처: {portfolio.contact}")

    return "\n".join(lines)


def index_all_posts() -> None:
    """
    모든 Post를 대상으로 임베딩을 만들어 PostEmbedding 테이블에 upsert.
    """
    db: Session = SessionLocal()

    try:
        posts: list[Post] = db.query(Post).all()

        for post in posts:
            portfolio: Optional[Portfolio] = (
                db.query(Portfolio)
                .filter(Portfolio.userId == post.authorId)
                .one_or_none()
            )

            doc_text = build_doc_text(post, portfolio)
            emb = get_embedding(doc_text)

            existing = (
                db.query(PostEmbedding)
                .filter(PostEmbedding.postId == post.id)
                .one_or_none()
            )

            if existing is None:
                pe = PostEmbedding(postId=post.id, embedding=emb)
                db.add(pe)
            else:
                existing.embedding = emb

        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    # 터미널에서 직접 실행할 때:
    #   python -m core.db_loader
    index_all_posts()
    print("모든 Post 임베딩 인덱싱 완료")
