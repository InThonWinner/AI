# init_post_embeddings.py
"""
기존 Post 테이블에 있는 글들을 읽어서
각 Post에 대한 임베딩을 PostEmbedding 테이블에 채워 넣는 스크립트.

사용법 (프로젝트 루트에서):
    $ python init_post_embeddings.py
    한 번만 실행하면 됨. 실행할 때 마다 새로 갱신된 포트폴리오, 게시물이 있으면 임베딩 생성.

    추가 목표: Nestjs 백엔드 서버에서 새로운 글이 추가될 때마다 자동으로 임베딩을 생성하도록 구현.
"""

from sqlalchemy.orm import Session

from core.db import SessionLocal
from core.models import Post, PostEmbedding
from core.llm_utils import get_embedding
from datetime import datetime, timezone

def init_embeddings() -> None:
    db: Session = SessionLocal()
    try:
        posts = db.query(Post).all()
        print(f"총 {len(posts)}개의 Post를 찾았습니다.")

        created = 0
        skipped = 0

        for post in posts:
            existing = (
                db.query(PostEmbedding)
                .filter(PostEmbedding.postId == post.id)
                .first()
            )
            if existing:
                skipped += 1
                continue

            parts = []
            if post.title:
                parts.append(post.title)
            if post.content:
                parts.append(post.content)

            full_text = "\n\n".join(parts).strip()
            if not full_text:
                skipped += 1
                continue

            emb_list = get_embedding(full_text)

            now = datetime.now(timezone.utc)
            pe = PostEmbedding(
                postId=post.id,
                embedding=emb_list,
                createdAt=now,
                updatedAt=now,
            )
            db.add(pe)
            created += 1

            if created % 20 == 0:
                db.flush()
                print(f"{created}개 임베딩 생성 중...")

        db.commit()
        print(f"완료! 새로 생성: {created}개, 스킵: {skipped}개")

    finally:
        db.close()


if __name__ == "__main__":
    init_embeddings()
