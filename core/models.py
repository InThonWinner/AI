# core/models.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, JSON
from sqlalchemy.sql import func
from .db import Base


class Portfolio(Base):
    __tablename__ = "Portfolio"

    id      = Column(Integer, primary_key=True, index=True)
    userId  = Column(Integer, unique=True, nullable=False, index=True)

    # "NICKNAME" / "REALNAME" 등 문자열로 저장된 타입
    displayNameType = Column(String, nullable=False, default="NICKNAME")

    # 원본 JSON에 있는 그대로의 필드들
    contact     = Column(Text, nullable=True)   # 연락처
    affiliation = Column(Text, nullable=True)   # 소속
    techStack   = Column(Text, nullable=True)   # 기술 스택
    career      = Column(Text, nullable=True)   # 커리어/진로
    projects    = Column(Text, nullable=True)   # 프로젝트
    activitiesAwards = Column(Text, nullable=True)  # 활동/수상

    # 공개 여부 플래그들 (JSON: showContact, showAffiliation, ...)
    showContact          = Column(Boolean, nullable=False, default=False)
    showAffiliation      = Column(Boolean, nullable=False, default=False)
    showTechStack        = Column(Boolean, nullable=False, default=True)
    showCareer           = Column(Boolean, nullable=False, default=True)
    showProjects         = Column(Boolean, nullable=False, default=True)
    showActivitiesAwards = Column(Boolean, nullable=False, default=True)

    createdAt = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updatedAt = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Post(Base):
    __tablename__ = "Post"

    id       = Column(Integer, primary_key=True, index=True)
    authorId = Column(Integer, nullable=False, index=True)

    # STUDY_PATH / COURSE / PROJECT / CAREER / ETC
    category = Column(String, nullable=False, index=True)

    title       = Column(String, nullable=True)
    content     = Column(Text, nullable=False)
    isAnonymous = Column(Boolean, nullable=False, default=True)

    createdAt = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updatedAt = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class PostEmbedding(Base):
    """
    각 Post에 대한 임베딩을 저장하는 테이블.
    embedding은 JSON 배열(List[float])로 저장.
    """
    __tablename__ = "PostEmbedding"

    id     = Column(Integer, primary_key=True, index=True)
    postId = Column(Integer, nullable=False, unique=True, index=True)

    # LLM에서 받은 임베딩 벡터 (List[float])
    embedding = Column(JSON, nullable=False)

    createdAt = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updatedAt = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
