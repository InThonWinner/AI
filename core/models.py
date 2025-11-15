# reg-service/core/models.py

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from .db import Base

# DisplayNameType, PostCategory는 DB에서 Enum일 수도 있지만,
# 파이썬 쪽에서는 일단 String 컬럼으로 매핑해두고 사용해도 됨.
# (나중에 실제 Enum 값들 알게 되면 sqlalchemy.Enum으로 바꿔도 됨.)

class Portfolio(Base):
    __tablename__ = "Portfolio" 

    id               = Column(Integer, primary_key=True, index=True)
    userId           = Column(Integer, unique=True, nullable=False, index=True)

    # DisplayNameType [default: 'NICKNAME']
    displayNameType  = Column(String, nullable=False, default="NICKNAME")

    # contactPublic Boolean [default: false]
    contactPublic     = Column(Boolean, nullable=False, default=False)

    # affiliationPublic Boolean [default: false]
    affiliationPublic = Column(Boolean, nullable=False, default=False)

    # techStack, career, projects, activitiesAwards: String (nullable 허용)
    techStack        = Column(Text, nullable=True)
    career           = Column(Text, nullable=True)
    projects         = Column(Text, nullable=True)
    activitiesAwards = Column(Text, nullable=True)

    # 공개 여부 플래그들
    showTechStack        = Column(Boolean, nullable=False, default=True)
    showCareer           = Column(Boolean, nullable=False, default=True)
    showProjects         = Column(Boolean, nullable=False, default=True)
    showActivitiesAwards = Column(Boolean, nullable=False, default=True)

    # createdAt DateTime [default: now()]
    createdAt        = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # updatedAt DateTime [default: now()]
    # DB 기본값 + 업데이트 시점에 서버에서 갱신되도록 onupdate 설정
    updatedAt        = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Post(Base):
    __tablename__ = "Post"

    id          = Column(Integer, primary_key=True, index=True)
    authorId    = Column(Integer, nullable=False, index=True)

    # PostCategory [not null]
    category    = Column(String, nullable=False, index=True)

    title       = Column(String, nullable=True)
    content     = Column(Text, nullable=False)

    isAnonymous = Column(Boolean, nullable=False, default=True)

    createdAt   = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updatedAt   = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )