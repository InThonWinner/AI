# reg-service/core/db.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# .env 읽기
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise RuntimeError("DATABASE_URL 환경변수가 설정되어 있지 않습니다.")

# SQLAlchemy 엔진 생성
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# 세션 팩토리
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스
Base = declarative_base()
