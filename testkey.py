# test_key.py
import os
from dotenv import load_dotenv

from google import genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("GOOGLE_API_KEY 환경변수가 없습니다. .env 파일을 확인하세요.")
    exit(1)

client = genai.Client(api_key=API_KEY)

try:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hello! This is a key test.",
    )
    print("API Key 테스트 성공")
    print("모델 응답:", response.text)
except Exception as e:
    print("API Key 테스트 실패!")
    print(e)
