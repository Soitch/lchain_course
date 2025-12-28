import os
from dotenv import load_dotenv, find_dotenv
import httpx
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

MODEL = os.getenv("OPENAI_MODEL", "hosted_vllm/Qwen/Qwen3-Coder-480B-A35B-Instruct")
KEY = os.getenv("OPENAI_API_KEY")
URL = os.getenv("OPENAI_API_BASE", "https://foundation-models.api.cloud.ru/v1")

# Отключаем SSL-проверку (если нужно, но лучше избегать в продакшене)
http_client = httpx.Client(verify=False)
MODELCA = ChatOpenAI(model=MODEL, api_key=KEY, base_url=URL, http_client=http_client)