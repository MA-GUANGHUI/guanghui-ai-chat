from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI(title="GuangHui AI Chat")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    model: str = "deepseek/deepseek-v4-flash:free"
    history: Optional[List[ChatMessage]] = []
    max_tokens: int = 1024


MODELS = [
    {"id": "deepseek/deepseek-v4-flash:free", "name": "DeepSeek V4 Flash Free"},
    {"id": "openrouter/free", "name": "OpenRouter Free"},
    {"id": "openrouter/owl-alpha", "name": "Owl Alpha Free"},
    {"id": "poolside/laguna-xs.2:free", "name": "Poolside Laguna XS.2 Free"},
    {"id": "poolside/laguna-m.1:free", "name": "Poolside Laguna M.1 Free"},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "name": "NVIDIA Nemotron 3 Free"},

    {"id": "deepseek/deepseek-v4-flash", "name": "DeepSeek V4 Flash"},
    {"id": "deepseek/deepseek-v4-pro", "name": "DeepSeek V4 Pro"},
    {"id": "qwen/qwen3.7-max", "name": "Qwen3.7 Max"},
    {"id": "qwen/qwen3.6-flash", "name": "Qwen3.6 Flash"},
    {"id": "google/gemini-3.5-flash", "name": "Gemini 3.5 Flash"},
    {"id": "google/gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite"},
    {"id": "~google/gemini-flash-latest", "name": "Gemini Flash Latest"},
    {"id": "openai/gpt-chat-latest", "name": "GPT Chat Latest"},
    {"id": "~openai/gpt-mini-latest", "name": "GPT Mini Latest"},
    {"id": "anthropic/claude-opus-4.8", "name": "Claude Opus 4.8"},
    {"id": "~anthropic/claude-haiku-latest", "name": "Claude Haiku Latest"},
    {"id": "~anthropic/claude-sonnet-latest", "name": "Claude Sonnet Latest"},
    {"id": "x-ai/grok-4.3", "name": "Grok 4.3"},
]


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    return {"models": MODELS}


@app.post("/chat")
def chat(req: ChatRequest):
    if not OPENROUTER_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "OPENROUTER_API_KEY 未配置，请检查 .env 文件"}
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://127.0.0.1:8000",
        "X-Title": "GuangHui AI Chat"
    }

    system_message = {
        "role": "system",
        "content": "你是 GuangHui AI Chat 的中文助手。请用简洁、清楚、自然的中文回答。"
    }

    history = []
    if req.history:
        for item in req.history[-12:]:
            if item.role in ["user", "assistant"] and item.content.strip():
                history.append({
                    "role": item.role,
                    "content": item.content
                })

    messages = [
        system_message,
        *history,
        {
            "role": "user",
            "content": req.message
        }
    ]

    payload = {
        "model": req.model,
        "messages": messages,
        "max_tokens": min(req.max_tokens, 2048),
        "temperature": 0.7
    }

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=90
        )

        result = resp.json()

        if resp.status_code != 200:
            error_msg = result.get("error", {}).get("message", str(result))
            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": error_msg,
                    "raw": result
                }
            )

        reply = result["choices"][0]["message"]["content"]
        used_model = result.get("model", req.model)
        provider = result.get("provider", "")

        return {
            "reply": reply,
            "model": used_model,
            "provider": provider
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )