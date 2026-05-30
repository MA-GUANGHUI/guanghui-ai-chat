from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI(title="GuangHui AI Chat")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    model: str = "openrouter/free"
    history: Optional[List[ChatMessage]] = []
    max_tokens: int = 1024


MODELS = [
    {"id": "openrouter/free", "name": "OpenRouter Free"},
    {"id": "deepseek/deepseek-v4-flash:free", "name": "DeepSeek V4 Flash Free"},
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


def get_openrouter_key():
    key = os.getenv("OPENROUTER_API_KEY", "")
    return key.strip()


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    return {"models": MODELS}


@app.get("/debug-key")
def debug_key():
    key = get_openrouter_key()

    return {
        "has_key": bool(key),
        "key_start": key[:10] if key else None,
        "key_length": len(key) if key else 0
    }


@app.get("/test-openrouter")
def test_openrouter():
    key = get_openrouter_key()

    if not key:
        return JSONResponse(
            status_code=500,
            content={"error": "OPENROUTER_API_KEY 未读取到"}
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://guanghui-ai-chat.onrender.com",
        "X-Title": "GuangHui AI Chat"
    }

    payload = {
        "model": "openrouter/free",
        "messages": [
            {"role": "user", "content": "hello"}
        ],
        "max_tokens": 64
    }

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        return {
            "status_code": resp.status_code,
            "response": resp.json()
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/chat")
def chat(req: ChatRequest):
    key = get_openrouter_key()

    if not key:
        return JSONResponse(
            status_code=500,
            content={
                "error": "OPENROUTER_API_KEY 未配置。请检查 Render Environment Variables。"
            }
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://guanghui-ai-chat.onrender.com",
        "X-Title": "GuangHui AI Chat"
    }

    system_message = {
        "role": "system",
        "content": "你是 GuangHui AI Chat 的中文助手。请用简洁、清楚、自然的中文回答。"
    }

    history = []
    if req.history:
        for item in req.history[-10:]:
            if item.role in ["user", "assistant"] and item.content.strip():
                history.append({
                    "role": item.role,
                    "content": item.content.strip()
                })

    messages = [
        system_message,
        *history,
        {
            "role": "user",
            "content": req.message.strip()
        }
    ]

    payload = {
        "model": req.model or "openrouter/free",
        "messages": messages,
        "max_tokens": min(int(req.max_tokens or 1024), 2048),
        "temperature": 0.7
    }

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=90
        )

        try:
            result = resp.json()
        except Exception:
            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": "OpenRouter 返回了非 JSON 内容",
                    "raw": resp.text
                }
            )

        if resp.status_code != 200:
            error_msg = (
                result.get("error", {}).get("message")
                if isinstance(result.get("error"), dict)
                else result.get("error")
            )

            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": error_msg or "OpenRouter 请求失败",
                    "status_code": resp.status_code,
                    "raw": result
                }
            )

        reply = result["choices"][0]["message"]["content"]

        return {
            "reply": reply,
            "model": result.get("model", req.model),
            "provider": result.get("provider", "")
        }

    except requests.exceptions.Timeout:
        return JSONResponse(
            status_code=504,
            content={"error": "请求超时，请换个模型或稍后再试"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )