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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI(title="GuangHui AI Chat")


class AuthRequest(BaseModel):
    password: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    model: str = "openrouter/free"
    history: Optional[List[ChatMessage]] = []
    max_tokens: int = 1024
    password: str = ""


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


def password_required() -> bool:
    return bool(APP_PASSWORD)


def check_password(password: str) -> bool:
    if not APP_PASSWORD:
        return True
    return password == APP_PASSWORD


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "password_required": password_required()
    }


@app.get("/auth/status")
def auth_status():
    return {
        "password_required": password_required()
    }


@app.post("/auth")
def auth(req: AuthRequest):
    if check_password(req.password):
        return {"ok": True}

    return JSONResponse(
        status_code=401,
        content={"ok": False, "error": "访问密码错误"}
    )


@app.get("/models")
def get_models():
    return {"models": MODELS}


@app.get("/debug-key")
def debug_key():
    return {
        "has_openrouter_key": bool(OPENROUTER_API_KEY),
        "key_start": OPENROUTER_API_KEY[:10] if OPENROUTER_API_KEY else None,
        "key_length": len(OPENROUTER_API_KEY) if OPENROUTER_API_KEY else 0,
        "has_app_password": bool(APP_PASSWORD)
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if not check_password(req.password):
        return JSONResponse(
            status_code=401,
            content={"error": "访问密码错误"}
        )

    if not OPENROUTER_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": "OPENROUTER_API_KEY 未配置，请检查 Render 环境变量"}
        )

    user_message = req.message.strip()
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"error": "消息不能为空"}
        )

    safe_max_tokens = min(max(int(req.max_tokens or 1024), 128), 2048)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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

    payload = {
        "model": req.model or "openrouter/free",
        "messages": [
            system_message,
            *history,
            {
                "role": "user",
                "content": user_message
            }
        ],
        "max_tokens": safe_max_tokens,
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
            error_obj = result.get("error")

            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = error_obj or "OpenRouter 请求失败"

            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": error_msg,
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