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
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"

app = FastAPI(title="MaGary AI")


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
    {
        "id": "openrouter/free",
        "name": "OpenRouter Free",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "deepseek/deepseek-v4-flash:free",
        "name": "DeepSeek V4 Flash Free",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "openrouter/owl-alpha",
        "name": "Owl Alpha Free",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "poolside/laguna-xs.2:free",
        "name": "Poolside Laguna XS.2 Free",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "poolside/laguna-m.1:free",
        "name": "Poolside Laguna M.1 Free",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "name": "NVIDIA Nemotron 3 Free",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "deepseek/deepseek-v4-flash",
        "name": "DeepSeek V4 Flash",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "deepseek/deepseek-v4-pro",
        "name": "DeepSeek V4 Pro",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "qwen/qwen3.7-max",
        "name": "Qwen3.7 Max",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "qwen/qwen3.6-flash",
        "name": "Qwen3.6 Flash",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "google/gemini-3.5-flash",
        "name": "Gemini 3.5 Flash",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "google/gemini-3.1-flash-lite",
        "name": "Gemini 3.1 Flash Lite",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "~google/gemini-flash-latest",
        "name": "Gemini Flash Latest",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "openai/gpt-chat-latest",
        "name": "GPT Chat Latest",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "~openai/gpt-mini-latest",
        "name": "GPT Mini Latest",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "anthropic/claude-opus-4.8",
        "name": "Claude Opus 4.8",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "~anthropic/claude-haiku-latest",
        "name": "Claude Haiku Latest",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "~anthropic/claude-sonnet-latest",
        "name": "Claude Sonnet Latest",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },
    {
        "id": "x-ai/grok-4.3",
        "name": "Grok 4.3",
        "provider": "OpenRouter",
        "channel": "OpenRouter",
    },

    # Cerebras models
    {
        "id": "cerebras/gpt-oss-120b",
        "name": "Cerebras GPT OSS 120B",
        "provider": "Cerebras",
        "channel": "Cerebras",
        "real_id": "gpt-oss-120b",
    },
    {
        "id": "cerebras/zai-glm-4.7",
        "name": "Cerebras ZAI GLM 4.7",
        "provider": "Cerebras",
        "channel": "Cerebras",
        "real_id": "zai-glm-4.7",
    },
]


def password_required() -> bool:
    return bool(APP_PASSWORD)


def check_password(password: str) -> bool:
    if not APP_PASSWORD:
        return True
    return password == APP_PASSWORD


def find_model(model_id: str) -> dict:
    for model in MODELS:
        if model["id"] == model_id:
            return model

    return MODELS[0]


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "password_required": password_required(),
        "has_openrouter_key": bool(OPENROUTER_API_KEY),
        "has_cerebras_key": bool(CEREBRAS_API_KEY),
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
        content={
            "ok": False,
            "error": "访问密码错误"
        }
    )


@app.get("/models")
def get_models():
    return {
        "models": MODELS
    }


@app.get("/debug-key")
def debug_key():
    return {
        "has_openrouter_key": bool(OPENROUTER_API_KEY),
        "openrouter_key_start": OPENROUTER_API_KEY[:10] if OPENROUTER_API_KEY else None,
        "openrouter_key_length": len(OPENROUTER_API_KEY) if OPENROUTER_API_KEY else 0,

        "has_cerebras_key": bool(CEREBRAS_API_KEY),
        "cerebras_key_start": CEREBRAS_API_KEY[:10] if CEREBRAS_API_KEY else None,
        "cerebras_key_length": len(CEREBRAS_API_KEY) if CEREBRAS_API_KEY else 0,

        "has_app_password": bool(APP_PASSWORD),
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if not check_password(req.password):
        return JSONResponse(
            status_code=401,
            content={
                "error": "访问密码错误"
            }
        )

    user_message = req.message.strip()

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={
                "error": "消息不能为空"
            }
        )

    safe_max_tokens = min(max(int(req.max_tokens or 1024), 128), 2048)

    selected_model = find_model(req.model)
    provider = selected_model.get("provider", "OpenRouter").lower()

    system_message = {
        "role": "system",
        "content": "你是 MaGary AI，是马广辉的中文 AI 助手。请用自然、清楚、适当幽默的中文回答。"
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
            "content": user_message
        }
    ]

    if provider == "cerebras":
        if not CEREBRAS_API_KEY:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "CEREBRAS_API_KEY 未配置，请检查 Render 环境变量"
                }
            )

        api_url = CEREBRAS_URL
        api_key = CEREBRAS_API_KEY
        model_name = selected_model.get("real_id", selected_model["id"])

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    else:
        if not OPENROUTER_API_KEY:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "OPENROUTER_API_KEY 未配置，请检查 Render 环境变量"
                }
            )

        api_url = OPENROUTER_URL
        api_key = OPENROUTER_API_KEY
        model_name = selected_model["id"]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://guanghui-ai-chat.onrender.com",
            "X-Title": "MaGary AI",
        }

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": safe_max_tokens,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=90,
        )

        try:
            result = resp.json()
        except Exception:
            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": "模型接口返回了非 JSON 内容",
                    "raw": resp.text,
                    "provider": selected_model.get("provider"),
                    "model": selected_model.get("id"),
                }
            )

        if resp.status_code != 200:
            error_obj = result.get("error")

            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = error_obj or "模型接口请求失败"

            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": error_msg,
                    "status_code": resp.status_code,
                    "raw": result,
                    "provider": selected_model.get("provider"),
                    "model": selected_model.get("id"),
                }
            )

        reply = ""

        if "choices" in result and result["choices"]:
            choice = result["choices"][0]

            if "message" in choice and isinstance(choice["message"], dict):
                reply = choice["message"].get("content", "")

            if not reply:
                reply = choice.get("text", "")

        if not reply:
            reply = "模型返回了空内容。"

        return {
            "reply": reply,
            "model": selected_model.get("id"),
            "real_model": model_name,
            "provider": selected_model.get("provider"),
            "channel": selected_model.get("channel"),
        }

    except requests.exceptions.Timeout:
        return JSONResponse(
            status_code=504,
            content={
                "error": "请求超时，请换个模型或稍后再试",
                "provider": selected_model.get("provider"),
                "model": selected_model.get("id"),
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "provider": selected_model.get("provider"),
                "model": selected_model.get("id"),
            }
        )