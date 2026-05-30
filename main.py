from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()

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


# ============================================================
# Provider 配置区
# 以后新增一个 OpenAI-compatible API，大多数情况只改这里：
# 1. PROVIDERS 里加 provider
# 2. MODELS 里加模型
# ============================================================
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "label": "OpenRouter",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "extra_headers": {
            "HTTP-Referer": "https://guanghui-ai-chat.onrender.com",
            "X-Title": "MaGary AI",
        },
    },
    "cerebras": {
        "label": "Cerebras",
        "api_key": CEREBRAS_API_KEY,
        "base_url": "https://api.cerebras.ai/v1",
        "extra_headers": {},
    },
    "nvidia": {
        "label": "NVIDIA",
        "api_key": NVIDIA_API_KEY,
        "base_url": "https://integrate.api.nvidia.com/v1",
        "extra_headers": {},
    },
}


# 可选：用环境变量继续扩展 Provider，不用改代码。
# EXTRA_PROVIDERS_JSON 示例：
# {
#   "groq": {
#     "label": "Groq",
#     "api_key_env": "GROQ_API_KEY",
#     "base_url": "https://api.groq.com/openai/v1"
#   }
# }
def load_extra_providers() -> None:
    raw = os.getenv("EXTRA_PROVIDERS_JSON", "").strip()
    if not raw:
        return

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return

        for key, cfg in data.items():
            if not isinstance(cfg, dict):
                continue

            provider_key = str(key).strip().lower()
            api_key_env = str(cfg.get("api_key_env", "")).strip()
            api_key = str(cfg.get("api_key", "")).strip()

            if api_key_env:
                api_key = os.getenv(api_key_env, "").strip()

            PROVIDERS[provider_key] = {
                "label": cfg.get("label", provider_key.title()),
                "api_key": api_key,
                "base_url": str(cfg.get("base_url", "")).rstrip("/"),
                "extra_headers": cfg.get("extra_headers", {}) if isinstance(cfg.get("extra_headers", {}), dict) else {},
            }

    except Exception:
        # 不让错误的 JSON 把服务拖死。配置地雷这种东西，人类已经埋得够多了。
        return


load_extra_providers()


MODELS: List[Dict[str, Any]] = [
    # OpenRouter
    {"id": "openrouter/free", "name": "OpenRouter Free", "provider": "openrouter", "real_id": "openrouter/free", "family": "Router"},
    {"id": "deepseek/deepseek-v4-flash:free", "name": "DeepSeek V4 Flash Free", "provider": "openrouter", "real_id": "deepseek/deepseek-v4-flash:free", "family": "Free"},
    {"id": "openrouter/owl-alpha", "name": "Owl Alpha Free", "provider": "openrouter", "real_id": "openrouter/owl-alpha", "family": "Free"},
    {"id": "poolside/laguna-xs.2:free", "name": "Poolside Laguna XS.2 Free", "provider": "openrouter", "real_id": "poolside/laguna-xs.2:free", "family": "Free"},
    {"id": "poolside/laguna-m.1:free", "name": "Poolside Laguna M.1 Free", "provider": "openrouter", "real_id": "poolside/laguna-m.1:free", "family": "Free"},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "name": "NVIDIA Nemotron 3 Free", "provider": "openrouter", "real_id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "family": "OpenRouter"},
    {"id": "deepseek/deepseek-v4-flash", "name": "DeepSeek V4 Flash", "provider": "openrouter", "real_id": "deepseek/deepseek-v4-flash", "family": "DeepSeek"},
    {"id": "deepseek/deepseek-v4-pro", "name": "DeepSeek V4 Pro", "provider": "openrouter", "real_id": "deepseek/deepseek-v4-pro", "family": "DeepSeek"},
    {"id": "qwen/qwen3.7-max", "name": "Qwen3.7 Max", "provider": "openrouter", "real_id": "qwen/qwen3.7-max", "family": "Qwen"},
    {"id": "qwen/qwen3.6-flash", "name": "Qwen3.6 Flash", "provider": "openrouter", "real_id": "qwen/qwen3.6-flash", "family": "Qwen"},
    {"id": "google/gemini-3.5-flash", "name": "Gemini 3.5 Flash", "provider": "openrouter", "real_id": "google/gemini-3.5-flash", "family": "Google"},
    {"id": "google/gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite", "provider": "openrouter", "real_id": "google/gemini-3.1-flash-lite", "family": "Google"},
    {"id": "~google/gemini-flash-latest", "name": "Gemini Flash Latest", "provider": "openrouter", "real_id": "~google/gemini-flash-latest", "family": "Google"},
    {"id": "openai/gpt-chat-latest", "name": "GPT Chat Latest", "provider": "openrouter", "real_id": "openai/gpt-chat-latest", "family": "OpenAI"},
    {"id": "~openai/gpt-mini-latest", "name": "GPT Mini Latest", "provider": "openrouter", "real_id": "~openai/gpt-mini-latest", "family": "OpenAI"},
    {"id": "anthropic/claude-opus-4.8", "name": "Claude Opus 4.8", "provider": "openrouter", "real_id": "anthropic/claude-opus-4.8", "family": "Anthropic"},
    {"id": "~anthropic/claude-haiku-latest", "name": "Claude Haiku Latest", "provider": "openrouter", "real_id": "~anthropic/claude-haiku-latest", "family": "Anthropic"},
    {"id": "~anthropic/claude-sonnet-latest", "name": "Claude Sonnet Latest", "provider": "openrouter", "real_id": "~anthropic/claude-sonnet-latest", "family": "Anthropic"},
    {"id": "x-ai/grok-4.3", "name": "Grok 4.3", "provider": "openrouter", "real_id": "x-ai/grok-4.3", "family": "xAI"},

    # Cerebras
    {"id": "cerebras/gpt-oss-120b", "name": "Cerebras GPT OSS 120B", "provider": "cerebras", "real_id": "gpt-oss-120b", "family": "Fast LLM"},
    {"id": "cerebras/zai-glm-4.7", "name": "Cerebras ZAI GLM 4.7", "provider": "cerebras", "real_id": "zai-glm-4.7", "family": "Fast LLM"},

    # NVIDIA NIM, direct API
    {"id": "nvidia/nemotron-3-nano-30b-a3b", "name": "NVIDIA Nemotron 3 Nano 30B", "provider": "nvidia", "real_id": "nvidia/nemotron-3-nano-30b-a3b", "family": "NIM"},
    {"id": "nvidia/nemotron-3-super-120b-a12b", "name": "NVIDIA Nemotron 3 Super 120B", "provider": "nvidia", "real_id": "nvidia/nemotron-3-super-120b-a12b", "family": "NIM"},
    {"id": "nvidia/llama-3.3-nemotron-super-49b-v1.5", "name": "NVIDIA Nemotron Super 49B", "provider": "nvidia", "real_id": "nvidia/llama-3.3-nemotron-super-49b-v1.5", "family": "NIM"},
    {"id": "nvidia/llama-3.1-nemotron-ultra-253b-v1", "name": "NVIDIA Nemotron Ultra 253B", "provider": "nvidia", "real_id": "nvidia/llama-3.1-nemotron-ultra-253b-v1", "family": "NIM"},
    {"id": "nvidia/qwen3-next-80b-a3b-instruct", "name": "NVIDIA Qwen3 Next 80B", "provider": "nvidia", "real_id": "qwen/qwen3-next-80b-a3b-instruct", "family": "NIM"},
    {"id": "nvidia/qwen2.5-coder-32b-instruct", "name": "NVIDIA Qwen2.5 Coder 32B", "provider": "nvidia", "real_id": "qwen/qwen2.5-coder-32b-instruct", "family": "Code"},
    {"id": "nvidia/kimi-k2-instruct", "name": "NVIDIA Kimi K2 Instruct", "provider": "nvidia", "real_id": "moonshotai/kimi-k2-instruct", "family": "NIM"},
]


# 可选：用环境变量继续扩展模型，不用改代码。
# EXTRA_MODELS_JSON 示例：
# [
#   {"id":"groq/llama", "name":"Groq Llama", "provider":"groq", "real_id":"llama-3.3-70b-versatile", "family":"Groq"}
# ]
def load_extra_models() -> None:
    raw = os.getenv("EXTRA_MODELS_JSON", "").strip()
    if not raw:
        return

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("id") and item.get("provider"):
                    MODELS.append(item)
    except Exception:
        return


load_extra_models()


def password_required() -> bool:
    return bool(APP_PASSWORD)


def check_password(password: str) -> bool:
    if not APP_PASSWORD:
        return True
    return password == APP_PASSWORD


def public_model(model: Dict[str, Any]) -> Dict[str, Any]:
    provider_key = str(model.get("provider", "openrouter")).lower()
    provider = PROVIDERS.get(provider_key, {})
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "provider": provider.get("label", provider_key.title()),
        "family": model.get("family", "LLM"),
        "category": model.get("family", "LLM"),
    }


def find_model(model_id: str) -> Dict[str, Any]:
    for model in MODELS:
        if model.get("id") == model_id:
            return model
    return MODELS[0]


def get_provider(provider_key: str) -> Optional[Dict[str, Any]]:
    return PROVIDERS.get(str(provider_key).lower())


def build_headers(provider: Dict[str, Any]) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {provider.get('api_key', '')}",
        "Content-Type": "application/json",
    }

    extra_headers = provider.get("extra_headers", {})
    if isinstance(extra_headers, dict):
        headers.update({str(k): str(v) for k, v in extra_headers.items()})

    return headers


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "password_required": password_required(),
        "providers": {
            key: {
                "label": cfg.get("label", key),
                "configured": bool(cfg.get("api_key")),
                "base_url": cfg.get("base_url"),
            }
            for key, cfg in PROVIDERS.items()
        },
    }


@app.get("/auth/status")
def auth_status():
    return {"password_required": password_required()}


@app.post("/auth")
def auth(req: AuthRequest):
    if check_password(req.password):
        return {"ok": True}

    return JSONResponse(
        status_code=401,
        content={"ok": False, "error": "访问密码错误"},
    )


@app.get("/providers")
def list_providers():
    return {
        "providers": [
            {
                "id": key,
                "label": cfg.get("label", key),
                "configured": bool(cfg.get("api_key")),
                "base_url": cfg.get("base_url"),
            }
            for key, cfg in PROVIDERS.items()
        ]
    }


@app.get("/models")
def get_models():
    return {"models": [public_model(model) for model in MODELS]}


@app.get("/debug-key")
def debug_key():
    # 不返回完整 key。互联网已经够像漏水水管了，别再手动拧开阀门。
    return {
        "has_app_password": bool(APP_PASSWORD),
        "providers": {
            key: {
                "label": cfg.get("label", key),
                "has_key": bool(cfg.get("api_key")),
                "key_start": str(cfg.get("api_key", ""))[:10] if cfg.get("api_key") else None,
                "key_length": len(str(cfg.get("api_key", ""))) if cfg.get("api_key") else 0,
            }
            for key, cfg in PROVIDERS.items()
        },
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if not check_password(req.password):
        return JSONResponse(status_code=401, content={"error": "访问密码错误"})

    user_message = req.message.strip()
    if not user_message:
        return JSONResponse(status_code=400, content={"error": "消息不能为空"})

    safe_max_tokens = min(max(int(req.max_tokens or 1024), 128), 4096)
    selected_model = find_model(req.model)
    provider_key = str(selected_model.get("provider", "openrouter")).lower()
    provider = get_provider(provider_key)

    if not provider:
        return JSONResponse(
            status_code=500,
            content={"error": f"Provider 未配置：{provider_key}", "model": selected_model.get("id")},
        )

    if not provider.get("api_key"):
        return JSONResponse(
            status_code=500,
            content={"error": f"{provider.get('label', provider_key)} API Key 未配置，请检查 Render 环境变量", "model": selected_model.get("id")},
        )

    base_url = str(provider.get("base_url", "")).rstrip("/")
    if not base_url:
        return JSONResponse(
            status_code=500,
            content={"error": f"{provider.get('label', provider_key)} base_url 未配置"},
        )

    system_message = {
        "role": "system",
        "content": "你是 MaGary AI，是马广辉的中文 AI 助手。请用自然、清楚、适当幽默的中文回答。",
    }

    history = []
    if req.history:
        for item in req.history[-10:]:
            if item.role in ["user", "assistant"] and item.content.strip():
                history.append({"role": item.role, "content": item.content.strip()})

    messages = [
        system_message,
        *history,
        {"role": "user", "content": user_message},
    ]

    model_name = selected_model.get("real_id") or selected_model.get("id")

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": safe_max_tokens,
        "temperature": 0.7,
    }

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=build_headers(provider),
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
                    "provider": provider.get("label"),
                    "model": selected_model.get("id"),
                },
            )

        if resp.status_code != 200:
            error_obj = result.get("error")
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = error_obj or result.get("message") or "模型接口请求失败"

            return JSONResponse(
                status_code=resp.status_code,
                content={
                    "error": error_msg,
                    "status_code": resp.status_code,
                    "raw": result,
                    "provider": provider.get("label"),
                    "model": selected_model.get("id"),
                    "real_model": model_name,
                },
            )

        reply = ""
        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            if isinstance(choice.get("message"), dict):
                reply = choice["message"].get("content", "") or ""
            if not reply:
                reply = choice.get("text", "") or ""

        if not reply:
            reply = "模型返回了空内容。"

        return {
            "reply": reply,
            "model": selected_model.get("id"),
            "real_model": model_name,
            "provider": provider.get("label"),
            "family": selected_model.get("family", "LLM"),
        }

    except requests.exceptions.Timeout:
        return JSONResponse(
            status_code=504,
            content={"error": "请求超时，请换个模型或稍后再试", "provider": provider.get("label"), "model": selected_model.get("id")},
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "provider": provider.get("label"), "model": selected_model.get("id")},
        )
