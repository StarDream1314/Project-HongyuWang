from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[3] / ".env.local"
    if not env_path.exists():
        raise RuntimeError(f"Missing local environment file: {env_path}")
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Missing dependency: python-dotenv. Run `python -m pip install -r requirements.txt`.") from exc
    load_dotenv(env_path, override=False)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_api_config() -> tuple[str, str, str]:
    load_local_env()
    api_key = _required_env("DEEPSEEK_API_KEY")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model_id = os.environ.get("DEEPSEEK_MODEL_ID", "deepseek-reasoner")
    return api_key, base_url, model_id


def build_chat_payload(messages: list[dict[str, str]], temperature: float, model_id: str) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": messages,
        "temperature": float(temperature),
        "stream": False,
    }


def extract_completion_text(response_json: dict[str, Any]) -> str:
    try:
        text = response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Malformed completion response") from exc
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Completion response did not contain text")
    return text


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return stripped


@dataclass
class DeepSeekClient:
    api_key: str
    base_url: str
    model_id: str

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        api_key, base_url, model_id = load_api_config()
        return cls(api_key=api_key, base_url=base_url, model_id=model_id)

    def complete(self, messages: list[dict[str, str]], temperature: float) -> str:
        payload = build_chat_payload(messages, temperature, self.model_id)
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API request failed with status {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc.reason}") from exc

        content = extract_completion_text(json.loads(body))
        return strip_code_fences(content)
