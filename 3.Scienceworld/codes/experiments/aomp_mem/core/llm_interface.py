"""LLM interface abstractions used by the A-OMP-Mem experiment package."""

from __future__ import annotations

import http.client
import json
import os
import ssl
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib import error, request


@dataclass
class GenerationResult:
    """Normalized response returned by an LLM backend."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallTracker:
    """Tracks aggregate LLM usage for cost accounting."""

    total_calls: int = 0
    prompts: List[str] = field(default_factory=list)

    def record(self, prompt: str) -> None:
        self.total_calls += 1
        self.prompts.append(prompt)


class RemoteLLMError(RuntimeError):
    """Raised when a remote LLM backend cannot return a usable response."""


@dataclass
class RequestAttempt:
    """Concrete HTTP request attempt for one protocol and endpoint combination."""

    protocol: str
    url: str
    headers: Dict[str, str]
    payload: Dict[str, Any]


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE env file without introducing extra dependencies."""

    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _load_project_env_values() -> Dict[str, str]:
    """Load local project env values from .env.local when present."""

    project_root = Path(__file__).resolve().parents[5]
    project_file = project_root / ".env.local"
    return _parse_env_file(project_file)


class BaseLLMClient(ABC):
    """Minimal abstract interface for text generation backends."""

    def __init__(self, tracker: Optional[CallTracker] = None) -> None:
        self.tracker = tracker or CallTracker()

    @abstractmethod
    def generate(self, prompt: str, *, metadata: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """Generate a text response for a prompt."""


class MockLLMClient(BaseLLMClient):
    """Deterministic mock backend for tests and local development."""

    def __init__(
        self,
        responses: Optional[Iterable[str]] = None,
        *,
        default_template: str = "mock-response-{index}",
        tracker: Optional[CallTracker] = None,
    ) -> None:
        super().__init__(tracker=tracker)
        self._responses = list(responses or [])
        self._default_template = default_template

    def generate(self, prompt: str, *, metadata: Optional[Dict[str, Any]] = None) -> GenerationResult:
        self.tracker.record(prompt)
        index = self.tracker.total_calls - 1
        if index < len(self._responses):
            text = self._responses[index]
        else:
            text = self._default_template.format(index=index)
        return GenerationResult(text=text, metadata=dict(metadata or {}))


class GeminiHTTPClient(BaseLLMClient):
    """HTTP-backed Gemini client with common proxy fallback strategies."""

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/133.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        protocol: str = "auto",
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        max_retries: int = 0,
        retry_backoff_s: float = 2.0,
        tracker: Optional[CallTracker] = None,
    ) -> None:
        super().__init__(tracker=tracker)
        if not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not api_key.strip():
            raise ValueError("api_key must be non-empty")
        if not model_id.strip():
            raise ValueError("model_id must be non-empty")
        if protocol not in {"auto", "openai", "responses", "gemini"}:
            raise ValueError("protocol must be one of: auto, openai, responses, gemini")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.protocol = protocol
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))

    def generate(self, prompt: str, *, metadata: Optional[Dict[str, Any]] = None) -> GenerationResult:
        self.tracker.record(prompt)
        failures: List[str] = []

        for attempt in self._build_attempts(prompt):
            for retry_index in range(self.max_retries + 1):
                try:
                    response_json = self._post_json(attempt)
                    text = self._extract_text(attempt.protocol, response_json)
                    result_metadata = dict(metadata or {})
                    result_metadata.update(
                        {
                            "protocol": attempt.protocol,
                            "url": attempt.url,
                            "model_id": self.model_id,
                            "response_model": response_json.get("model"),
                            "retry_count": retry_index,
                        }
                    )
                    return GenerationResult(text=text, metadata=result_metadata)
                except RemoteLLMError as exc:
                    if retry_index < self.max_retries and self._is_retryable_error(exc):
                        self._sleep_before_retry(retry_index)
                        continue
                    failures.append(f"{attempt.protocol}: {exc}")
                    break

        raise RemoteLLMError(
            "All remote generation attempts failed. "
            + " | ".join(failures or ["no request attempts were generated"])
        )

    def _build_attempts(self, prompt: str) -> List[RequestAttempt]:
        attempts: List[RequestAttempt] = []
        for protocol in self._protocol_order():
            if protocol == "openai":
                attempts.extend(self._build_openai_attempts(prompt))
            elif protocol == "responses":
                attempts.extend(self._build_responses_attempts(prompt))
            elif protocol == "gemini":
                attempts.append(self._build_gemini_attempt(prompt))
        return attempts

    def _protocol_order(self) -> List[str]:
        if self.protocol != "auto":
            return [self.protocol]
        if self.base_url.endswith("/v1") or "/openai" in self.base_url.lower():
            return ["openai", "gemini"]
        return ["gemini", "openai"]

    def _build_openai_attempts(self, prompt: str) -> List[RequestAttempt]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.DEFAULT_USER_AGENT,
        }
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        if self.base_url.endswith("/chat/completions"):
            candidates = [self.base_url]
        elif self.base_url.endswith("/v1"):
            candidates = [f"{self.base_url}/chat/completions"]
        else:
            candidates = [f"{self.base_url}/v1/chat/completions", f"{self.base_url}/chat/completions"]

        attempts: List[RequestAttempt] = []
        seen_urls = set()
        for url in candidates:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            attempts.append(RequestAttempt(protocol="openai", url=url, headers=headers, payload=payload))
        return attempts

    def _build_responses_attempts(self, prompt: str) -> List[RequestAttempt]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.DEFAULT_USER_AGENT,
        }
        payload = {
            "model": self.model_id,
            "input": prompt,
            "temperature": self.temperature,
            "store": False,
            "stream": True,
        }

        if self.base_url.endswith("/responses"):
            candidates = [self.base_url]
        elif self.base_url.endswith("/v1"):
            candidates = [f"{self.base_url}/responses"]
        else:
            candidates = [f"{self.base_url}/v1/responses", f"{self.base_url}/responses"]

        attempts: List[RequestAttempt] = []
        seen_urls = set()
        for url in candidates:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            attempts.append(RequestAttempt(protocol="responses", url=url, headers=headers, payload=payload))
        return attempts

    def _build_gemini_attempt(self, prompt: str) -> RequestAttempt:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self.DEFAULT_USER_AGENT,
            "x-goog-api-key": self.api_key,
        }
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        return RequestAttempt(
            protocol="gemini",
            url=f"{self.base_url}/v1beta/models/{self.model_id}:generateContent",
            headers=headers,
            payload=payload,
        )

    def _post_json(self, attempt: RequestAttempt) -> Dict[str, Any]:
        body = json.dumps(attempt.payload).encode("utf-8")
        req = request.Request(attempt.url, data=body, headers=attempt.headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RemoteLLMError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc
        except error.URLError as exc:
            raise RemoteLLMError(f"URL error: {exc.reason}") from exc
        except ssl.SSLError as exc:
            raise RemoteLLMError(f"SSL error: {exc}") from exc
        except http.client.RemoteDisconnected as exc:
            raise RemoteLLMError("Remote connection closed without response") from exc
        except http.client.IncompleteRead as exc:
            raise RemoteLLMError(f"Incomplete read: {exc}") from exc
        except ConnectionResetError as exc:
            raise RemoteLLMError("Connection reset by peer") from exc
        except OSError as exc:
            raise RemoteLLMError(f"OS error: {exc}") from exc
        except TimeoutError as exc:
            raise RemoteLLMError("Request timed out") from exc

        if attempt.payload.get("stream") and raw.lstrip().startswith(("event:", "data:")):
            return self._parse_responses_sse(raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RemoteLLMError(f"Invalid JSON response: {raw[:200]}") from exc

    @staticmethod
    def _parse_responses_sse(raw: str) -> Dict[str, Any]:
        events: List[str] = []
        delta_texts: List[str] = []
        done_text: Optional[str] = None
        completed_response: Dict[str, Any] = {}

        for block in raw.replace("\r", "").split("\n\n"):
            event_name: Optional[str] = None
            data_lines: List[str] = []
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())
            if event_name:
                events.append(event_name)
            if not data_lines:
                continue

            data_text = "\n".join(data_lines)
            if data_text == "[DONE]":
                continue
            try:
                event_payload = json.loads(data_text)
            except json.JSONDecodeError:
                continue

            if event_name == "response.output_text.delta" and isinstance(event_payload.get("delta"), str):
                delta_texts.append(event_payload["delta"])
            elif event_name == "response.output_text.done" and isinstance(event_payload.get("text"), str):
                done_text = event_payload["text"]
            elif event_name == "response.completed" and isinstance(event_payload.get("response"), dict):
                completed_response = dict(event_payload["response"])

        text = "".join(delta_texts)
        if not text and done_text is not None:
            text = done_text

        if completed_response:
            completed_response["stream_events"] = events
            if text:
                completed_response["output_text"] = text
            return completed_response

        if text:
            return {"output_text": text, "stream_events": events}
        raise RemoteLLMError(f"Responses stream missing completed response and output text: {raw[:200]}")

    def _sleep_before_retry(self, retry_index: int) -> None:
        delay_s = self.retry_backoff_s * (2**retry_index)
        if delay_s > 0:
            time.sleep(delay_s)

    @staticmethod
    def _is_retryable_error(exc: RemoteLLMError) -> bool:
        message = str(exc).lower()
        retryable_markers = [
            "request timed out",
            "remote connection closed without response",
            "incomplete read:",
            "connection reset by peer",
            "url error: timed out",
            "url error: [ssl:",
            "ssl error:",
            "unexpected_eof_while_reading",
            "eof occurred in violation of protocol",
            "unknown error (_ssl",
            "os error: [ssl:",
            "os error: [winerror 10053]",
            "os error: [winerror 10054]",
            "http 408",
            "http 409",
            "http 425",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "http 524",
        ]
        return any(marker in message for marker in retryable_markers)

    @staticmethod
    def _extract_text(protocol: str, payload: Dict[str, Any]) -> str:
        if protocol == "openai":
            choices = payload.get("choices") or []
            if not choices:
                raise RemoteLLMError(f"OpenAI-style response missing choices: {payload}")
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                        texts.append(str(item["text"]))
                if texts:
                    return "\n".join(texts).strip()
            raise RemoteLLMError(f"OpenAI-style response missing text content: {payload}")

        if protocol == "responses":
            output_text = payload.get("output_text")
            if isinstance(output_text, str):
                return output_text.strip()

            texts = []
            for output_item in payload.get("output") or []:
                if not isinstance(output_item, dict):
                    continue
                for content_item in output_item.get("content") or []:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            if texts:
                return "\n".join(texts).strip()
            raise RemoteLLMError(f"Responses API response missing output text: {payload}")

        candidates = payload.get("candidates") or []
        if not candidates:
            raise RemoteLLMError(f"Gemini response missing candidates: {payload}")
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        texts = [str(part.get("text", "")).strip() for part in parts if isinstance(part, dict) and part.get("text")]
        if texts:
            return "\n".join(texts).strip()
        raise RemoteLLMError(f"Gemini response missing text parts: {payload}")


class OpenAIHTTPClient(GeminiHTTPClient):
    """OpenAI-compatible client for chat-completions or responses wire APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout_s: float = 60.0,
        temperature: float = 0.0,
        wire_api: str = "chat_completions",
        max_retries: int = 0,
        retry_backoff_s: float = 2.0,
        tracker: Optional[CallTracker] = None,
    ) -> None:
        normalized_wire_api = wire_api.strip().lower().replace("-", "_")
        if normalized_wire_api in {"chat", "chat_completion", "chat_completions", "openai"}:
            protocol = "openai"
        elif normalized_wire_api in {"response", "responses"}:
            protocol = "responses"
        else:
            raise ValueError("wire_api must be one of: chat_completions, responses")

        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            protocol=protocol,
            timeout_s=timeout_s,
            temperature=temperature,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            tracker=tracker,
        )


def build_gemini_client_from_env(
    *,
    tracker: Optional[CallTracker] = None,
    timeout_s: float = 60.0,
    temperature: float = 0.0,
    protocol: Optional[str] = None,
    max_retries: Optional[int] = None,
    retry_backoff_s: Optional[float] = None,
) -> GeminiHTTPClient:
    """Create a GeminiHTTPClient from environment variables."""

    local_values = _load_project_env_values()
    base_url = (
        os.environ.get("GOOGLE_GEMINI_BASE_URL")
        or os.environ.get("GEMINI_BASE_URL")
        or local_values.get("GOOGLE_GEMINI_BASE_URL")
        or local_values.get("GEMINI_BASE_URL")
    )
    api_key = os.environ.get("GEMINI_API_KEY") or local_values.get("GEMINI_API_KEY")
    model_id = os.environ.get("GEMINI_MODEL") or local_values.get("GEMINI_MODEL") or "gemini-2.5-flash"
    resolved_protocol = protocol or os.environ.get("GEMINI_API_PROTOCOL") or local_values.get("GEMINI_API_PROTOCOL") or "auto"
    resolved_retries = max_retries
    if resolved_retries is None:
        raw_retries = os.environ.get("GEMINI_MAX_RETRIES") or local_values.get("GEMINI_MAX_RETRIES") or "0"
        resolved_retries = int(raw_retries)
    resolved_backoff = retry_backoff_s
    if resolved_backoff is None:
        raw_backoff = os.environ.get("GEMINI_RETRY_BACKOFF_S") or local_values.get("GEMINI_RETRY_BACKOFF_S") or "2.0"
        resolved_backoff = float(raw_backoff)

    missing = []
    if not base_url:
        missing.append("GOOGLE_GEMINI_BASE_URL")
    if not api_key:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

    return GeminiHTTPClient(
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        protocol=resolved_protocol,
        timeout_s=timeout_s,
        temperature=temperature,
        max_retries=resolved_retries,
        retry_backoff_s=resolved_backoff,
        tracker=tracker,
    )


def build_openai_client_from_env(
    *,
    tracker: Optional[CallTracker] = None,
    timeout_s: float = 60.0,
    temperature: float = 0.0,
    max_retries: Optional[int] = None,
    retry_backoff_s: Optional[float] = None,
) -> OpenAIHTTPClient:
    """Create an OpenAI-compatible client from environment variables."""

    local_values = _load_project_env_values()
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or local_values.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    api_key = os.environ.get("OPENAI_API_KEY") or local_values.get("OPENAI_API_KEY")
    model_id = os.environ.get("OPENAI_MODEL") or local_values.get("OPENAI_MODEL") or "gpt-5.2"
    wire_api = os.environ.get("OPENAI_WIRE_API") or local_values.get("OPENAI_WIRE_API") or "chat_completions"
    resolved_retries = max_retries
    if resolved_retries is None:
        raw_retries = os.environ.get("OPENAI_MAX_RETRIES") or local_values.get("OPENAI_MAX_RETRIES") or "0"
        resolved_retries = int(raw_retries)
    resolved_backoff = retry_backoff_s
    if resolved_backoff is None:
        raw_backoff = os.environ.get("OPENAI_RETRY_BACKOFF_S") or local_values.get("OPENAI_RETRY_BACKOFF_S") or "2.0"
        resolved_backoff = float(raw_backoff)

    if not api_key:
        raise ValueError("Missing required environment variable(s): OPENAI_API_KEY")

    return OpenAIHTTPClient(
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
        wire_api=wire_api,
        timeout_s=timeout_s,
        temperature=temperature,
        max_retries=resolved_retries,
        retry_backoff_s=resolved_backoff,
        tracker=tracker,
    )
