import re
from collections.abc import Iterator
from typing import Any

import httpx
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, BadRequestError, OpenAI

from app.config import get_settings
from app.utils.logger import setup_logger


logger = setup_logger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]+")


class DeepSeekClientError(RuntimeError):
    """Friendly error raised for upstream LLM failures."""


def infer_output_language(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return "zh"
    cjk_chars = len(_CJK_RE.findall(raw))
    ascii_words = _ASCII_WORD_RE.findall(raw)
    ascii_letters = sum(len(word) for word in ascii_words)
    if cjk_chars == 0 and ascii_letters > 0:
        return "en"
    if ascii_letters >= max(12, cjk_chars * 2):
        return "en"
    return "zh"


class DeepSeekClient:
    """OpenAI-compatible LLM client used by ChemAnalyst.

    The class name is retained for backward-compatible imports. Runtime calls use
    the unified LLM_* settings, with legacy DEEPSEEK_* or OPENAI_* variables only
    as fallbacks in app.config.Settings.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        http_client = httpx.Client(trust_env=bool(self.settings.active_llm_trust_env))
        self.client = OpenAI(
            api_key=self.settings.active_llm_api_key or "MISSING_API_KEY",
            base_url=self.settings.active_llm_base_url or "https://api.deepseek.com",
            timeout=self.settings.llm_timeout_seconds,
            http_client=http_client,
        )

    def _ensure_configured(self) -> None:
        if not self.settings.active_llm_api_key:
            raise DeepSeekClientError("LLM_API_KEY is not configured. Set it in .env or the runtime environment.")
        if not self.settings.active_llm_base_url:
            raise DeepSeekClientError("LLM_BASE_URL is not configured. Set an OpenAI-compatible API base URL.")
        if not self.settings.active_llm_model:
            raise DeepSeekClientError("LLM_MODEL is not configured. Use the exact model name provided by your LLM service.")

    def _build_chat_messages(self, query: str, context: str | None = None) -> list[dict[str, str]]:
        output_language = infer_output_language(query)
        if output_language == "en":
            system_prompt = (
                "You are ChemAnalyst, a petroleum-domain analytical assistant. "
                "Answer in professional, restrained, accurate English. "
                "Do not overclaim unsupported capabilities or facts."
            )
            user_prompt = query if not context else f"Answer using the context below.\n\nContext:\n{context}\n\nQuestion:\n{query}"
        else:
            system_prompt = (
                "You are ChemAnalyst, a petroleum-domain analytical assistant. "
                "Answer in the user's language unless the user requests another language. "
                "Provide only the data, reasoning, or conclusion directly required by the question. "
                "Do not invent unsupported capabilities or facts."
            )
            user_prompt = query if not context else f"Answer using the context below.\n\nContext:\n{context}\n\nQuestion:\n{query}"
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _build_reason_messages(self, query: str) -> list[dict[str, str]]:
        system_prompt = (
            "You are ChemAnalyst's petroleum-chemistry reasoning module. "
            "Use the provided evidence conservatively, distinguish measured values from inference, "
            "and make confidence and limitations explicit."
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]

    def _request_kwargs(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None,
        thinking: bool,
        max_tokens: int | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        selected_model = (model or self.settings.active_llm_model).strip()
        kwargs: dict[str, Any] = {"model": selected_model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            if selected_model.lower().startswith("gpt-5"):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
        if thinking:
            kwargs["extra_body"] = {"thinking": {"enabled": True}}
        if stream:
            kwargs["stream"] = True
        return kwargs

    def _chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = 0.2,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        self._ensure_configured()
        request_kwargs = self._request_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            thinking=thinking,
            max_tokens=max_tokens,
        )
        try:
            logger.info("Calling configured LLM model=%s thinking=%s", request_kwargs["model"], thinking)
            response = self.client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            if thinking:
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop("extra_body", None)
                response = self.client.chat.completions.create(**retry_kwargs)
            else:
                raise DeepSeekClientError(f"LLM request was rejected: {exc}") from exc
        except AuthenticationError as exc:
            raise DeepSeekClientError("LLM authentication failed. Check LLM_API_KEY.") from exc
        except APITimeoutError as exc:
            raise DeepSeekClientError("LLM request timed out. Increase LLM_TIMEOUT_SECONDS or retry later.") from exc
        except APIConnectionError as exc:
            raise DeepSeekClientError("Could not connect to the configured LLM endpoint. Check network access and LLM_BASE_URL.") from exc
        except APIError as exc:
            raise DeepSeekClientError(f"LLM API error: {exc}") from exc
        except Exception as exc:
            raise DeepSeekClientError(f"LLM call failed: {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise DeepSeekClientError("LLM returned an empty response.")
        return content.strip()

    def _extract_stream_delta(self, chunk: Any) -> str:
        if not getattr(chunk, "choices", None):
            return ""
        delta = getattr(chunk.choices[0], "delta", None)
        if not delta:
            return ""
        content = getattr(delta, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(getattr(item, "text", "") or "") for item in content)
        return str(content)

    def _chat_completion_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float | None = 0.2,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        self._ensure_configured()
        request_kwargs = self._request_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            thinking=thinking,
            max_tokens=max_tokens,
            stream=True,
        )
        try:
            logger.info("Streaming configured LLM model=%s thinking=%s", request_kwargs["model"], thinking)
            stream = self.client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            if thinking:
                retry_kwargs = dict(request_kwargs)
                retry_kwargs.pop("extra_body", None)
                stream = self.client.chat.completions.create(**retry_kwargs)
            else:
                raise DeepSeekClientError(f"LLM streaming request was rejected: {exc}") from exc
        except AuthenticationError as exc:
            raise DeepSeekClientError("LLM authentication failed. Check LLM_API_KEY.") from exc
        except APITimeoutError as exc:
            raise DeepSeekClientError("LLM streaming request timed out.") from exc
        except APIConnectionError as exc:
            raise DeepSeekClientError("Could not connect to the configured LLM endpoint. Check network access and LLM_BASE_URL.") from exc
        except APIError as exc:
            raise DeepSeekClientError(f"LLM API error: {exc}") from exc
        except Exception as exc:
            raise DeepSeekClientError(f"LLM streaming call failed: {exc}") from exc

        yielded = False
        for chunk in stream:
            delta_text = self._extract_stream_delta(chunk)
            if delta_text:
                yielded = True
                yield delta_text
        if not yielded:
            raise DeepSeekClientError("LLM returned an empty streaming response.")

    def chat(self, query: str, context: str | None = None, thinking: bool = False) -> str:
        return self._chat_completion(
            model=self.settings.active_llm_model,
            messages=self._build_chat_messages(query, context),
            thinking=thinking,
        )

    def chat_with_model(
        self,
        *,
        model: str,
        query: str,
        system_prompt: str | None = None,
        temperature: float | None = 0.0,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        return self._chat_completion(
            model=(model or self.settings.active_llm_model),
            messages=messages,
            temperature=temperature,
            thinking=thinking,
            max_tokens=max_tokens,
        )

    def chat_stream(self, query: str, context: str | None = None, thinking: bool = False) -> Iterator[str]:
        return self._chat_completion_stream(
            model=self.settings.active_llm_model,
            messages=self._build_chat_messages(query, context),
            thinking=thinking,
        )

    def reason(self, query: str) -> str:
        return self._chat_completion(
            model=self.settings.active_llm_model,
            messages=self._build_reason_messages(query),
            temperature=0.1,
        )

    def reason_stream(self, query: str) -> Iterator[str]:
        return self._chat_completion_stream(
            model=self.settings.active_llm_model,
            messages=self._build_reason_messages(query),
            temperature=0.1,
        )

