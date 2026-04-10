"""Azure OpenAI provider implementation using the OpenAI Responses API."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urljoin

import httpx
import json_repair

from loguru import logger

from clicomp.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from clicomp.utils.helpers import estimate_prompt_tokens

_AZURE_MSG_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider via the Responses API.

    Notes:
    - Uses model field as Azure deployment name in the request body.
    - Uses api-key header instead of Authorization Bearer when not using MI.
    - Sends OpenAI-style input messages and tool schemas to /openai/v1/responses.
    - Parses both non-streaming and SSE streaming Responses API payloads into
      the project's provider-neutral ``LLMResponse`` shape.
    """

    def __init__(
        self,
        api_key: str = "",
        api_base: str = "",
        default_model: str = "gpt-5.2-chat",
        use_managed_identity: bool = False,
        managed_identity_client_id: str | None = None,
        timeout: float = 1800.0,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.api_version = "preview"
        self.use_managed_identity = use_managed_identity
        self.managed_identity_client_id = managed_identity_client_id
        self._aad_token_cache: str | None = None
        self._aad_token_expires_at: float = 0
        self._credential: Any | None = None
        self.timeout = timeout

        if not api_base:
            raise ValueError("Azure OpenAI api_base is required")
        if not use_managed_identity and not api_key:
            raise ValueError("Azure OpenAI api_key is required when managed identity is disabled")

        if not api_base.endswith("/"):
            api_base += "/"
        self.api_base = api_base

    def _build_responses_url(self) -> str:
        """Build the Azure OpenAI Responses API URL."""
        base_url = self.api_base
        if not base_url.endswith("/"):
            base_url += "/"
        url = urljoin(base_url, "openai/v1/responses")
        return url

    def _get_bearer_token(self) -> str:
        """Get Azure AD bearer token for Cognitive Services scope."""
        now = time.time()
        if self._aad_token_cache and now < self._aad_token_expires_at - 60:
            return self._aad_token_cache

        try:
            from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
        except ImportError as exc:
            raise RuntimeError(
                "azure-identity is required for Azure Managed Identity auth. "
                "Install it with: uv add azure-identity"
            ) from exc

        if self.managed_identity_client_id:
            credential: Any = ManagedIdentityCredential(client_id=self.managed_identity_client_id)
        else:
            credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)

        token = credential.get_token("https://cognitiveservices.azure.com/.default")
        self._credential = credential
        self._aad_token_cache = token.token
        self._aad_token_expires_at = float(getattr(token, "expires_on", 0) or 0)
        return token.token

    def _build_headers(self) -> dict[str, str]:
        """Build headers for Azure OpenAI API with api-key or bearer auth."""
        headers = {
            "Content-Type": "application/json",
            "x-session-affinity": uuid.uuid4().hex,
        }
        if self.use_managed_identity:
            headers["Authorization"] = f"Bearer {self._get_bearer_token()}"
        else:
            headers["api-key"] = self.api_key
        return headers

    @staticmethod
    def _supports_temperature(
        deployment_name: str,
        reasoning_effort: str | None = None,
    ) -> bool:
        if reasoning_effort:
            return False
        name = deployment_name.lower()
        return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))

    @staticmethod
    def _coerce_text_content(content: Any) -> str | list[dict[str, Any]] | None:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return [content]
        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            all_text = True
            text_buf: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_buf.append(item)
                    parts.append({"type": "input_text", "text": item})
                    continue
                if not isinstance(item, dict):
                    all_text = False
                    continue
                item_type = item.get("type")
                if item_type in {"text", "input_text", "output_text"} and isinstance(item.get("text"), str):
                    text = item["text"]
                    text_buf.append(text)
                    parts.append({"type": "input_text", "text": text})
                elif item_type == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                    else:
                        url = image_url
                    if isinstance(url, str) and url:
                        all_text = False
                        parts.append({"type": "input_image", "image_url": url})
                else:
                    all_text = False
            if all_text:
                return "".join(text_buf)
            return parts or None
        return str(content)

    @staticmethod
    def _split_tool_call_id(tool_call_id: Any) -> tuple[str | None, str | None]:
        if isinstance(tool_call_id, str) and tool_call_id:
            if "|" in tool_call_id:
                call_id, item_id = tool_call_id.split("|", 1)
                return call_id or None, item_id or None
            return tool_call_id, None
        return None, None

    def _prepare_responses_input(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        sanitized = self._sanitize_request_messages(
            self._sanitize_empty_content(messages),
            _AZURE_MSG_KEYS,
        )

        instructions_parts: list[str] = []
        items: list[dict[str, Any]] = []
        declared_call_ids: set[str] = set()

        for idx, msg in enumerate(sanitized):
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                sys_text = self._coerce_text_content(content)
                if isinstance(sys_text, str) and sys_text:
                    instructions_parts.append(sys_text)
                elif isinstance(sys_text, list):
                    text_parts = [part.get("text", "") for part in sys_text if isinstance(part, dict)]
                    text = "".join(p for p in text_parts if isinstance(p, str))
                    if text:
                        instructions_parts.append(text)
                continue

            if role == "user":
                coerced = self._coerce_text_content(content)
                if coerced is None:
                    coerced = ""
                items.append({"type": "message", "role": "user", "content": coerced})
                continue

            if role == "assistant":
                coerced = self._coerce_text_content(content)
                if coerced not in (None, "", []):
                    items.append({"type": "message", "role": "assistant", "content": coerced})
                for tool_call in msg.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    fn = tool_call.get("function") or {}
                    name = str(fn.get("name") or "").strip()
                    if not name:
                        logger.error("Dropping Azure Responses function_call with empty name from reconstructed history: {}", tool_call)
                        continue
                    call_id, item_id = self._split_tool_call_id(tool_call.get("id"))
                    arguments = fn.get("arguments") or "{}"
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    resolved_call_id = call_id or f"call_{idx}_{uuid.uuid4().hex[:8]}"
                    declared_call_ids.add(resolved_call_id)
                    items.append({
                        "type": "function_call",
                        "id": item_id or f"fc_{idx}_{uuid.uuid4().hex[:8]}",
                        "call_id": resolved_call_id,
                        "name": name,
                        "arguments": arguments,
                    })
                continue

            if role == "tool":
                call_id, _ = self._split_tool_call_id(msg.get("tool_call_id"))
                if not call_id or call_id not in declared_call_ids:
                    logger.error("Dropping Azure Responses function_call_output with unknown call_id from reconstructed history: {}", msg.get("tool_call_id"))
                    continue
                output = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                })

        instructions = "\n\n".join(part for part in instructions_parts if part) or None
        return instructions, items

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            fn = tool.get("function") or {}
            converted.append({
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted or None

    @staticmethod
    def _map_tool_choice(tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            if tool_choice in {"auto", "none", "required"}:
                return tool_choice
            return "auto"
        if isinstance(tool_choice, dict):
            fn = tool_choice.get("function") or {}
            name = fn.get("name")
            if name:
                return {"type": "function", "name": name}
        return None

    def _prepare_request_payload(
        self,
        deployment_name: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stream: bool = False,
        previous_response_id: str | None = None,
    ) -> dict[str, Any]:
        instructions, input_items = self._prepare_responses_input(messages)
        payload: dict[str, Any] = {
            "model": deployment_name,
            "input": input_items,
            "max_output_tokens": max(1, max_tokens),
        }

        if previous_response_id:
            payload["previous_response_id"] = previous_response_id

        if instructions:
            payload["instructions"] = instructions

        if stream:
            payload["stream"] = True

        if self._supports_temperature(deployment_name, reasoning_effort):
            payload["temperature"] = temperature

        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}

        converted_tools = self._convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools
            mapped_choice = self._map_tool_choice(tool_choice)
            if mapped_choice is not None:
                payload["tool_choice"] = mapped_choice

        return payload

    @staticmethod
    def _extract_text_from_output(output: list[dict[str, Any]] | None) -> str | None:
        if not output:
            return None
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                        text = block.get("text")
                        if isinstance(text, str):
                            texts.append(text)
        return "".join(texts) or None

    @staticmethod
    def _extract_tool_calls(output: list[dict[str, Any]] | None) -> list[ToolCallRequest]:
        if not output:
            return []
        tool_calls: list[ToolCallRequest] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call":
                continue
            args = item.get("arguments") or {}
            if isinstance(args, str):
                args = json_repair.loads(args)
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                ToolCallRequest(
                    id=str(item.get("call_id") or item.get("id") or uuid.uuid4().hex[:9]),
                    name=str(item.get("name") or ""),
                    arguments=args,
                )
            )
        return tool_calls

    @staticmethod
    def _extract_usage(response: dict[str, Any]) -> dict[str, int]:
        usage_data = response.get("usage") or {}
        input_tokens = int(usage_data.get("input_tokens") or usage_data.get("prompt_tokens") or 0)
        output_tokens = int(usage_data.get("output_tokens") or usage_data.get("completion_tokens") or 0)
        total = int(usage_data.get("total_tokens") or (input_tokens + output_tokens))
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total,
        }

    def _parse_response(self, response: dict[str, Any]) -> LLMResponse:
        try:
            output = response.get("output") or []
            content = self._extract_text_from_output(output)
            tool_calls = self._extract_tool_calls(output)
            finish_reason = str(response.get("status") or "stop")
            parsed = LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=self._extract_usage(response),
            )
            setattr(parsed, "provider_response_id", response.get("id"))
            setattr(parsed, "provider_output_items", output)
            return parsed
        except Exception as e:
            return LLMResponse(
                content=f"Error parsing Azure OpenAI response: {str(e)}",
                finish_reason="error",
            )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        deployment_name = model or self.default_model
        url = self._build_responses_url()
        headers = self._build_headers()
        payload = self._prepare_request_payload(
            deployment_name,
            messages,
            tools,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice=tool_choice,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=True) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    return LLMResponse(
                        content=f"Azure OpenAI API Error {response.status_code}: {response.text}",
                        finish_reason="error",
                    )
                return self._parse_response(response.json())
        except Exception as e:
            return LLMResponse(
                content=f"Error calling Azure OpenAI: {repr(e)}",
                finish_reason="error",
            )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        deployment_name = model or self.default_model
        url = self._build_responses_url()
        headers = self._build_headers()
        payload = self._prepare_request_payload(
            deployment_name,
            messages,
            tools,
            max_tokens,
            temperature,
            reasoning_effort,
            tool_choice=tool_choice,
            stream=True,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=True) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        text = await response.aread()
                        return LLMResponse(
                            content=f"Azure OpenAI API Error {response.status_code}: {text.decode('utf-8', 'ignore')}",
                            finish_reason="error",
                        )
                    return await self._consume_stream(response, on_content_delta)
        except Exception as e:
            return LLMResponse(content=f"Error calling Azure OpenAI: {repr(e)}", finish_reason="error")

    async def _consume_stream(
        self,
        response: httpx.Response,
        on_content_delta: Callable[[str], Awaitable[None]] | None,
    ) -> LLMResponse:
        content_parts: list[str] = []
        tool_call_buffers: dict[str, dict[str, str]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except Exception:
                continue

            event_type = event.get("type")
            if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                delta = event.get("delta")
                if isinstance(delta, str) and delta:
                    content_parts.append(delta)
                    if on_content_delta:
                        await on_content_delta(delta)
            elif event_type == "response.output_item.added":
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    key = str(item.get("call_id") or item.get("id") or uuid.uuid4().hex[:9])
                    tool_call_buffers.setdefault(key, {
                        "id": key,
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or ""),
                    })
            elif event_type == "response.function_call_arguments.delta":
                call_id = str(event.get("call_id") or event.get("item_id") or "")
                if call_id:
                    buf = tool_call_buffers.setdefault(call_id, {"id": call_id, "name": "", "arguments": ""})
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        buf["arguments"] += delta
            elif event_type == "response.function_call_arguments.done":
                call_id = str(event.get("call_id") or event.get("item_id") or "")
                if call_id:
                    buf = tool_call_buffers.setdefault(call_id, {"id": call_id, "name": "", "arguments": ""})
                    args = event.get("arguments")
                    if isinstance(args, str):
                        buf["arguments"] = args
            elif event_type == "response.completed":
                resp = event.get("response") or {}
                finish_reason = str(resp.get("status") or "stop")
                usage = self._extract_usage(resp)
            elif event_type == "response.failed":
                resp = event.get("response") or {}
                error = resp.get("error") or event.get("error") or {}
                return LLMResponse(
                    content=f"Azure OpenAI API Error: {json.dumps(error, ensure_ascii=False)}",
                    finish_reason="error",
                )

        tool_calls = []
        for buf in tool_call_buffers.values():
            args_raw = buf.get("arguments") or "{}"
            try:
                args = json_repair.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                ToolCallRequest(
                    id=buf["id"],
                    name=buf.get("name", ""),
                    arguments=args,
                )
            )

        parsed = LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )
        setattr(parsed, "provider_output_items", [])
        return parsed

    def estimate_prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> tuple[int, str]:
        instructions, input_items = self._prepare_responses_input(messages)
        parts: list[dict[str, Any]] = []
        if instructions:
            parts.append({"role": "system", "content": instructions})
        parts.extend(input_items)
        return estimate_prompt_tokens(parts, self._convert_tools(tools)), "azure_responses_estimate"

    def get_default_model(self) -> str:
        return self.default_model
