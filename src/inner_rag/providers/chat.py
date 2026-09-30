"""Chat 模型构造：把 ``ProviderSpec`` 变成 LangChain 的 BaseChatModel。

* Ollama 走 ``langchain-ollama``（本地、无需密钥）；
* DeepSeek 走官方集成 ``langchain-deepseek``（字段名与 OpenAI 兼容层略有差异，
  且支持 ``reasoning_effort``）；
* OpenRouter / OpenAI 都是 OpenAI 兼容接口，统一走 ``langchain-openai`` 的
  ChatOpenAI，只换 ``base_url`` 与 ``model``；
* Mock 是本项目自带的离线模型，用于本地演示、CI 与降级验收。

Phase 7 起不再有 ``if name == ...`` 的分发函数：构造器按名字进 ``CHAT_BUILDERS`` 查表，
由 ``providers/specs.py`` 连同 spec 一起注册进插件注册表（第三方实现走 entry point）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from inner_rag.core.config import settings

if TYPE_CHECKING:  # 只在类型检查时引入，避免 specs <-> chat 的 import 环
    from inner_rag.providers.specs import ProviderSpec

# OpenRouter 用它做应用归因（不参与计费，便于在后台区分流量来源）
OPENROUTER_HEADERS = {"X-Title": settings.APP_NAME}


def _message_text(message: BaseMessage) -> str:
    """把消息内容统一成字符串（多模态内容会被拼接为文本）。"""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(str(part) for part in content)


class MockChatModel(BaseChatModel):
    """离线假模型：不联网、不需要密钥。

    回答是确定性的，并回显「检索到几段参考 + 问题」，因此 mock provider 下也能
    端到端验证「检索 -> 组装 Prompt -> 流式输出 -> 引用来源」这条链路是否真的通了。
    """

    model_name: str = "mock-chat"
    streaming_chunk_size: int = 8

    @property
    def _llm_type(self) -> str:
        return "inner-rag-mock-chat"

    def _answer(self, messages: list[BaseMessage]) -> str:
        system = next((_message_text(m) for m in messages if m.type == "system"), "")
        question = next(
            (_message_text(m) for m in reversed(messages) if m.type == "human"), ""
        ).strip()
        segments = 0
        if system:
            # RAGService 用 "\n\n---\n\n" 拼接参考片段
            segments = system.count("\n\n---\n\n") + 1
        return f"[mock] 已检索到 {segments} 段参考（上下文 {len(system)} 字符）。问题：{question}"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._answer(messages)))]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """按固定长度切片输出，用来验证 SSE 流式链路。"""
        answer = self._answer(messages)
        size = max(1, self.streaming_chunk_size)
        for start in range(0, len(answer), size):
            piece = answer[start : start + size]
            if run_manager is not None:
                run_manager.on_llm_new_token(piece)
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


# ── 各后端的构造器（不发起任何网络请求）────────────────────────────────


def build_mock_chat(spec: ProviderSpec) -> BaseChatModel:
    return MockChatModel(model_name=spec.model)


def build_ollama_chat(spec: ProviderSpec) -> BaseChatModel:
    # LangChain 1.x 的 ChatOllama 不再有 streaming 初始化参数，流式与否由调用方决定
    return ChatOllama(
        base_url=spec.base_url,
        model=spec.model,
        temperature=settings.LLM_TEMPERATURE,
        num_predict=settings.LLM_MAX_TOKENS,
    )


def build_deepseek_chat(spec: ProviderSpec) -> BaseChatModel:
    # 官方集成：输出长度字段是 max_tokens（不是兼容层的 max_completion_tokens）
    return ChatDeepSeek(
        base_url=spec.base_url,
        model=spec.model,
        api_key=SecretStr(spec.api_key),
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
        # 留空表示不下发该参数（DeepSeek 默认行为）
        reasoning_effort=settings.LLM_REASONING_EFFORT.strip() or None,
    )


def build_openai_compatible_chat(spec: ProviderSpec) -> BaseChatModel:
    """OpenRouter / OpenAI / vLLM 这类 OpenAI 兼容后端共用的构造器。"""
    # 本地推理服务（vLLM / SGLang）默认不校验 API Key，但 ChatOpenAI 要求非空，
    # 传一个占位值即可——本地服务会忽略它。
    api_key = spec.api_key or "EMPTY"
    return ChatOpenAI(
        base_url=spec.base_url,
        model=spec.model,
        api_key=SecretStr(api_key),
        temperature=settings.LLM_TEMPERATURE,
        # langchain-openai 1.x 的输出长度字段是 max_completion_tokens（max_tokens 是旧别名）
        max_completion_tokens=settings.LLM_MAX_TOKENS,
        # 字段名是 request_timeout，构造参数的别名才是 timeout
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
        default_headers=OPENROUTER_HEADERS if spec.name == "openrouter" else None,
    )


CHAT_BUILDERS: dict[str, Callable[[ProviderSpec], BaseChatModel]] = {
    "ollama": build_ollama_chat,
    "openrouter": build_openai_compatible_chat,
    "deepseek": build_deepseek_chat,
    "openai": build_openai_compatible_chat,
    "vllm": build_openai_compatible_chat,
    "mock": build_mock_chat,
}
