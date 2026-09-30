"""RAG 推理服务：检索 -> 组装 Prompt -> 调用 LLM（非流式与 SSE 流式）。

模型实例全部来自 ``inner_rag.providers``：本模块不关心后端是 Ollama、
OpenRouter、DeepSeek、OpenAI 还是离线 mock。

Phase 5 起本模块同时是 **trace 的主干**：一次问答产生

    rag.request -> retrieve -> (cache.query | vector.search -> embed.query)
                -> prompt.build -> llm.generate

（``embed.query`` 由 ``services/embedding.py`` 自己埋，它天然嵌在 ``vector.search`` 里。）
检索耗时、空召回、token 用量、首 token 延迟这些指标也落在这里——只有这里同时知道
「用了什么参数」和「拿到了什么结果」。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from inner_rag.core.config import settings
from inner_rag.core.metrics import metrics, record_llm_tokens
from inner_rag.core.observability import tracer
from inner_rag.providers import get_chat_model
from inner_rag.providers.specs import chat_spec
from inner_rag.services.cache import query_cache
from inner_rag.services.retrieval_log import log_prompt, log_retrieval
from inner_rag.services.vector_store import Strategy, vector_service

SYSTEM_PROMPT = """你是企业内部知识库助手，请根据以下参考文档回答用户问题。

重要规则：
1. 参考文档是从向量数据库模糊检索出来的，内容可能与问题有一定相关性，请仔细阅读后作答
2. 即使文档标题与问题措辞不完全一致，只要内容相关就应提取并回答
3. 若文档中确实完全没有任何相关内容，才回复"未找到相关信息"
4. 引用内容时标注来源：【来源：文件名】
5. 回答清晰、专业、用中文

参考文档：
{context}
"""

DEFAULT_STRATEGY: Strategy = "hybrid"
PROMPT_HISTORY_TURNS = 3

RetrievalResult = list[tuple[Document, float | None]]


def sse_event(event_type: str, data: Any) -> str:
    """构造一条 SSE 消息。"""
    return f"data: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"


def _llm_identity() -> tuple[str, str]:
    """当前 chat 后端的 (provider, model)，用作指标与 trace 的标签。

    走到这里时模型实例已经取到（配置非法会在 ``get_chat_model`` 就抛 ProviderError），
    所以这里不需要再兜配置错误。
    """
    spec = chat_spec()
    return spec.name, spec.model


def _token_usage(message: AIMessage) -> dict[str, int]:
    """从消息里读 token 用量。

    ``usage_metadata`` 由 provider 回填，流式且未开 ``stream_usage`` 时整块缺失——
    这时返回空字典：「没有用量数据」和「用了 0 个 token」是两回事，不能混成一个 0。
    """
    meta = message.usage_metadata
    if not meta:
        return {}
    prompt_tokens = int(meta.get("input_tokens") or 0)
    completion_tokens = int(meta.get("output_tokens") or 0)
    if not (prompt_tokens or completion_tokens):
        return {}
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


class RAGService:
    def _get_llm(self) -> BaseChatModel:
        """获取对话模型（由 LLM_PROVIDER 决定，实例在 provider 工厂内复用）。

        配置非法（未知 provider / 缺 API Key）时抛 ProviderError，
        由 API 层映射成 503 + 可读提示。
        """
        return get_chat_model()

    def _build_prompt(self, history: list[dict[str, Any]]) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(self._build_prompt_messages(history))

    # ── 检索（QueryCache + 日志 + 指标）────────────────────────────────

    async def retrieve(
        self,
        kb_id: int,
        query: str,
        k: int | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ) -> tuple[RetrievalResult, bool]:
        """返回 (results, cache_hit)。检索异常不在此处吞掉，由调用方决定如何呈现。"""
        k = k or settings.TOP_K
        started = time.perf_counter()

        async with tracer.span(
            "retrieve",
            kb_id=kb_id,
            strategy=strategy,
            k=k,
            embedding_identity=settings.embedding_key,
        ) as span:
            async with tracer.span("cache.query", kb_id=kb_id) as cache_span:
                cached = await query_cache.get(kb_id, query, k)
            cache_span.set(cache_hit=cached is not None)

            if cached is not None:
                latency_ms = (time.perf_counter() - started) * 1000
                metrics.increment("rag_cache_hits_total", labels={"namespace": "query"})
                self._record_retrieval(strategy, latency_ms, len(cached), 0)
                span.set(cache_hit=True, hits=len(cached))
                log_retrieval(kb_id, query, cached, latency_ms, True, strategy)
                return cached, True

            async with tracer.span(
                "vector.search", kb_id=kb_id, strategy=strategy, k=k
            ) as search_span:
                results, filtered_out = await vector_service.search(
                    kb_id=kb_id,
                    query=query,
                    k=k,
                    strategy=strategy,
                )
                search_span.set(hits=len(results), filtered_out=filtered_out)
                top_score = results[0][1] if results else None
                if top_score is not None:
                    search_span.set(top_score=round(float(top_score), 4))

            latency_ms = (time.perf_counter() - started) * 1000
            self._record_retrieval(strategy, latency_ms, len(results), filtered_out)
            span.set(cache_hit=False, hits=len(results), filtered_out=filtered_out)
            log_retrieval(
                kb_id, query, results, latency_ms, False, strategy, filtered_count=filtered_out
            )

            if results:
                await query_cache.set(kb_id, query, k, results)
            return results, False

    @staticmethod
    def _record_retrieval(
        strategy: Strategy, latency_ms: float, hits: int, filtered_out: int
    ) -> None:
        """检索指标：耗时按策略分，空召回与过滤数单独计数（阈值调高/调低都体现在这两个数上）。"""
        metrics.observe("rag_retrieve_duration_ms", latency_ms, labels={"strategy": strategy})
        if hits == 0:
            metrics.increment("rag_retrieve_empty_total")
        if filtered_out:
            metrics.increment("rag_retrieve_filtered_total", filtered_out)

    # ── Prompt 组装 ────────────────────────────────────────────────────

    def _build_context(self, results: RetrievalResult) -> tuple[str, list[dict[str, Any]]]:
        parts: list[str] = []
        sources: list[dict[str, Any]] = []
        total_len = 0

        for index, (doc, score) in enumerate(results[: settings.RERANK_TOP_K]):
            filename = doc.metadata.get("filename", "未知文件")
            page = doc.metadata.get("page", "")
            page_info = f"·第{page}页" if page else ""
            content = doc.page_content

            if total_len + len(content) > settings.MAX_CONTEXT_LENGTH:
                remain = settings.MAX_CONTEXT_LENGTH - total_len
                if remain < 100:
                    break
                content = content[:remain] + "…"

            score_info = f"（相关度：{score:.1%}）" if score is not None else ""
            parts.append(f"[{index + 1}] 【{filename}{page_info}】{score_info}\n{content}")
            total_len += len(content)

            sources.append(
                {
                    "index": index + 1,
                    "filename": filename,
                    "page": page,
                    "score": round(float(score), 4) if score is not None else None,
                    "doc_id": doc.metadata.get("doc_id"),
                    "content": (
                        doc.page_content[:300] + "…"
                        if len(doc.page_content) > 300
                        else doc.page_content
                    ),
                }
            )

        return "\n\n---\n\n".join(parts), sources

    def _build_prompt_messages(self, history: list[dict[str, Any]]) -> list[tuple[str, str]]:
        messages: list[tuple[str, str]] = [("system", SYSTEM_PROMPT)]
        for message in history[-PROMPT_HISTORY_TURNS * 2 :]:
            if message["role"] == "user":
                messages.append(("human", message["content"]))
            elif message["role"] == "assistant":
                messages.append(("ai", message["content"]))
        messages.append(("human", "{question}"))
        return messages

    def _log_prompt(self, kb_id: int, question: str, context: str, history: list) -> None:
        log_prompt(
            kb_id=kb_id,
            question=question,
            context_length=len(context),
            history_turns=len(history) // 2,
            prompt_tokens_est=len(context) // 2 + len(question) // 2,
        )

    async def _build_context_block(
        self, kb_id: int, results: RetrievalResult
    ) -> tuple[str, list[dict[str, Any]]]:
        """组装 context 并记 ``prompt.build`` span（分块数与字符数用于诊断 Prompt 膨胀）。"""
        async with tracer.span("prompt.build", kb_id=kb_id) as span:
            context, sources = self._build_context(results)
            span.set(chunk_count=len(sources), context_chars=len(context))
        return context, sources

    # ── 模型调用 ───────────────────────────────────────────────────────

    def _prepare_llm(
        self, history: list[dict[str, Any]], context: str, question: str
    ) -> tuple[Any, Any, str, str]:
        """取模型实例并渲染消息，返回 (llm, messages, provider, model)。

        为什么不直接串成 ``prompt | llm | StrOutputParser`` 一条链：那样 ``ainvoke`` 出来
        是纯字符串，拿不到 ``AIMessage.usage_metadata``，token 用量与首 token 延迟就没数了。
        """
        llm = self._get_llm()
        provider, model = _llm_identity()
        messages = self._build_prompt(history).invoke({"context": context, "question": question})
        return llm, messages, provider, model

    # ── 非流式 ─────────────────────────────────────────────────────────

    async def chat(
        self,
        kb_id: int,
        question: str,
        history: list[dict[str, Any]] | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
        conv_id: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        history = history or []
        async with tracer.span("rag.request", kind="chat", kb_id=kb_id, session_id=conv_id) as root:
            results, cache_hit = await self.retrieve(kb_id, question, strategy=strategy)
            root.set(cache_hit=cache_hit)
            context, sources = await self._build_context_block(kb_id, results)
            self._log_prompt(kb_id, question, context, history)

            if not results:
                root.set(empty_result=True)
                return "根据当前知识库内容，未找到与该问题相关的信息。", sources

            llm, messages, provider, model = self._prepare_llm(history, context, question)
            async with tracer.span("llm.generate", provider=provider, model=model) as span:
                started = time.perf_counter()
                ai = cast("AIMessage", await llm.ainvoke(messages))
                span.set(latency_ms=round((time.perf_counter() - started) * 1000, 2))
                usage = _token_usage(ai)
                if usage:
                    span.set(**usage)
            record_llm_tokens(
                provider, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            )
            return StrOutputParser().invoke(ai), sources

    # ── SSE 流式 ───────────────────────────────────────────────────────

    async def chat_stream(
        self,
        kb_id: int,
        question: str,
        history: list[dict[str, Any]] | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
        conv_id: int | None = None,
    ) -> AsyncGenerator[str]:
        history = history or []
        async with tracer.span("rag.request", kind="chat", kb_id=kb_id, session_id=conv_id) as root:
            results, cache_hit = await self.retrieve(kb_id, question, strategy=strategy)
            root.set(cache_hit=cache_hit)
            context, sources = await self._build_context_block(kb_id, results)
            self._log_prompt(kb_id, question, context, history)

            yield sse_event("sources", sources)

            if not results:
                root.set(empty_result=True)
                message = "根据当前知识库内容，未找到与该问题相关的信息。"
                yield sse_event("token", message)
                yield sse_event("done", message)
                return

            llm, messages, provider, model = self._prepare_llm(history, context, question)
            async with tracer.span("llm.generate", provider=provider, model=model) as span:
                started = time.perf_counter()
                full_answer = ""
                ttfb_recorded = False
                usage: dict[str, int] = {}
                async for chunk in llm.astream(messages):
                    if not ttfb_recorded:
                        # 首 token 延迟是流式体验的关键指标：只记第一个 chunk 之前的那段时间
                        ttfb_ms = (time.perf_counter() - started) * 1000
                        span.set(ttfb_ms=round(ttfb_ms, 2))
                        metrics.observe("rag_llm_ttfb_ms", ttfb_ms, labels={"provider": provider})
                        ttfb_recorded = True
                    text = chunk.text
                    full_answer += text
                    usage = _token_usage(chunk) or usage
                    yield sse_event("token", text)
                if usage:
                    span.set(**usage)
            record_llm_tokens(
                provider, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            )

            yield sse_event("done", full_answer)


rag_service = RAGService()
