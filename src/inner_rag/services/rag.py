"""RAG 推理服务：检索 -> 组装 Prompt -> 调用 LLM（非流式与 SSE 流式）。

模型实例全部来自 ``inner_rag.providers``：本模块不关心后端是 Ollama、
OpenRouter、DeepSeek、OpenAI 还是离线 mock。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from inner_rag.core.config import settings
from inner_rag.providers import get_chat_model
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


class RAGService:
    def _get_llm(self) -> BaseChatModel:
        """获取对话模型（由 LLM_PROVIDER 决定，实例在 provider 工厂内复用）。

        配置非法（未知 provider / 缺 API Key）时抛 ProviderError，
        由 API 层映射成 503 + 可读提示。
        """
        return get_chat_model()

    def _build_chain(self, history: list[dict[str, Any]]):
        prompt = ChatPromptTemplate.from_messages(self._build_prompt_messages(history))
        return prompt | self._get_llm() | StrOutputParser()

    # ── 检索（QueryCache + 日志）───────────────────────────────────────

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

        cached = await query_cache.get(kb_id, query, k)
        if cached is not None:
            latency_ms = (time.perf_counter() - started) * 1000
            log_retrieval(kb_id, query, cached, latency_ms, True, strategy)
            return cached, True

        results, filtered_out = await vector_service.search(
            kb_id=kb_id,
            query=query,
            k=k,
            strategy=strategy,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        log_retrieval(
            kb_id, query, results, latency_ms, False, strategy, filtered_count=filtered_out
        )

        if results:
            await query_cache.set(kb_id, query, k, results)
        return results, False

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

    # ── 非流式 ─────────────────────────────────────────────────────────

    async def chat(
        self,
        kb_id: int,
        question: str,
        history: list[dict[str, Any]] | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ) -> tuple[str, list[dict[str, Any]]]:
        history = history or []
        results, _ = await self.retrieve(kb_id, question, strategy=strategy)
        context, sources = self._build_context(results)
        self._log_prompt(kb_id, question, context, history)

        if not results:
            return "根据当前知识库内容，未找到与该问题相关的信息。", sources

        chain = self._build_chain(history)
        answer = await chain.ainvoke({"context": context, "question": question})
        return answer, sources

    # ── SSE 流式 ───────────────────────────────────────────────────────

    async def chat_stream(
        self,
        kb_id: int,
        question: str,
        history: list[dict[str, Any]] | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ) -> AsyncGenerator[str]:
        history = history or []
        results, _ = await self.retrieve(kb_id, question, strategy=strategy)
        context, sources = self._build_context(results)
        self._log_prompt(kb_id, question, context, history)

        yield sse_event("sources", sources)

        if not results:
            message = "根据当前知识库内容，未找到与该问题相关的信息。"
            yield sse_event("token", message)
            yield sse_event("done", message)
            return

        chain = self._build_chain(history)
        full_answer = ""
        async for chunk in chain.astream({"context": context, "question": question}):
            full_answer += chunk
            yield sse_event("token", chunk)

        yield sse_event("done", full_answer)


rag_service = RAGService()
