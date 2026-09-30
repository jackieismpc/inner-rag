"""对话 API：RAG 问答（非流式 / SSE 流式）与会话管理。

本层只做参数校验、判权与序列化；会话与消息的读写都在 ``ConversationRepository``。
"""

from __future__ import annotations

import json
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger

from inner_rag.api.deps import ensure_kb_access, get_current_user, get_repositories
from inner_rag.core.access import AccessLevel
from inner_rag.core.config import settings
from inner_rag.core.database import SessionLocal
from inner_rag.models import Conversation, Message, User
from inner_rag.providers import ProviderError
from inner_rag.repositories import ConversationRepository, Repositories, build_repositories
from inner_rag.schemas import ChatRequest, ConversationOut, MessageOut, PageData, ResponseModel
from inner_rag.services.rag import DEFAULT_STRATEGY, rag_service, sse_event
from inner_rag.services.vector_store import Strategy

router = APIRouter(prefix="/api/chat", tags=["对话"])

VALID_STRATEGIES = {"similarity", "mmr", "hybrid"}


def _resolve_strategy(value: str | None) -> Strategy:
    if value is None:
        return DEFAULT_STRATEGY
    if value not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的检索策略: {value}（可选 {sorted(VALID_STRATEGIES)}）",
        )
    return cast(Strategy, value)


def _get_or_create_conversation(
    convs: ConversationRepository, kb_id: int, conv_id: int | None, question: str
) -> Conversation:
    if conv_id is not None:
        conv = convs.get(conv_id)
        if conv is not None:
            # 会话必须属于当前 kb：否则可以拿「自己有权限的 kb_id + 别人库的 conv_id」
            # 把别人的会话历史读进 prompt，并把回答写回别人的会话里
            if conv.kb_id != kb_id:
                raise HTTPException(status_code=404, detail="会话不存在")
            return conv

    title = question[:20] + "..." if len(question) > 20 else question
    return convs.create(kb_id=kb_id, title=title)


def _history(convs: ConversationRepository, conv_id: int) -> list[dict[str, Any]]:
    """取最近 HISTORY_MAX_MESSAGES 条历史（仓储保证按时间正序）。"""
    return [
        {"role": row.role, "content": row.content}
        for row in convs.history(conv_id, settings.HISTORY_MAX_MESSAGES)
    ]


def _parse_sse_event(chunk: str) -> tuple[str, Any] | None:
    """把 rag_service 产出的 SSE 文本还原为 (type, data)。"""
    if not chunk.startswith("data: "):
        return None
    try:
        payload = json.loads(chunk[len("data: ") :].strip())
    except json.JSONDecodeError:
        return None
    return payload.get("type"), payload.get("data")


def _save_assistant_message(conv_id: int, content: str, sources: list[dict]) -> None:
    """流式响应结束后补存 AI 回答：使用独立会话，避免请求级 Session 已关闭。"""
    db = SessionLocal()
    try:
        convs = build_repositories(db).convs
        convs.append_message(conv_id, "assistant", content, sources)
        convs.touch(conv_id)
    except Exception as exc:
        logger.error(f"[CHAT] 保存 AI 回复失败 conv={conv_id}: {exc}")
        db.rollback()
    finally:
        db.close()


# ── 会话管理 ───────────────────────────────────────────────────────────


@router.get("/conversations", response_model=ResponseModel)
def list_conversations(
    kb_id: int = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    ensure_kb_access(repos.kbs, kb_id, user, AccessLevel.READ)
    items, total = repos.convs.list_page(kb_id=kb_id, page=page, page_size=page_size)
    return ResponseModel(
        data=PageData(
            total=total,
            items=[ConversationOut.model_validate(item) for item in items],
            page=page,
            page_size=page_size,
        )
    )


@router.get("/conversations/{conv_id}/messages", response_model=ResponseModel)
def get_messages(
    conv_id: int,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    conv = repos.convs.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    ensure_kb_access(repos.kbs, conv.kb_id, user, AccessLevel.READ)
    return ResponseModel(
        data=[MessageOut.model_validate(item) for item in repos.convs.messages(conv_id)]
    )


@router.delete("/conversations/{conv_id}", response_model=ResponseModel)
def delete_conversation(
    conv_id: int,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    conv = repos.convs.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    ensure_kb_access(repos.kbs, conv.kb_id, user, AccessLevel.READ)
    repos.convs.delete(conv)
    return ResponseModel(message="删除成功")


# ── 问答 ───────────────────────────────────────────────────────────────


@router.post("/send", response_model=ResponseModel)
async def send_message(
    body: ChatRequest,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    """发送消息（非流式）。"""
    ensure_kb_access(repos.kbs, body.kb_id, user, AccessLevel.READ)
    strategy = _resolve_strategy(body.strategy)

    conv = _get_or_create_conversation(repos.convs, body.kb_id, body.conv_id, body.question)
    history = _history(repos.convs, conv.id)

    # 先落库用户提问：即使推理失败，提问也不会丢
    repos.convs.append_message(conv.id, "user", body.question)

    try:
        answer, sources = await rag_service.chat(
            body.kb_id, body.question, history, strategy=strategy, conv_id=conv.id
        )
    except ProviderError as exc:
        # 模型后端没配好：503 + 直接给出该怎么改 .env
        logger.error(f"[CHAT] provider 不可用 kb={body.kb_id}: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"[CHAT] 推理失败 kb={body.kb_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    ai_msg: Message = repos.convs.append_message(conv.id, "assistant", answer, sources)
    repos.convs.touch(conv.id)

    return ResponseModel(data={"conv_id": conv.id, "message": MessageOut.model_validate(ai_msg)})


@router.post("/stream")
async def stream_message(
    body: ChatRequest,
    repos: Repositories = Depends(get_repositories),
    user: User = Depends(get_current_user),
):
    """发送消息（SSE 流式）。判权必须在返回 StreamingResponse 之前：否则 403 会被夹在流里发出。"""
    ensure_kb_access(repos.kbs, body.kb_id, user, AccessLevel.READ)
    strategy = _resolve_strategy(body.strategy)

    conv = _get_or_create_conversation(repos.convs, body.kb_id, body.conv_id, body.question)
    # 生成器在响应返回后才执行，此时请求级 Session 已关闭，因此这里只保留标量值
    conv_id = conv.id
    kb_id = body.kb_id
    question = body.question
    history = _history(repos.convs, conv_id)

    repos.convs.append_message(conv_id, "user", question)

    async def event_generator():
        yield sse_event("conv_id", conv_id)
        full_answer = ""
        sources: list[dict] = []
        try:
            async for chunk in rag_service.chat_stream(
                kb_id, question, history, strategy=strategy, conv_id=conv_id
            ):
                event = _parse_sse_event(chunk)
                if event is not None:
                    event_type, data = event
                    if event_type == "sources":
                        sources = data or []
                    elif event_type == "done":
                        full_answer = data or ""
                yield chunk
        except ProviderError as exc:
            logger.error(f"[CHAT] provider 不可用 kb={kb_id}: {exc}")
            yield sse_event("error", str(exc))
        except Exception as exc:
            logger.error(f"[CHAT] 流式推理失败 kb={kb_id}: {exc}")
            yield sse_event("error", str(exc))
        finally:
            if full_answer:
                _save_assistant_message(conv_id, full_answer, sources)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
