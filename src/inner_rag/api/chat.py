"""对话 API：RAG 问答（非流式 / SSE 流式）与会话管理。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.orm import Session

from inner_rag.core.config import settings
from inner_rag.core.database import SessionLocal, get_db
from inner_rag.models import Conversation, KnowledgeBase, Message
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
    db: Session, kb_id: int, conv_id: int | None, question: str
) -> Conversation:
    if conv_id is not None:
        conv = db.get(Conversation, conv_id)
        if conv is not None:
            return conv

    title = question[:20] + "..." if len(question) > 20 else question
    conv = Conversation(kb_id=kb_id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def _get_history(db: Session, conv_id: int) -> list[dict[str, Any]]:
    """取最近 HISTORY_MAX_MESSAGES 条历史，并按时间正序返回。

    旧实现是 ``order_by(created_at.asc()).limit(20)``，拿到的是**最早**的 20 条，
    长对话里等于完全丢失近期上下文。
    """
    rows = (
        db.query(Message)
        .filter(Message.conv_id == conv_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(settings.HISTORY_MAX_MESSAGES)
        .all()
    )
    return [{"role": row.role, "content": row.content} for row in reversed(rows)]


def _touch_conversation(db: Session, conv_id: int) -> None:
    """刷新会话活跃时间，保证会话列表按最近使用排序。"""
    db.query(Conversation).filter(Conversation.id == conv_id).update(
        {"updated_at": datetime.now(UTC)}
    )


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
        db.add(Message(conv_id=conv_id, role="assistant", content=content, sources=sources))
        _touch_conversation(db, conv_id)
        db.commit()
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
    db: Session = Depends(get_db),
):
    query = db.query(Conversation).filter(Conversation.kb_id == kb_id)
    total = query.count()
    items = (
        query.order_by(Conversation.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ResponseModel(
        data=PageData(
            total=total,
            items=[ConversationOut.model_validate(item) for item in items],
            page=page,
            page_size=page_size,
        )
    )


@router.get("/conversations/{conv_id}/messages", response_model=ResponseModel)
def get_messages(conv_id: int, db: Session = Depends(get_db)):
    if db.get(Conversation, conv_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = (
        db.query(Message)
        .filter(Message.conv_id == conv_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .all()
    )
    return ResponseModel(data=[MessageOut.model_validate(item) for item in messages])


@router.delete("/conversations/{conv_id}", response_model=ResponseModel)
def delete_conversation(conv_id: int, db: Session = Depends(get_db)):
    conv = db.get(Conversation, conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.delete(conv)
    db.commit()
    return ResponseModel(message="删除成功")


# ── 问答 ───────────────────────────────────────────────────────────────


@router.post("/send", response_model=ResponseModel)
async def send_message(body: ChatRequest, db: Session = Depends(get_db)):
    """发送消息（非流式）。"""
    if db.get(KnowledgeBase, body.kb_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    strategy = _resolve_strategy(body.strategy)

    conv = _get_or_create_conversation(db, body.kb_id, body.conv_id, body.question)
    history = _get_history(db, conv.id)

    # 先落库用户提问：即使推理失败，提问也不会丢
    db.add(Message(conv_id=conv.id, role="user", content=body.question))
    db.commit()

    try:
        answer, sources = await rag_service.chat(
            body.kb_id, body.question, history, strategy=strategy
        )
    except Exception as exc:
        logger.error(f"[CHAT] 推理失败 kb={body.kb_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"推理失败: {exc}") from exc

    ai_msg = Message(conv_id=conv.id, role="assistant", content=answer, sources=sources)
    db.add(ai_msg)
    _touch_conversation(db, conv.id)
    db.commit()
    db.refresh(ai_msg)

    return ResponseModel(data={"conv_id": conv.id, "message": MessageOut.model_validate(ai_msg)})


@router.post("/stream")
async def stream_message(body: ChatRequest, db: Session = Depends(get_db)):
    """发送消息（SSE 流式）。"""
    if db.get(KnowledgeBase, body.kb_id) is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    strategy = _resolve_strategy(body.strategy)

    conv = _get_or_create_conversation(db, body.kb_id, body.conv_id, body.question)
    # 生成器在响应返回后才执行，此时请求级 Session 已关闭，因此这里只保留标量值
    conv_id = conv.id
    kb_id = body.kb_id
    question = body.question
    history = _get_history(db, conv_id)

    db.add(Message(conv_id=conv_id, role="user", content=question))
    db.commit()

    async def event_generator():
        yield sse_event("conv_id", conv_id)
        full_answer = ""
        sources: list[dict] = []
        try:
            async for chunk in rag_service.chat_stream(kb_id, question, history, strategy=strategy):
                event = _parse_sse_event(chunk)
                if event is not None:
                    event_type, data = event
                    if event_type == "sources":
                        sources = data or []
                    elif event_type == "done":
                        full_answer = data or ""
                yield chunk
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
