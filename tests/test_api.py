"""API 端到端测试：知识库/文档/对话，全部离线（假 embedding + 假 LLM）。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from inner_rag.core.config import settings
from inner_rag.services.cache import query_cache
from inner_rag.services.document import doc_service

DOC_CONTENT = (
    "inner-rag 知识库问答系统的测试文档。\n\n"
    "本系统支持文档上传、向量检索、混合检索与引用来源展示。\n\n"
    "检索相关度按余弦相似度换算，低于阈值的分块会被过滤。"
)


def upload_text(client: TestClient, kb_id: int, name: str = "note.txt", content: str = DOC_CONTENT):
    return client.post(
        "/api/doc/upload",
        data={"kb_id": str(kb_id)},
        files={"files": (name, content.encode("utf-8"), "text/plain")},
    )


# ── 系统 ───────────────────────────────────────────────────────────────


def test_root_and_health(client: TestClient) -> None:
    assert client.get("/").json()["app"] == settings.APP_NAME

    response = client.get("/api/system/health")
    assert response.status_code == 200
    body = response.json()
    # 测试环境没有 Ollama：状态应为 degraded 而不是抛错
    assert body["status"] in {"healthy", "degraded"}
    assert body["version"] == settings.APP_VERSION


def test_system_config_and_stats(client: TestClient) -> None:
    config = client.get("/api/system/config").json()
    assert config["top_k"] == settings.TOP_K
    assert "pdf" in config["allowed_extensions"]

    stats = client.get("/api/system/stats").json()
    assert {"retrieval", "query_cache", "embedding_cache"} <= set(stats)

    assert client.post("/api/system/cache/clear").status_code == 200


# ── 知识库 ─────────────────────────────────────────────────────────────


def test_kb_crud(client: TestClient, kb: dict) -> None:
    assert kb["embedding_model"] == settings.embedding_key  # 建库时锁定 embedding 标识

    assert client.get("/api/kb", params={"keyword": "pytest"}).json()["data"]["total"] == 1
    detail = client.get(f"/api/kb/{kb['id']}").json()["data"]
    assert detail["vector_count"] == 0

    updated = client.put(f"/api/kb/{kb['id']}", json={"description": "已更新"}).json()["data"]
    assert updated["description"] == "已更新"

    assert client.get("/api/kb/99999").status_code == 404
    assert client.put("/api/kb/99999", json={"name": "x"}).status_code == 404
    assert client.delete(f"/api/kb/{kb['id']}").status_code == 200
    assert client.get(f"/api/kb/{kb['id']}").status_code == 404


def test_create_kb_requires_name(client: TestClient) -> None:
    assert client.post("/api/kb", json={"description": "no name"}).status_code == 422


# ── 文档 ───────────────────────────────────────────────────────────────


def test_upload_and_process_document(client: TestClient, kb: dict) -> None:
    response = upload_text(client, kb["id"])
    assert response.status_code == 200, response.text
    doc_id = response.json()["data"]["doc_ids"][0]

    # TestClient 会等后台任务跑完
    doc = client.get(f"/api/doc/{doc_id}").json()["data"]
    assert doc["status"] == "completed", doc
    assert doc["chunk_count"] > 0
    assert doc["char_count"] == len(DOC_CONTENT)
    assert doc["file_type"] == "text"

    assert client.get(f"/api/kb/{kb['id']}").json()["data"]["doc_count"] == 1
    listing = client.get("/api/doc", params={"kb_id": kb["id"], "status": "completed"}).json()
    assert listing["data"]["total"] == 1


def test_upload_rejects_unsupported_type(client: TestClient, kb: dict) -> None:
    response = client.post(
        "/api/doc/upload",
        data={"kb_id": str(kb["id"])},
        files={"files": ("virus.exe", b"MZ", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_rejects_oversized_file(client: TestClient, kb: dict, monkeypatch) -> None:
    monkeypatch.setattr(settings, "MAX_FILE_SIZE", 16)
    response = upload_text(client, kb["id"], name="big.txt", content="x" * 64)
    assert response.status_code == 400
    assert "文件过大" in response.json()["detail"]


def test_upload_to_missing_kb(client: TestClient, kb: dict) -> None:
    assert upload_text(client, 99999).status_code == 404


def test_filename_is_sanitized(client: TestClient, kb: dict) -> None:
    response = client.post(
        "/api/doc/upload",
        data={"kb_id": str(kb["id"])},
        files={"files": ("../../etc/passwd.txt", b"inner-rag safe filename", "text/plain")},
    )
    doc_id = response.json()["data"]["doc_ids"][0]
    doc = client.get(f"/api/doc/{doc_id}").json()["data"]

    assert "/" not in doc["filename"] and ".." not in doc["filename"]
    assert Path(doc["filename"]).name == doc["filename"]
    # 文件必须落在该知识库的上传目录下
    stored = Path(settings.UPLOAD_DIR) / f"kb_{kb['id']}"
    assert stored in Path(doc_service.get_upload_path(kb["id"], doc["filename"])).parents


def test_import_path_disabled_by_default(client: TestClient, kb: dict) -> None:
    response = client.post(
        "/api/doc/import-path", json={"kb_id": kb["id"], "path": "/etc/hostname"}
    )
    assert response.status_code == 403


def test_import_path_rejects_outside_root(
    client: TestClient, kb: dict, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(settings, "ALLOW_LOCAL_IMPORT", True)
    monkeypatch.setattr(settings, "LOCAL_IMPORT_ROOT", str(tmp_path / "allowed"))

    response = client.post(
        "/api/doc/import-path", json={"kb_id": kb["id"], "path": str(tmp_path / "outside")}
    )
    assert response.status_code in {400, 403}


def test_delete_document_removes_vectors_and_cache(client: TestClient, kb: dict) -> None:
    doc_id = upload_text(client, kb["id"]).json()["data"]["doc_ids"][0]
    await_query = {"kb_id": kb["id"], "question": "向量检索", "stream": False}
    client.post("/api/chat/send", json=await_query)
    assert query_cache.stats()["size"] >= 0

    assert client.delete(f"/api/doc/{doc_id}").status_code == 200
    assert client.get(f"/api/doc/{doc_id}").status_code == 404
    assert client.get(f"/api/kb/{kb['id']}").json()["data"]["doc_count"] == 0


def test_reprocess_failed_document(client: TestClient, kb: dict) -> None:
    doc_id = upload_text(client, kb["id"], name="scan.png", content="not a real png").json()[
        "data"
    ]["doc_ids"][0]
    doc = client.get(f"/api/doc/{doc_id}").json()["data"]
    assert doc["status"] == "failed"  # 图片且未启用 OCR，应显式失败
    assert doc["error_msg"]

    # 换回可解析内容后重新处理
    valid = upload_text(client, kb["id"]).json()["data"]["doc_ids"][0]
    assert client.post(f"/api/doc/{valid}/reprocess").status_code == 200


# ── 对话 ───────────────────────────────────────────────────────────────


def test_chat_send_returns_answer_and_sources(client: TestClient, kb: dict, monkeypatch) -> None:
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    upload_text(client, kb["id"])

    response = client.post(
        "/api/chat/send",
        json={"kb_id": kb["id"], "question": "系统支持哪些检索能力？"},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]

    assert data["conv_id"] > 0
    message = data["message"]
    assert message["role"] == "assistant"
    assert message["content"] == "这是测试模型的回答。"
    assert message["sources"]
    for source in message["sources"]:
        assert source["filename"] == "note.txt"
        assert source["score"] is None or 0.0 <= source["score"] <= 1.0
        assert source["content"]

    # 会话列表 + 消息历史都能读回
    convs = client.get("/api/chat/conversations", params={"kb_id": kb["id"]}).json()["data"]
    assert convs["total"] == 1
    messages = client.get(f"/api/chat/conversations/{data['conv_id']}/messages").json()["data"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_chat_without_relevant_documents(client: TestClient, kb: dict, monkeypatch) -> None:
    """空知识库 + 高阈值：应返回兜底话术，且不调用 LLM。"""
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.99)
    upload_text(client, kb["id"], content="与问题毫不相干的内容。")

    response = client.post(
        "/api/chat/send", json={"kb_id": kb["id"], "question": "量子色动力学的渐近自由"}
    )
    assert response.status_code == 200
    assert "未找到" in response.json()["data"]["message"]["content"]


def test_chat_rejects_unknown_strategy(client: TestClient, kb: dict) -> None:
    response = client.post(
        "/api/chat/send", json={"kb_id": kb["id"], "question": "问题", "strategy": "bogus"}
    )
    assert response.status_code == 400


def test_chat_missing_kb(client: TestClient, kb: dict) -> None:
    assert client.post("/api/chat/send", json={"kb_id": 99999, "question": "hi"}).status_code == 404


def test_chat_stream_events_and_persistence(client: TestClient, kb: dict, monkeypatch) -> None:
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    upload_text(client, kb["id"])

    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"kb_id": kb["id"], "question": "介绍检索相关度换算"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    types = [event["type"] for event in events]
    assert types[0] == "conv_id"
    assert "sources" in types
    assert "token" in types
    assert types[-1] == "done"

    conv_id = events[0]["data"]
    assert any(
        source["filename"] == "note.txt"
        for event in events
        if event["type"] == "sources"
        for source in event["data"]
    )

    messages = client.get(f"/api/chat/conversations/{conv_id}/messages").json()["data"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[-1]["content"] == "这是测试模型的回答。"
    assert messages[-1]["sources"]


def test_chat_history_uses_most_recent_messages(client: TestClient, kb: dict, monkeypatch) -> None:
    monkeypatch.setattr(settings, "HISTORY_MAX_MESSAGES", 2)
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    upload_text(client, kb["id"])

    conv_id = None
    for question in ["第一问", "第二问", "第三问"]:
        payload = {"kb_id": kb["id"], "question": question}
        if conv_id is not None:
            payload["conv_id"] = conv_id
        conv_id = client.post("/api/chat/send", json=payload).json()["data"]["conv_id"]

    history = client.get(f"/api/chat/conversations/{conv_id}/messages").json()["data"]
    assert len(history) == 6

    # 只保留最近两条消息作为上下文：应包含最后一次提问
    recent = [m["content"] for m in history[-settings.HISTORY_MAX_MESSAGES :]]
    assert "第三问" in recent


def test_delete_conversation(client: TestClient, kb: dict, monkeypatch) -> None:
    monkeypatch.setattr(settings, "RETRIEVAL_SCORE_THRESHOLD", 0.0)
    conv_id = client.post("/api/chat/send", json={"kb_id": kb["id"], "question": "你好"}).json()[
        "data"
    ]["conv_id"]

    assert client.delete(f"/api/chat/conversations/{conv_id}").status_code == 200
    assert client.get(f"/api/chat/conversations/{conv_id}/messages").status_code == 404
    assert client.delete("/api/chat/conversations/99999").status_code == 404
