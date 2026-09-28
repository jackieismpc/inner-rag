"""认证与授权：401（未登录）、403（跨库越权）、Token 与密码哈希、知识库 ACL。

用例只铺「唯一的失效场景」：
* token 伪造的各种变体用 parametrize 覆盖，不重复写用例；
* 跨库越权的各接口用一张路由表 parametrize，漏挂鉴权时能直接指出是哪个接口。
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient

from conftest import TEST_PASSWORD, bearer, create_user, login, set_user_active
from inner_rag.core.config import settings
from inner_rag.core.security import (
    DEFAULT_SECRET,
    create_access_token,
    hash_password,
    verify_password,
    verify_production_secret,
)
from inner_rag.models import User

# ≥32 字节：HS256 密钥短于哈希输出长度时 PyJWT 会告警，且可被离线爆破
ATTACKER_SECRET = "attacker-secret-attacker-secret-0123456789"


def upload_text(
    client: TestClient, kb_id: int, name: str = "note.txt", content: str = "文档内容。"
):
    return client.post(
        "/api/doc/upload",
        data={"kb_id": str(kb_id)},
        files={"files": (name, content.encode("utf-8"), "text/plain")},
    )


# ── 未登录（401） ──────────────────────────────────────────────────────

# 覆盖面 = 每个业务模块至少一个代表接口 + 写操作接口；漏挂 Depends(get_current_user) 会在这里红
PROTECTED_ROUTES: list[tuple[str, str, dict[str, Any]]] = [
    ("GET", "/api/kb", {}),
    ("POST", "/api/kb", {"json": {"name": "x"}}),
    ("GET", "/api/doc", {"params": {"kb_id": 1}}),
    ("POST", "/api/chat/send", {"json": {"kb_id": 1, "question": "hi"}}),
    ("POST", "/api/chat/stream", {"json": {"kb_id": 1, "question": "hi"}}),
    ("GET", "/api/chat/conversations", {"params": {"kb_id": 1}}),
    ("GET", "/api/system/config", {}),
    ("GET", "/api/system/stats", {}),
    ("POST", "/api/system/cache/clear", {}),
    ("GET", "/api/auth/me", {}),
]


@pytest.mark.parametrize(
    "method, path, kwargs",
    PROTECTED_ROUTES,
    ids=[f"{method} {path}" for method, path, _ in PROTECTED_ROUTES],
)
def test_anonymous_request_is_unauthorized(
    anonymous_client: TestClient, method: str, path: str, kwargs: dict[str, Any]
) -> None:
    response = anonymous_client.request(method, path, **kwargs)
    assert response.status_code == 401
    # 客户端靠这个头区分「未登录（401，去登录页）」与「无权限（403）」
    assert response.headers["www-authenticate"] == "Bearer"


def test_health_is_public(anonymous_client: TestClient) -> None:
    """探活接口供负载均衡 / 容器编排调用，必须免鉴权，且依赖异常时也不能 5xx。"""
    response = anonymous_client.get("/api/system/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"healthy", "degraded"}


def test_auth_me_rejects_token_of_deleted_user(anonymous_client: TestClient, user: User) -> None:
    """用户被删除后（库里已无此人），旧 token 必须失效。"""
    token, _ = create_access_token(user.id + 10_000_000)
    assert anonymous_client.get("/api/auth/me", headers=bearer(token)).status_code == 401


# ── 登录 ───────────────────────────────────────────────────────────────


def test_login_returns_token_and_user(anonymous_client: TestClient, user: User) -> None:
    response = anonymous_client.post(
        "/api/auth/login", json={"username": user.username, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["token_type"] == "bearer"
    assert len(data["access_token"].split(".")) == 3
    assert datetime.fromisoformat(data["expires_at"]) > datetime.now(UTC)
    assert data["user"] == {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
    }

    me = anonymous_client.get("/api/auth/me", headers=bearer(data["access_token"]))
    assert me.json()["data"]["username"] == user.username


def test_login_failure_message_does_not_leak_account_existence(
    anonymous_client: TestClient, user: User
) -> None:
    unknown = anonymous_client.post(
        "/api/auth/login", json={"username": "no-such-user", "password": TEST_PASSWORD}
    )
    wrong = anonymous_client.post(
        "/api/auth/login", json={"username": user.username, "password": "wrong-password"}
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["detail"] == wrong.json()["detail"] == "用户名或密码错误"


def test_disabled_account_cannot_login(anonymous_client: TestClient) -> None:
    disabled = create_user(is_active=False)

    wrong = anonymous_client.post(
        "/api/auth/login", json={"username": disabled.username, "password": "wrong"}
    )
    # 先校验密码再判停用：不知道密码的人问不出「该账号已停用」
    assert wrong.json()["detail"] == "用户名或密码错误"

    right = anonymous_client.post(
        "/api/auth/login", json={"username": disabled.username, "password": TEST_PASSWORD}
    )
    assert right.status_code == 401
    assert right.json()["detail"] == "账号已停用"


def test_token_is_rejected_after_account_disabled(anonymous_client: TestClient, user: User) -> None:
    """JWT 无状态：停用后不是靠撤销列表，而是每次请求都回查用户表的 is_active。"""
    token = login(anonymous_client, user)
    assert anonymous_client.get("/api/kb", headers=bearer(token)).status_code == 200

    set_user_active(user.id, False)
    assert anonymous_client.get("/api/kb", headers=bearer(token)).status_code == 401


# ── Token 与密码哈希 ───────────────────────────────────────────────────


def _swap_payload(token: str, **changes: Any) -> str:
    """改写 payload 但保留原签名（模拟「客户端篡改 token 想换个身份」）。"""
    header, payload, signature = token.split(".")
    decoded = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    decoded.update(changes)
    forged = base64.urlsafe_b64encode(json.dumps(decoded).encode()).rstrip(b"=").decode()
    return f"{header}.{forged}.{signature}"


@pytest.fixture
def forged_tokens(user: User, other_user: User) -> dict[str, str]:
    """一组「不该被接受」的 token：过期 / 换密钥 / 改载荷 / 无签名 / 缺 exp。"""
    now = datetime.now(UTC)
    claims = {"sub": str(user.id), "iat": now, "exp": now + timedelta(minutes=5)}

    expired, _ = create_access_token(user.id, ttl_minutes=-1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "AUTH_SECRET_KEY", ATTACKER_SECRET)
        wrong_secret, _ = create_access_token(user.id)

    valid, _ = create_access_token(user.id)
    return {
        "expired": expired,
        "wrong-secret": wrong_secret,
        "tampered-sub": _swap_payload(valid, sub=str(other_user.id)),
        "alg-none": jwt.encode(claims, key=None, algorithm="none"),
        # 没有 exp 的 token 必须被拒，否则「一次签发、永不过期」
        "no-exp": jwt.encode(
            {"sub": str(user.id), "iat": now}, settings.AUTH_SECRET_KEY, algorithm="HS256"
        ),
    }


def test_forged_tokens_are_rejected(
    anonymous_client: TestClient, forged_tokens: dict[str, str]
) -> None:
    for name, token in forged_tokens.items():
        response = anonymous_client.get("/api/auth/me", headers=bearer(token))
        assert response.status_code == 401, f"{name}: {response.text}"


def test_password_hash_is_salted_and_irreversible() -> None:
    first, second = hash_password("hunter2"), hash_password("hunter2")
    assert first != second  # 每次随机盐：库拖走也不能靠彩虹表还原
    assert first.startswith("$argon2id$") and "hunter2" not in first
    assert verify_password("hunter2", first)
    assert not verify_password("hunter3", first)
    # 迁移给历史知识库补的 bootstrap 账号是哨兵哈希 "!"，任何密码都验不过
    assert not verify_password("hunter2", "!")


@pytest.mark.parametrize(
    "debug, secret, should_raise",
    [
        (False, DEFAULT_SECRET, True),
        (False, "short-secret", True),
        (False, ATTACKER_SECRET, False),
        (True, DEFAULT_SECRET, False),
        (True, "short-secret", False),
    ],
    ids=[
        "production-rejects-default",
        "production-rejects-short",
        "production-accepts-random",
        "debug-allows-default",
        "debug-allows-short",
    ],
)
def test_production_secret_guard(monkeypatch, debug: bool, secret: str, should_raise: bool) -> None:
    """生产环境不允许「默认密钥」或「短密钥」；开发环境不拦（本地跑得起来优先）。"""
    monkeypatch.setattr(settings, "DEBUG", debug)
    monkeypatch.setattr(settings, "AUTH_SECRET_KEY", secret)
    if should_raise:
        with pytest.raises(RuntimeError, match="AUTH_SECRET_KEY"):
            verify_production_secret()
    else:
        verify_production_secret()


# ── 跨用户隔离（403） ──────────────────────────────────────────────────

# 占位符 {kb_id} / {doc_id} 在用例里替换：kb 属于 client 的用户，other_client 是无权限的第三方
CROSS_KB_ROUTES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "kb-detail": ("GET", "/api/kb/{kb_id}", {}),
    "kb-update": ("PUT", "/api/kb/{kb_id}", {"json": {"name": "hijack"}}),
    "kb-delete": ("DELETE", "/api/kb/{kb_id}", {}),
    "kb-members": ("GET", "/api/kb/{kb_id}/members", {}),
    "doc-list": ("GET", "/api/doc", {"params": {"kb_id": "{kb_id}"}}),
    "doc-detail": ("GET", "/api/doc/{doc_id}", {}),
    "doc-delete": ("DELETE", "/api/doc/{doc_id}", {}),
    "doc-reprocess": ("POST", "/api/doc/{doc_id}/reprocess", {}),
    "conv-list": ("GET", "/api/chat/conversations", {"params": {"kb_id": "{kb_id}"}}),
    "chat-send": ("POST", "/api/chat/send", {"json": {"kb_id": "{kb_id}", "question": "套话"}}),
    "chat-stream": ("POST", "/api/chat/stream", {"json": {"kb_id": "{kb_id}", "question": "套话"}}),
}


def _fill(value: Any, kb_id: int, doc_id: int) -> Any:
    if isinstance(value, str):
        return value.format(kb_id=kb_id, doc_id=doc_id)
    if isinstance(value, dict):
        return {key: _fill(item, kb_id, doc_id) for key, item in value.items()}
    return value


@pytest.mark.parametrize("route", list(CROSS_KB_ROUTES), ids=list(CROSS_KB_ROUTES))
def test_other_user_cannot_touch_kb(
    client: TestClient, other_client: TestClient, kb: dict, doc_id: int, route: str
) -> None:
    method, path, kwargs = CROSS_KB_ROUTES[route]
    # 对照组：有权限的人做同一件事是 200，证明 403 是权限判定而不是接口本身坏了
    # （chat 接口的对照组放在 test_chat_cannot_reuse_... 之外的单测里，这里只保证库归属正确）
    assert client.get(f"/api/kb/{kb['id']}").status_code == 200

    response = other_client.request(
        method, path.format(kb_id=kb["id"], doc_id=doc_id), **_fill(kwargs, kb["id"], doc_id)
    )
    assert response.status_code == 403, f"{route}: {response.status_code} {response.text}"


def test_kb_list_only_shows_accessible(
    kb: dict, client: TestClient, other_client: TestClient
) -> None:
    theirs = other_client.post("/api/kb", json={"name": "别人的库"}).json()["data"]

    mine_ids = {
        item["id"]
        for item in client.get("/api/kb", params={"page_size": 100}).json()["data"]["items"]
    }
    their_ids = {
        item["id"]
        for item in other_client.get("/api/kb", params={"page_size": 100}).json()["data"]["items"]
    }
    assert kb["id"] in mine_ids and theirs["id"] not in mine_ids
    assert theirs["id"] in their_ids and kb["id"] not in their_ids

    detail = client.get(f"/api/kb/{kb['id']}").json()["data"]
    assert detail["my_permission"] == "owner"


def test_chat_cannot_reuse_conversation_from_another_kb(
    client: TestClient, other_client: TestClient, kb: dict
) -> None:
    """回归：拿「自己有权限的 kb_id」配「别人库的 conv_id」曾能读写别人的会话。"""
    other_kb = other_client.post("/api/kb", json={"name": "别人的库"}).json()["data"]
    conv_id = other_client.post(
        "/api/chat/send", json={"kb_id": other_kb["id"], "question": "秘密"}
    ).json()["data"]["conv_id"]

    response = client.post(
        "/api/chat/send", json={"kb_id": kb["id"], "question": "套话", "conv_id": conv_id}
    )
    assert response.status_code == 404

    history = other_client.get(f"/api/chat/conversations/{conv_id}/messages").json()["data"]
    assert [message["role"] for message in history] == ["user", "assistant"]
    assert history[0]["content"] == "秘密"
    assert "套话" not in "".join(message["content"] for message in history)


# ── 成员授权 ───────────────────────────────────────────────────────────


def test_member_with_read_permission_cannot_write(
    client: TestClient, other_client: TestClient, other_user: User, kb: dict, doc_id: int
) -> None:
    granted = client.post(
        f"/api/kb/{kb['id']}/members",
        json={"username": other_user.username, "permission": "read"},
    )
    assert granted.status_code == 200, granted.text

    assert other_client.get(f"/api/kb/{kb['id']}").json()["data"]["my_permission"] == "read"
    assert other_client.get("/api/doc", params={"kb_id": kb["id"]}).status_code == 200
    assert other_client.get(f"/api/doc/{doc_id}").status_code == 200
    assert (
        other_client.post(
            "/api/chat/send", json={"kb_id": kb["id"], "question": "你好"}
        ).status_code
        == 200
    )

    assert upload_text(other_client, kb["id"], name="new.txt").status_code == 403
    assert other_client.delete(f"/api/doc/{doc_id}").status_code == 403
    assert other_client.put(f"/api/kb/{kb['id']}", json={"name": "改名"}).status_code == 403
    assert other_client.delete(f"/api/kb/{kb['id']}").status_code == 403
    assert other_client.get(f"/api/kb/{kb['id']}/members").status_code == 403


def test_member_with_write_permission_cannot_manage_kb(
    client: TestClient, other_client: TestClient, other_user: User, kb: dict
) -> None:
    client.post(
        f"/api/kb/{kb['id']}/members",
        json={"username": other_user.username, "permission": "write"},
    )

    assert other_client.get(f"/api/kb/{kb['id']}").json()["data"]["my_permission"] == "write"
    assert upload_text(other_client, kb["id"], name="by-member.txt").status_code == 200
    assert other_client.get("/api/doc", params={"kb_id": kb["id"]}).json()["data"]["total"] == 1

    assert other_client.put(f"/api/kb/{kb['id']}", json={"name": "改名"}).status_code == 403
    assert other_client.delete(f"/api/kb/{kb['id']}").status_code == 403


def test_member_management(
    client: TestClient, other_client: TestClient, other_user: User, user: User, kb: dict
) -> None:
    members_url = f"/api/kb/{kb['id']}/members"
    assert (
        client.post(
            members_url, json={"username": other_user.username, "permission": "read"}
        ).status_code
        == 200
    )

    listed = client.get(members_url).json()["data"]
    assert listed["owner"]["username"] == user.username
    assert [(item["username"], item["permission"]) for item in listed["items"]] == [
        (other_user.username, "read")
    ]

    # 重复授权即改权限（幂等），不需要单独的「改权限」接口
    client.post(members_url, json={"username": other_user.username, "permission": "write"})
    assert client.get(members_url).json()["data"]["items"][0]["permission"] == "write"

    assert (
        client.post(
            members_url, json={"username": "no-such-user", "permission": "read"}
        ).status_code
        == 404
    )
    assert (
        client.post(members_url, json={"username": user.username, "permission": "read"}).status_code
        == 400
    )

    assert client.delete(f"{members_url}/{other_user.id}").status_code == 200
    # 移除后立刻失去权限（每次请求实时判权，没有缓存的权限副本）
    assert other_client.get(f"/api/kb/{kb['id']}").status_code == 403
    assert client.delete(f"{members_url}/{other_user.id}").status_code == 404
