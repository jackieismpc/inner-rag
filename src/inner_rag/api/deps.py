"""请求级身份与权限依赖。

分工：身份解析与 HTTP 语义（401/403/404）在这里，权限策略本身在 ``core/access.py``。
路由统一用 ``ensure_kb_access`` / ``ensure_doc_access`` 判权，并让一次判权顺带把
目标对象取回来（省一次查询，也避免「判权用 kb A、操作用 kb B」这类错配）。
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from inner_rag.core.access import AccessLevel, level_of
from inner_rag.core.database import get_db
from inner_rag.core.security import InvalidToken, decode_access_token
from inner_rag.models import Document, KnowledgeBase, User

# auto_error=False：缺少 Authorization 头时由我们给出统一文案与 WWW-Authenticate 头，
# 而不是 FastAPI 默认的 403（语义上「未登录」必须是 401）
_bearer = HTTPBearer(auto_error=False)


def unauthorized(detail: str = "未登录或登录已过期") -> HTTPException:
    """401 的唯一出口：带上 Bearer 挑战头，方便客户端区分「需要登录」与「无权限」。"""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    """解析 Bearer token 并加载用户。

    缺失 / 过期 / 签名不符 / 用户不存在 / 账号已停用，一律 401（不区分原因，
    避免给攻击者反馈）；「已登录但没权限」才是 403。
    """
    if credentials is None:
        raise unauthorized()
    try:
        user_id = decode_access_token(credentials.credentials)
    except InvalidToken:
        raise unauthorized() from None

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise unauthorized()
    return user


def ensure_kb_access(
    db: Session, kb_id: int, user: User, level: AccessLevel
) -> tuple[KnowledgeBase, AccessLevel]:
    """校验当前用户对知识库的权限，返回 ``(知识库, 实际权限级别)``。

    一次调用同时回答「能不能用」和「以什么身份用」：出参里的 ``my_permission`` 正需要后者，
    顺手带回可以少一次重复判权。库不存在 → 404（与既有语义一致）；存在但级别不够 → 403。
    """
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise HTTPException(status_code=404, detail="知识库不存在")

    actual = level_of(db, kb, user)
    if actual is None or actual < level:
        raise HTTPException(
            status_code=403,
            detail=f"无权访问该知识库（需要 {level.name.lower()} 权限）",
        )
    return kb, actual


def ensure_doc_access(db: Session, doc_id: int, user: User, level: AccessLevel) -> Document:
    """文档级路由的入口：按文档所属知识库判权。"""
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    ensure_kb_access(db, doc.kb_id, user, level)
    return doc
