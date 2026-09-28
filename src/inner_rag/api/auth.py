"""认证 API：登录与当前用户。

不提供注册（账号由 ``scripts/create_user.py`` 创建），也不提供 logout：
JWT 是无状态的，服务端没有会话可注销，前端丢弃 token 即可——留一个空接口
只会让人误以为「服务端已撤销」。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy.orm import Session

from inner_rag.api.deps import get_current_user, unauthorized
from inner_rag.core.database import get_db
from inner_rag.core.security import (
    create_access_token,
    verify_dummy_password,
    verify_password,
)
from inner_rag.models import User
from inner_rag.schemas import LoginRequest, ResponseModel, TokenOut, UserOut

router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post("/login", response_model=ResponseModel)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """用户名 + 密码换取访问 Token。失败一律 401 且文案相同（不泄露账号是否存在）。"""
    user = db.query(User).filter(User.username == body.username).one_or_none()
    if user is None:
        # 不存在的账号也做一次等量哈希校验，避免用响应时间枚举用户名
        verify_dummy_password(body.password)
        raise unauthorized("用户名或密码错误")

    if not verify_password(body.password, user.password_hash):
        raise unauthorized("用户名或密码错误")

    # 校验密码之后再判停用：不知道密码的人不该能问出「这个账号被停用了」
    if not user.is_active:
        raise unauthorized("账号已停用")

    token, expires_at = create_access_token(user.id)
    logger.info(f"[AUTH] 登录成功 user={user.username} id={user.id}")
    return ResponseModel(
        data=TokenOut(
            access_token=token,
            expires_at=expires_at,
            user=UserOut.model_validate(user),
        )
    )


@router.get("/me", response_model=ResponseModel)
def me(user: User = Depends(get_current_user)):
    """当前登录用户（前端刷新页面后恢复登录态用）。"""
    return ResponseModel(data=UserOut.model_validate(user))
