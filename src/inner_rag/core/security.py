"""密码哈希与访问令牌（JWT）。

职责边界：这里只有密码学原语，不碰数据库、不抛 HTTP 状态码。
请求级鉴权（401 的映射、账号是否停用）在 ``api/deps.py``，知识库 ACL 在 ``core/access.py``。

密码只存 argon2id 哈希；参数写在哈希串里（``$argon2id$v=19$m=...``），
因此将来调参不会让旧哈希失效，也不需要额外的配置旋钮。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Final

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from inner_rag.core.config import settings

ALGORITHM: Final = "HS256"
# 默认值本身就是「请改掉」的哨兵（比对用），同时必须 ≥32 字节：HS256 的密钥短于
# 哈希输出长度时，攻击者拿到一个 token 就能离线爆破密钥（PyJWT 也会告警）。
DEFAULT_SECRET: Final = "dev-only-insecure-secret-change-me-in-production"
MIN_SECRET_BYTES: Final = 32

_hasher: Final = PasswordHasher()

# 未知用户名时也要走一次真实校验，否则「用户不存在」会因为立即返回而快出一个数量级，
# 等于把用户名单送给攻击者（响应时间枚举）。这个哈希对应的明文无意义，只用于对齐耗时。
_TIMING_EQUALIZER_HASH: Final = "$argon2id$v=19$m=65536,t=3,p=4$4ePwQoBEchkbUfQtTxtWcA$055cFJp+VFYHu7O8P3KJY+VO2zDDjpIPDZq1zVQZl74"


class InvalidToken(Exception):
    """Token 缺失 / 过期 / 签名不符 / 结构非法。调用方一律映射成 401。"""


def hash_password(password: str) -> str:
    """生成 argon2id 哈希。自带随机盐，同一密码两次的结果不同。"""
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """校验密码。

    哈希串非法（例如迁移里的 bootstrap 哨兵值 ``"!"``）返回 False 而不是抛异常：
    「账号存在但尚未设置密码」是真实状态，表现为「密码不对」即可。
    """
    try:
        _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False
    return True


def verify_dummy_password(password: str) -> None:
    """给不存在的用户名消耗一次等量耗时（见 ``_TIMING_EQUALIZER_HASH``）。结果丢弃。"""
    verify_password(password, _TIMING_EQUALIZER_HASH)


def create_access_token(
    user_id: int,
    ttl_minutes: int | None = None,
    now: datetime | None = None,
) -> tuple[str, datetime]:
    """签发 HS256 JWT，返回 ``(token, 过期时间)``。

    ``ttl_minutes`` / ``now`` 可注入：测试要构造「已过期」的 token，生产一律用配置值。
    JWT 无状态，服务端不保存会话，因此撤销能力等同于 TTL 的长度（默认 12 小时）。
    """
    issued_at = now or datetime.now(UTC)
    ttl = settings.AUTH_TOKEN_TTL_MINUTES if ttl_minutes is None else ttl_minutes
    expires_at = issued_at + timedelta(minutes=ttl)
    payload = {"sub": str(user_id), "iat": issued_at, "exp": expires_at}
    return jwt.encode(payload, settings.AUTH_SECRET_KEY, algorithm=ALGORITHM), expires_at


def decode_access_token(token: str) -> int:
    """校验签名与有效期，返回 user_id。

    显式限定算法并强制三个声明都存在：不限定算法是被广泛利用的 JWT 漏洞
    （算法混淆 / ``alg=none``），强制声明则避免「缺少 exp 就永不过期」。
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.AUTH_SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidToken(str(exc)) from exc

    try:
        return int(payload["sub"])
    except (TypeError, ValueError) as exc:
        raise InvalidToken("token 里的 sub 不是合法的用户 ID") from exc


def verify_production_secret() -> None:
    """生产环境（``DEBUG=false``）拒绝用「默认密钥」或「过短的密钥」启动。

    默认密钥公开写在 ``.env.example`` 里，用它签发的 token 谁都能伪造；密钥短于 32 字节时
    攻击者拿到任意一个 token 就能离线爆破出密钥。与其等第一次登录才暴露，不如启动即失败
    ——这是配置边界，不是无效防御。
    """
    if settings.DEBUG:
        return

    if settings.AUTH_SECRET_KEY == DEFAULT_SECRET:
        msg = (
            "DEBUG=false 时必须把 AUTH_SECRET_KEY 改为随机长密钥（默认值人人可见，可伪造 token）；"
            '生成方式：python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
        raise RuntimeError(msg)

    if len(settings.AUTH_SECRET_KEY.encode()) < MIN_SECRET_BYTES:
        msg = (
            f"AUTH_SECRET_KEY 至少 {MIN_SECRET_BYTES} 字节（当前 "
            f"{len(settings.AUTH_SECRET_KEY.encode())} 字节），否则 HS256 密钥可被离线爆破；"
            '生成方式：python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
        raise RuntimeError(msg)
