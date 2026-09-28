#!/usr/bin/env python
"""运维脚本：创建 / 管理本地账号（系统不提供注册接口）。

用法:
    uv run scripts/create_user.py zhangsan                      # 交互式输入密码
    printf 'pwd\\n' | uv run scripts/create_user.py zhangsan --password-stdin
    uv run scripts/create_user.py zhangsan --reset-password      # 重置密码（bootstrap 账号接管历史库）
    uv run scripts/create_user.py zhangsan --disable             # 停用（保留其知识库）
    uv run scripts/create_user.py --list

不提供 ``--password`` 参数：密码走交互输入或标准输入，避免进入 shell 历史与进程列表。
密码强度由运维把关（脚本只拒绝空密码，过短会给出提示但不阻断）。
"""

from __future__ import annotations

import argparse
import getpass
import sys

from loguru import logger

from inner_rag.core.database import SessionLocal, init_db
from inner_rag.core.security import hash_password
from inner_rag.models import User

WEAK_PASSWORD_LENGTH = 8


def read_password(from_stdin: bool) -> str:
    """读密码：``--password-stdin`` 供脚本化调用，否则交互式输入两次确认。"""
    if from_stdin:
        password = sys.stdin.readline().rstrip("\n")
        if not password:
            raise SystemExit("从标准输入读到的密码为空")
        return password

    password = getpass.getpass("密码: ")
    if not password:
        raise SystemExit("密码不能为空")
    if password != getpass.getpass("再输一次: "):
        raise SystemExit("两次输入的密码不一致")
    return password


def warn_if_weak(password: str) -> None:
    if len(password) < WEAK_PASSWORD_LENGTH:
        logger.warning(f"密码长度不足 {WEAK_PASSWORD_LENGTH} 位，仅建议用于本地测试环境")


def list_users() -> None:
    db = SessionLocal()
    try:
        users = db.query(User).order_by(User.id).all()
        if not users:
            logger.info("没有任何账号；用 `uv run scripts/create_user.py <username>` 创建")
            return
        for user in users:
            state = "启用" if user.is_active else "停用"
            logger.info(
                f"id={user.id:<4} {user.username:<20} {state}  "
                f"显示名={user.display_name or '-'} 创建于={user.created_at}"
            )
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="创建 / 管理 inner-rag 本地账号")
    parser.add_argument("username", nargs="?", help="登录名")
    parser.add_argument("--display-name", default="", help="显示名（默认与登录名相同）")
    parser.add_argument("--password-stdin", action="store_true", help="从标准输入读一行作为密码")
    parser.add_argument("--list", action="store_true", help="列出全部账号")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--reset-password", action="store_true", help="重置已有账号的密码")
    actions.add_argument("--disable", action="store_true", help="停用账号")
    actions.add_argument("--enable", action="store_true", help="重新启用账号")
    args = parser.parse_args()

    if args.list:
        list_users()
        return
    if not args.username:
        parser.error("需要指定用户名，或用 --list 查看现有账号")

    try:
        init_db()
    except RuntimeError as exc:
        # 表结构没迁移：给出可照做的命令，而不是一串 traceback
        raise SystemExit(str(exc)) from exc

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == args.username).one_or_none()

        if args.disable or args.enable:
            if user is None:
                raise SystemExit(f"账号不存在: {args.username}")
            user.is_active = args.enable
            db.commit()
            logger.info(f"账号 {'已启用' if args.enable else '已停用'}: {user.username}")
            return

        if args.reset_password:
            if user is None:
                raise SystemExit(f"账号不存在: {args.username}")
            password = read_password(args.password_stdin)
            warn_if_weak(password)
            user.password_hash = hash_password(password)
            db.commit()
            logger.info(f"已重置密码: {user.username}")
            return

        if user is not None:
            raise SystemExit(f"账号已存在: {args.username}（重置密码请加 --reset-password）")

        password = read_password(args.password_stdin)
        warn_if_weak(password)
        user = User(
            username=args.username,
            display_name=args.display_name or args.username,
            password_hash=hash_password(password),
            is_active=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"已创建账号: {user.username} (id={user.id})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
