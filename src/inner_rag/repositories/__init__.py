"""关系库插件点：仓储契约（``base.py``）+ SQLAlchemy 实现（``sqlalchemy.py``）。

业务代码只 import 本包：换关系库（SQLite → PostgreSQL → 其它 ORM / 远程服务）时，
只需新增一个实现并在 ``build_repositories`` 里换掉，``api/`` 与 ``services/`` 不用动。

```python
repos = build_repositories(db)
kb = repos.kbs.get(kb_id)
```

一处刻意的边界：**用户表的写入**（建号 / 改口令）只在 `scripts/create_user.py`——那是需要
口令哈希与交互式输入的动作，不属于业务链路；读取走 `UserRepository`（见 ``base.py``）。
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from inner_rag.repositories.base import (
    ConversationRepository,
    DocumentRepository,
    KnowledgeBaseRepository,
    NewDocument,
    UserRepository,
)
from inner_rag.repositories.sqlalchemy import (
    SqlConversationRepository,
    SqlDocumentRepository,
    SqlKnowledgeBaseRepository,
    SqlUserRepository,
)

__all__ = [
    "ConversationRepository",
    "DocumentRepository",
    "KnowledgeBaseRepository",
    "NewDocument",
    "Repositories",
    "UserRepository",
    "build_repositories",
]


@dataclass(frozen=True)
class Repositories:
    """一次请求（或一个后台任务）用到的仓储，共享同一个会话。"""

    kbs: KnowledgeBaseRepository
    docs: DocumentRepository
    convs: ConversationRepository
    users: UserRepository


def build_repositories(db: Session) -> Repositories:
    """按当前实现构造仓储集合（换关系库时只改这里）。"""
    return Repositories(
        kbs=SqlKnowledgeBaseRepository(db),
        docs=SqlDocumentRepository(db),
        convs=SqlConversationRepository(db),
        users=SqlUserRepository(db),
    )
