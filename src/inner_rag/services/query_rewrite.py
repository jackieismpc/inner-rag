"""查询改写插件点：把「用户怎么问」和「语料怎么写」之间的措辞差抹平。

小库实测里有两类题是纯措辞问题，跟向量质量无关：

* **别名题**（「Sakura 是谁？」）：Sakura 与「路明非」在字面上毫无重合，词面检索帮不上，
  embedding 也拉不到一起——这一路只能靠**外部知识**把别名展开；
* **口语化提问**（「诺诺的真名是什么？」）：`是什么` / `的` 这类疑问框架会让 BM25 的
  词元集被稀释，真正有区分度的实词（`诺诺` `真名`）反而压不过噪声。

契约：

* ``rewrite`` 返回**要检索的查询列表，且必须把原查询放在第一位**——这样「改写没帮上忙」
  时最坏也只是多跑一路召回，不会比不改写更差；
* 返回的变体数由 ``QUERY_REWRITE_MAX_QUERIES`` 截断（含原查询），避免改写把检索成本放大；
* 改写**不改变阈值语义**：多路召回的结果按分块取最大分融合，阈值仍在融合之后统一生效；
* 失败退回 ``[query]``：改写是「有则更好」的一层。

默认 ``none``（不改写）。与 rerank 同理：改写会同时增加成本与召回，必须先在评测集上
证明「净收益为正」再打开（回归 >2pp 不允许合入）。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from inner_rag.core.config import settings
from inner_rag.plugins.registry import query_rewriters

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

# 中文/英文疑问框架与虚词。按长度降序匹配，避免「为什么」被「为」先吃掉。
STOPWORDS: tuple[str, ...] = (
    "什么时候",
    "为什么",
    "怎么样",
    "是什么",
    "告诉我",
    "帮我",
    "请问",
    "一下",
    "哪些",
    "哪个",
    "哪里",
    "怎么",
    "怎样",
    "如何",
    "多少",
    "什么",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "please",
    "the",
    "and",
    "are",
    "is",
    "of",
    "to",
    "in",
    "的",
    "了",
    "是",
    "吗",
    "呢",
    "吧",
    "啊",
    "呀",
    "哦",
    "么",
    "谁",
)

_STOPWORD_PATTERN = re.compile(
    "|".join(re.escape(word) for word in sorted(STOPWORDS, key=len, reverse=True)),
    re.IGNORECASE,
)


class QueryRewriter(Protocol):
    """改写契约：返回若干条查询，**第一条必须是原查询**。"""

    name: str

    async def rewrite(self, kb_id: int, query: str) -> list[str]: ...


class NoopQueryRewriter:
    """默认实现：不改写。"""

    name = "none"

    async def rewrite(self, kb_id: int, query: str) -> list[str]:
        return [query]


def strip_stopwords(query: str) -> str:
    """抹掉疑问框架与虚词，只留实词。抹完为空则返回空串（调用方据此丢弃该变体）。"""
    return _STOPWORD_PATTERN.sub(" ", query).strip(" \t\r\n？?")


def parse_aliases(raw: str) -> dict[str, str]:
    """解析 ``QUERY_ALIASES``：``别名=正式名`` 以逗号分隔；写错的行跳过并告警。"""
    aliases: dict[str, str] = {}
    for entry in raw.split(","):
        item = entry.strip()
        if not item:
            continue
        key, separator, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value:
            logger.warning(f"[QUERY] 忽略无法解析的别名条目: {item!r}（应为 别名=正式名）")
            continue
        aliases[key] = value
    return aliases


class AliasQueryRewriter:
    """别名扩展：把查询里出现的别名替换成正式名，产出一条平行查询。

    ``Sakura=路明非`` 时，「Sakura 是谁？」会补出一条「路明非 是谁？」——
    后者才可能命中语料里「路明非」的原文。字面毫无重合的别名题只有靠这一路能救。
    """

    name = "alias"

    async def rewrite(self, kb_id: int, query: str) -> list[str]:
        aliases = parse_aliases(settings.QUERY_ALIASES)
        if not aliases:
            return [query]

        expanded = query
        for alias, canonical in aliases.items():
            # 拉丁别名大小写不敏感（Sakura / SAKURA），中文别名天然精确匹配
            expanded = re.sub(re.escape(alias), canonical, expanded, flags=re.IGNORECASE)
        if expanded == query:
            return [query]
        return [query, expanded]


class KeywordsQueryRewriter:
    """关键词抽取：去掉疑问框架与虚词，补一条「实词查询」提升 BM25 的信噪比。"""

    name = "keywords"

    async def rewrite(self, kb_id: int, query: str) -> list[str]:
        keywords = strip_stopwords(query)
        if not keywords or keywords == query:
            return [query]
        return [query, keywords]


class LLMQueryRewriter:
    """让 chat 模型给出若干条同义改写（口语 → 书面、隐式指代 → 显式表述）。

    输出按行解析并去重；解析不出任何一条时退回 ``[query]``。改写条数受
    ``QUERY_REWRITE_MAX_QUERIES`` 约束——每次改写都要多跑一遍召回，成本是线性的。
    """

    name = "llm"

    def _get_llm(self) -> BaseChatModel:
        # 延迟 import：providers 与 plugins 之间本就有惰性注册的约定（见 plugins/registry.py）
        from inner_rag.providers import get_chat_model

        return get_chat_model()

    async def rewrite(self, kb_id: int, query: str) -> list[str]:
        wanted = max(1, settings.QUERY_REWRITE_MAX_QUERIES - 1)
        prompt = (
            "你是检索查询改写器。用户会问一个问题，请给出若干条同义改写，"
            "用于在文档库里做检索。要求：\n"
            "1. 保持原意，不要回答问题，不要编造原文里没有的专名；\n"
            "2. 把口语化表达改成书面表达；把隐式指代补全；\n"
            f"3. 每行一条，共 {wanted} 条，不要编号，不要输出任何解释。\n\n"
            f"问题：{query}"
        )
        try:
            message = await self._get_llm().ainvoke(prompt)
        except Exception as exc:
            logger.warning(f"[QUERY] llm 改写失败，退回原查询: {exc}")
            return [query]
        return [query, *_parse_lines(str(message.content), query)]


def _parse_lines(raw: str, original: str) -> list[str]:
    """按行抽取改写结果：去掉编号前缀、去重、剔除与原查询等价的条目。"""
    variants: list[str] = []
    for line in raw.splitlines():
        text = re.sub(r"^\s*(?:[-*]|\d+[.、)])\s*", "", line).strip()
        if not text or text == original or text in variants:
            continue
        variants.append(text)
    return variants


query_rewriters.register(NoopQueryRewriter.name, NoopQueryRewriter)
query_rewriters.register(AliasQueryRewriter.name, AliasQueryRewriter)
query_rewriters.register(KeywordsQueryRewriter.name, KeywordsQueryRewriter)
query_rewriters.register(LLMQueryRewriter.name, LLMQueryRewriter)


def build_query_rewriter(name: str | None = None) -> QueryRewriter:
    """按名字（默认取 ``QUERY_REWRITE_BACKEND``）构造改写器；未知名字抛可读错误。"""
    query_rewriters.load_entry_points()
    backend = (name or settings.QUERY_REWRITE_BACKEND).strip().lower()
    factory: Callable[[], QueryRewriter] | None = query_rewriters.get(backend)
    if factory is None:
        msg = (
            f"不支持的 QUERY_REWRITE_BACKEND: {backend!r}"
            f"（可选 {'/'.join(query_rewriters.names())}）"
        )
        raise ValueError(msg)
    return factory()


query_rewriter = build_query_rewriter()


async def rewrite(kb_id: int, query: str) -> list[str]:
    """对外入口：返回要检索的查询列表，首条恒为原查询，条数受配置截断。

    去重与保序都在这里做（而不是在各个实现里）：实现只负责「想出变体」，
    「原查询必须在、且必须在第一位」是契约，应该由入口统一兜住——
    否则每加一个实现就要重写一遍同样的三行，迟早会漏。
    """
    queries = await query_rewriter.rewrite(kb_id, query)
    limit = max(1, settings.QUERY_REWRITE_MAX_QUERIES)

    ordered = [query]
    for item in queries:
        text = item.strip()
        if text and text != query and text not in ordered:
            ordered.append(text)
    return ordered[:limit]
