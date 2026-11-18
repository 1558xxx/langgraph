from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeAlias

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.store.base import BaseStore

from langgraph._internal._typing import EMPTY_SEQ
from langgraph.runtime import Runtime
from langgraph.types import (
    CachePolicy,
    RetryPolicy,
    StreamWriter,
    TimeoutPolicy,
    TracePolicy,
)
from langgraph.typing import ContextT, NodeInputT, NodeInputT_contra


class _Node(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra) -> Any: ...


class _NodeWithConfig(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, config: RunnableConfig) -> Any: ...


class _NodeWithWriter(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, writer: StreamWriter) -> Any: ...


class _NodeWithStore(Protocol[NodeInputT_contra]):
    def __call__(self, state: NodeInputT_contra, *, store: BaseStore) -> Any: ...


class _NodeWithWriterStore(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, writer: StreamWriter, store: BaseStore
    ) -> Any: ...


class _NodeWithConfigWriter(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, config: RunnableConfig, writer: StreamWriter
    ) -> Any: ...


class _NodeWithConfigStore(Protocol[NodeInputT_contra]):
    def __call__(
        self, state: NodeInputT_contra, *, config: RunnableConfig, store: BaseStore
    ) -> Any: ...


class _NodeWithConfigWriterStore(Protocol[NodeInputT_contra]):
    def __call__(
        self,
        state: NodeInputT_contra,
        *,
        config: RunnableConfig,
        writer: StreamWriter,
        store: BaseStore,
    ) -> Any: ...


class _NodeWithRuntime(Protocol[NodeInputT_contra, ContextT]):
    def __call__(
        self, state: NodeInputT_contra, *, runtime: Runtime[ContextT]
    ) -> Any: ...


# TODO：引入 context 参数后，可能不再显式支持 config / store 签名；
# 也许可以改用 ParamSpec 为关键字参数提供类型支持。
# 这里仅涉及类型标注，因此未来几周仍可调整。
StateNode: TypeAlias = (
    _Node[NodeInputT]
    | _NodeWithConfig[NodeInputT]
    | _NodeWithWriter[NodeInputT]
    | _NodeWithStore[NodeInputT]
    | _NodeWithWriterStore[NodeInputT]
    | _NodeWithConfigWriter[NodeInputT]
    | _NodeWithConfigStore[NodeInputT]
    | _NodeWithConfigWriterStore[NodeInputT]
    | _NodeWithRuntime[NodeInputT, ContextT]
    | Runnable[NodeInputT, Any]
)


@dataclass(slots=True)
class StateNodeSpec(Generic[NodeInputT, ContextT]):
    runnable: StateNode[NodeInputT, ContextT]
    metadata: dict[str, Any] | None
    input_schema: type[NodeInputT]
    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None
    cache_policy: CachePolicy | None
    is_error_handler: bool = False
    error_handler_node: str | None = None
    ends: tuple[str, ...] | dict[str, str] | None = EMPTY_SEQ
    defer: bool = False
    timeout: TimeoutPolicy | None = None
    trace_policy: TracePolicy | None = None
    """可选策略，用于控制此节点在追踪运行中记录的内容。"""
