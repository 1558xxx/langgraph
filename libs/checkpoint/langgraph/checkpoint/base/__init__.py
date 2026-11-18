from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator, Collection, Iterator, Mapping, Sequence
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypedDict,
    TypeVar,
    cast,
)

from langchain_core.runnables import RunnableConfig
from typing_extensions import NotRequired

from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.base import SerializerProtocol, maybe_add_typed_methods
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.serde.types import (
    ERROR,
    INTERRUPT,
    RESUME,
    SCHEDULED,
    ChannelProtocol,
)

V = TypeVar("V", int, float, str)
PendingWrite = tuple[str, str, Any]


logger = logging.getLogger(__name__)


# 设为 total=False，为后续扩展预留空间。
class CheckpointMetadata(TypedDict, total=False):
    """检查点关联的元数据。"""

    source: Literal["input", "loop", "update", "fork"]
    """检查点的来源。

    - `"input"`：由 invoke/stream/batch 的输入创建。
    - `"loop"`：在 Pregel 循环内部创建。
    - `"update"`：由手动更新状态创建。
    - `"fork"`：通过复制另一个检查点创建。
    """
    step: int
    """检查点对应的步骤编号。

    首个 `"input"` 检查点为 `-1`，首个 `"loop"` 检查点为 `0`，后续检查点依次递增。
    """
    parents: dict[str, str]
    """父检查点的 ID。

    从检查点命名空间到检查点 ID 的映射。
    """
    run_id: str
    """创建此检查点的运行 ID。"""
    counters_since_delta_snapshot: dict[str, tuple[int, int]]
    """各通道自上次写入 `_DeltaSnapshot` 以来的计数。

    !!! warning "Beta"

        此元数据字段用于仍处于测试阶段的 `DeltaChannel`；设计稳定前，字段名和内容可能调整。

    通道名映射到 `(updates, supersteps)`：

    - 索引 0（`updates`）：自上次快照以来，写入该通道的超步数。
    - 索引 1（`supersteps`）：自上次快照以来经过的总超步数，无论该通道是否被写入。

    当 `updates >= ch.snapshot_frequency` 或 `supersteps >= DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` 时生成快照。后者为系统级上限，默认 5000，可通过环境变量 `LANGGRAPH_DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` 设置。超步上限可避免已停止更新的增量通道无限回溯祖先检查点。

    不使用增量通道的线程不包含此字段。JSON 不支持原生元组，因此持久化为含两个元素的列表。
    """


ChannelVersions = dict[str, str | int | float]


class Checkpoint(TypedDict):
    """某一时刻的状态快照。"""

    v: int
    """检查点格式的版本，当前为 `1`。"""
    id: str
    """检查点 ID。

    该 ID 唯一且单调递增，可据此按创建顺序排列检查点。
    """
    ts: str
    """采用 ISO 8601 格式的检查点时间戳。"""
    channel_values: dict[str, Any]
    """检查点创建时各通道的值。

    从通道名到反序列化后的通道快照值的映射。
    """
    channel_versions: ChannelVersions
    """检查点创建时各通道的版本。

    键为通道名，值为该通道单调递增的版本字符串。
    """
    versions_seen: dict[str, ChannelVersions]
    """节点 ID 到“通道名与已读版本映射”的映射。

    记录各节点已读到的通道版本，用于决定下一步执行哪些节点。
    """
    updated_channels: list[str] | None
    """此检查点中发生更新的通道。"""


def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        pending_sends=checkpoint.get("pending_sends", []).copy(),
        updated_channels=checkpoint.get("updated_channels", None),
    )


class CheckpointTuple(NamedTuple):
    """包含检查点及其关联数据的元组。"""

    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None


class DeltaChannelHistory(TypedDict):
    """`BaseCheckpointSaver.get_delta_channel_history` 返回的单个通道历史条目。

    !!! warning "Beta"

        属于仍处于测试阶段的 `DeltaChannel` 支持接口，字段名和语义可能调整。

    从存储层描述目标检查点祖先链上某个通道的数据：

    - `writes`：沿目标分支从旧到新排列的增量，以 `PendingWrite` 元组表示。始终存在，可以为空，并已按通道过滤。目标检查点自身保存的写入属于下一超步，因此不包含在内。

        同一检查点内按 `writes_sort_key`，即 `(task_path, task_id, idx)` 排序，与实际执行应用超步写入的顺序一致。`task_id` 是路径的哈希；仅按它排序会改变并行任务的顺序，而归约函数可能对顺序敏感。没有 `task_path` 的写入（图输入、`update_state`、`Command` 更新以及添加此列前的数据）排在前面，并按 `task_id` 排序。
    - `seed`：最近一个在 `channel_values[ch]` 中保存了值的祖先检查点所提供的初始值。若回溯到根仍未找到，则省略该字段，调用方将从空状态开始。对具有有限快照频率的增量通道通常是 `_DeltaSnapshot`；对从旧通道类型迁移的线程则可能是普通值。
    """

    writes: list[PendingWrite]
    seed: NotRequired[Any]


def writes_sort_key(
    task_path: str, task_id: str = "", idx: int = 0
) -> tuple[str, str, int]:
    """单个超步写入的排序键。

    实际执行按此顺序应用任务写入，因此存储器重放历史写入时（例如 `get_delta_channel_history`）也必须使用此排序，否则顺序敏感的归约函数会重建出不同的结果。`task_path` 是传给 `put_writes` 的路径字符串。
    """
    return (task_path, task_id, idx)


class BaseCheckpointSaver(Generic[V]):
    """图检查点存储器的基类。

    检查点存储器使 LangGraph 智能体能够在一次交互内部以及多次交互之间持久化状态。配置存储器后，调用图时应在配置中传入 `thread_id`，示例见下方。

    `thread_id` 是存取检查点的主键。缺少它便无法保存状态、从中断恢复或进行时间旅行调试。独立的单次工作流应为每次运行使用唯一 ID（例如 uuid4）；需要对话记忆时，应在同一会话的多次调用间复用 `thread_id`，以累积聊天记录等状态。

    Attributes:
        serde (SerializerProtocol): 用于编解码检查点的序列化器。

    Note:
        自定义存储器时建议同时实现异步方法，避免阻塞主线程。

    Examples:
    ```python
    config = {"configurable": {"thread_id": "my-thread"}}
    graph.invoke(inputs, config)
    ```
    """

    serde: SerializerProtocol = JsonPlusSerializer()

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        self.serde = maybe_add_typed_methods(serde or self.serde)

    @property
    def config_specs(self) -> list:
        """定义检查点存储器的配置项。

        Returns:
            list: 配置字段规范列表。
        """
        return []

    def get(self, config: RunnableConfig) -> Checkpoint | None:
        """根据给定配置获取检查点。

        Args:
            config: 指定目标检查点的配置。

        Returns:
            目标检查点；未找到时返回 `None`。
        """
        if value := self.get_tuple(config):
            return value.checkpoint

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """根据给定配置获取检查点元组。

        Args:
            config: 指定目标检查点的配置。

        Returns:
            目标检查点元组；未找到时返回 `None`。其中 `pending_writes` 必须按 `writes_sort_key` 排序。

        Raises:
            NotImplementedError: 自定义存储器必须实现此方法。
        """
        raise NotImplementedError

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """列出符合条件的检查点。

        Args:
            config: 用于筛选检查点的基础配置。
            filter: 额外筛选条件。
            before: 仅列出此配置所指检查点之前的检查点。
            limit: 最多返回的检查点数量。

        Returns:
            匹配的检查点元组迭代器。

        Raises:
            NotImplementedError: 自定义存储器必须实现此方法。
        """
        raise NotImplementedError

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """保存检查点及其配置和元数据。

        Args:
            config: 检查点配置。
            checkpoint: 待保存的检查点。
            metadata: 检查点的附加元数据。
            new_versions: 本次写入产生的新通道版本。

        Returns:
            RunnableConfig: 保存检查点后更新的配置。

        Raises:
            NotImplementedError: 自定义存储器必须实现此方法。
        """
        raise NotImplementedError

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """保存关联到某个检查点的中间写入。

        Args:
            config: 关联检查点的配置。
            writes: 待保存的写入列表。
            task_id: 产生写入的任务 ID。
            task_path: 产生写入的任务路径。

        Raises:
            NotImplementedError: 自定义存储器必须实现此方法。
        """
        raise NotImplementedError

    def delete_thread(
        self,
        thread_id: str,
    ) -> None:
        """删除指定线程的全部检查点及写入。

        Args:
            thread_id: 待删除检查点所属的线程 ID。
        """
        raise NotImplementedError

    def delete_for_runs(
        self,
        run_ids: Sequence[str],
    ) -> None:
        """删除指定运行 ID 关联的全部检查点及写入。

        Args:
            run_ids: 待删除数据所属的运行 ID。

        !!! warning "DeltaChannel"

            若仍在使用的线程依赖某次运行产生的祖先 `checkpoint_writes` 或唯一的 `_DeltaSnapshot`，删除该运行会破坏相关 `DeltaChannel` 的状态重建。安全处理方式见 `prune` 中的说明。
        """
        raise NotImplementedError

    def copy_thread(
        self,
        source_thread_id: str,
        target_thread_id: str,
    ) -> None:
        """将一个线程的全部检查点及写入复制到另一个线程。

        Args:
            source_thread_id: 来源线程 ID。
            target_thread_id: 目标线程 ID。

        !!! warning "DeltaChannel"

            实现必须复制完整父链，包括祖先检查点及其 `checkpoint_writes`。只复制头检查点会使目标线程无法沿历史找到 `_DeltaSnapshot`，从而无法重建增量状态。等价地，每个增量通道都必须能够回溯到含有该通道 `_DeltaSnapshot` 的祖先，或拥有一直到根检查点的完整写入历史。
        """
        raise NotImplementedError

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """清理指定线程的检查点。

        Args:
            thread_ids: 待清理的线程 ID。
            strategy: `"keep_latest"` 仅保留每个命名空间最新的检查点；`"delete"` 删除全部检查点。

        !!! warning "DeltaChannel"

            自定义实现必须正确处理 `DeltaChannel`。非快照步骤仅在 `channel_values` 中保存哨兵值；重建时通过 `get_delta_channel_history` 沿父链累积 `checkpoint_writes`，直到找到包含 `_DeltaSnapshot` 的祖先。快照按 `snapshot_frequency` 更新频率写入。

            简单删除中间检查点及写入的 `"keep_latest"` 策略会截断这条链。保留的最新检查点通常不是快照点，此时历史查询返回的结果没有 `seed`，通道会被静默重建为空，不会报错。可采用以下安全策略：

            - 从每个保留检查点向前回溯，保留全部必要祖先及写入，直到每个增量通道都能找到 `_DeltaSnapshot`。
            - 删除祖先前先解析各通道当前值，将保留检查点的 `channel_values[k]` 改写为 `_DeltaSnapshot(value)`，再执行清理。
            - 在实现上述方案前，跳过使用 `DeltaChannel` 的线程。
        """
        raise NotImplementedError

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        """异步根据给定配置获取检查点。

        Args:
            config: 指定目标检查点的配置。

        Returns:
            目标检查点；未找到时返回 `None`。
        """
        if value := await self.aget_tuple(config):
            return value.checkpoint

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """异步根据给定配置获取检查点元组。

        Args:
            config: 指定目标检查点的配置。

        Returns:
            目标检查点元组；未找到时返回 `None`。其中 `pending_writes` 必须按 `writes_sort_key` 排序。

        Raises:
            NotImplementedError: 自定义存储器必须实现此方法。
        """
        raise NotImplementedError

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """异步列出符合条件的检查点。

        Args:
            config: 用于筛选检查点的基础配置。
            filter: 额外的元数据筛选条件。
            before: 仅列出此配置所指检查点之前的检查点。
            limit: 最多返回的检查点数量。

        Returns:
            匹配的检查点元组异步迭代器。

        Raises:
            NotImplementedError: 自定义存储器必须实现此方法。
        """
        raise NotImplementedError
        yield

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """异步保存检查点及其配置和元数据。

        Args:
            config: 检查点配置。
            checkpoint: 待保存的检查点。
            metadata: 检查点的附加元数据。
            new_versions: 本次写入产生的新通道版本。

        Returns:
            RunnableConfig: 保存检查点后更新的配置。

        Raises:
            NotImplementedError: 自定义存储器必须实现此方法。
        """
        raise NotImplementedError

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """异步保存关联到某个检查点的中间写入。

        Args:
            config: 关联检查点的配置。
            writes: 待保存的写入列表。
            task_id: 产生写入的任务 ID。
            task_path: 产生写入的任务路径。

        Raises:
            NotImplementedError: 自定义存储器必须实现此方法。
        """
        raise NotImplementedError

    async def adelete_thread(
        self,
        thread_id: str,
    ) -> None:
        """删除指定线程的全部检查点及写入。

        Args:
            thread_id: 待删除检查点所属的线程 ID。
        """
        raise NotImplementedError

    async def adelete_for_runs(
        self,
        run_ids: Sequence[str],
    ) -> None:
        """异步删除指定运行 ID 关联的全部检查点及写入。

        Args:
            run_ids: 待删除数据所属的运行 ID。

        !!! warning "DeltaChannel"

            参见 `delete_for_runs`。删除仍在使用的线程进行增量重建所依赖的数据，即头检查点与最近 `_DeltaSnapshot` 祖先之间的写入，会静默破坏通道状态。
        """
        raise NotImplementedError

    async def acopy_thread(
        self,
        source_thread_id: str,
        target_thread_id: str,
    ) -> None:
        """异步将一个线程的全部检查点及写入复制到另一个线程。

        Args:
            source_thread_id: 来源线程 ID。
            target_thread_id: 目标线程 ID。

        !!! warning "DeltaChannel"

            参见 `copy_thread`。复制必须覆盖完整父链，或至少使每个增量通道都能回溯到 `_DeltaSnapshot` 祖先，保证目标线程可重建状态。
        """
        raise NotImplementedError

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """异步清理指定线程的检查点。

        Args:
            thread_ids: 待清理的线程 ID。
            strategy: `"keep_latest"` 仅保留每个命名空间最新的检查点；`"delete"` 删除全部检查点。

        !!! warning "DeltaChannel"

            完整说明见 `prune`。`"keep_latest"` 必须保留目标检查点与最近 `_DeltaSnapshot` 祖先之间的检查点及写入，否则增量通道会被静默重建为空。
        """
        raise NotImplementedError

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """沿父检查点链回溯，返回各通道的写入及初始值。

        !!! warning "Beta"

            此方法属于测试阶段的 `DeltaChannel` 支持接口。方法签名、`DeltaChannelHistory` 返回结构及其与 `_DeltaSnapshot` 的交互可能调整。覆盖时应注意兼容性；默认实现会继续遵循公开的 `BaseCheckpointSaver` 契约。

        从 `config` 指定的检查点开始，沿 `parent_config` 回溯祖先，为各通道收集 `pending_writes`。每个通道在最近一个包含 `channel_values[ch]` 的祖先处停止，返回该值作为 `seed`。若到达根仍没有值，则不返回 `seed`，调用方从空状态开始。

        遍历的是父链，而非 `list(before=...)`；在线程分叉时，只收集当前路径上的祖先。默认实现使用 `get_tuple` 和 `parent_config` 为所有通道执行一次联合遍历，每个祖先仅访问一次。`InMemorySaver`、`PostgresSaver` 等可直接访问存储的实现会覆盖此方法以提高性能，但须遵循此处的返回契约。

        `PendingWrite` 不携带 `task_path`，因此默认实现按 `get_tuple` 返回的 `pending_writes` 顺序重放；该顺序必须符合 `writes_sort_key`。

        Args:
            config: 指定目标检查点的配置。
            channels: 要回溯的通道名；为空时返回空映射。

        Returns:
            每个请求通道对应的 `DeltaChannelHistory`。
        """
        if not channels:
            return {}
        collected_by_ch: dict[str, list[PendingWrite]] = {c: [] for c in channels}
        seed_by_ch: dict[str, Any] = {}
        remaining: set[str] = set(channels)
        target_tuple = self.get_tuple(config)
        cursor_config: RunnableConfig | None = (
            target_tuple.parent_config if target_tuple else None
        )
        while cursor_config is not None and remaining:
            tup = self.get_tuple(cursor_config)
            if tup is None:
                break
            if tup.pending_writes:
                for write in reversed(tup.pending_writes):
                    ch = write[1]
                    if ch in remaining:
                        collected_by_ch[ch].append(write)
            for ch in list(remaining):
                if ch in tup.checkpoint["channel_values"]:
                    seed_by_ch[ch] = tup.checkpoint["channel_values"][ch]
                    remaining.discard(ch)
            cursor_config = tup.parent_config
        result: dict[str, DeltaChannelHistory] = {}
        for ch in channels:
            entry: DeltaChannelHistory = {"writes": list(reversed(collected_by_ch[ch]))}
            if ch in seed_by_ch:
                entry["seed"] = seed_by_ch[ch]
            result[ch] = entry
        return result

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """`get_delta_channel_history` 的异步版本。

        !!! warning "Beta"

            属于测试阶段的 `DeltaChannel` 支持接口，注意事项见 `get_delta_channel_history`。
        """
        if not channels:
            return {}
        collected_by_ch: dict[str, list[PendingWrite]] = {c: [] for c in channels}
        seed_by_ch: dict[str, Any] = {}
        remaining: set[str] = set(channels)
        target_tuple = await self.aget_tuple(config)
        cursor_config: RunnableConfig | None = (
            target_tuple.parent_config if target_tuple else None
        )
        while cursor_config is not None and remaining:
            tup = await self.aget_tuple(cursor_config)
            if tup is None:
                break
            if tup.pending_writes:
                for write in reversed(tup.pending_writes):
                    ch = write[1]
                    if ch in remaining:
                        collected_by_ch[ch].append(write)
            for ch in list(remaining):
                if ch in tup.checkpoint["channel_values"]:
                    seed_by_ch[ch] = tup.checkpoint["channel_values"][ch]
                    remaining.discard(ch)
            cursor_config = tup.parent_config
        result: dict[str, DeltaChannelHistory] = {}
        for ch in channels:
            entry: DeltaChannelHistory = {"writes": list(reversed(collected_by_ch[ch]))}
            if ch in seed_by_ch:
                entry["seed"] = seed_by_ch[ch]
            result[ch] = entry
        return result

    def get_next_version(self, current: V | None, channel: None) -> V:
        """生成通道的下一个版本 ID。

        默认使用整数版本，每次递增 `1`。覆盖此方法时可使用 `str`、`int` 或 `float`，但版本必须单调递增。

        Args:
            current: 当前版本标识，可以是 `int`、`float` 或 `str`。
            channel: 已弃用，为向后兼容而保留的参数。

        Returns:
            V: 必须大于当前版本的下一版本标识。
        """
        if isinstance(current, str):
            raise NotImplementedError
        elif current is None:
            return 1
        else:
            return current + 1

    def with_allowlist(
        self, extra_allowlist: Collection[tuple[str, ...]]
    ) -> BaseCheckpointSaver[V]:
        """返回浅拷贝，并为其派生 msgpack 允许列表。"""
        serde = _with_msgpack_allowlist(self.serde, extra_allowlist)
        if serde is self.serde:
            return self
        clone = copy.copy(self)
        clone.serde = maybe_add_typed_methods(serde)
        return clone


def _with_msgpack_allowlist(
    serde: SerializerProtocol, extra_allowlist: Collection[tuple[str, ...]]
) -> SerializerProtocol:
    if isinstance(serde, JsonPlusSerializer):
        return serde.with_msgpack_allowlist(extra_allowlist)
    if isinstance(serde, EncryptedSerializer):
        inner = serde.serde
        if isinstance(inner, JsonPlusSerializer):
            updated_inner = inner.with_msgpack_allowlist(extra_allowlist)
            if updated_inner is inner:
                return serde
            return EncryptedSerializer(serde.cipher, updated_inner)
    logger.warning(
        "Serializer %s does not support msgpack allowlist. "
        "Strict msgpack deserialization will not be enforced.",
        type(serde).__name__,
    )
    return serde


class EmptyChannelError(Exception):
    """尝试读取尚未经历首次更新的通道值时抛出。"""

    pass


def get_checkpoint_id(config: RunnableConfig) -> str | None:
    """获取检查点 ID。"""
    return config["configurable"].get("checkpoint_id")


def get_checkpoint_metadata(
    config: RunnableConfig, metadata: CheckpointMetadata
) -> CheckpointMetadata:
    """以向后兼容的方式获取检查点元数据。"""
    metadata = {
        k: v.replace("\u0000", "") if isinstance(v, str) else v
        for k, v in metadata.items()
    }
    for obj in (config.get("metadata"), config.get("configurable")):
        if not obj:
            continue
        for key, v in obj.items():
            if key in metadata or key in EXCLUDED_METADATA_KEYS or key.startswith("__"):
                continue
            elif isinstance(v, str):
                metadata[key] = v.replace("\u0000", "")
            elif isinstance(v, (int, bool, float)):
                metadata[key] = v
    return metadata


def get_serializable_checkpoint_metadata(
    config: RunnableConfig, metadata: CheckpointMetadata
) -> CheckpointMetadata:
    """以向后兼容的方式获取检查点元数据。"""
    checkpoint_metadata = get_checkpoint_metadata(config, metadata)
    if "writes" in checkpoint_metadata:
        cast(dict[str, Any], checkpoint_metadata).pop("writes")
    return checkpoint_metadata


"""错误类型到错误索引的映射。
普通写入使用其在待保存列表中的索引；错误等特殊写入使用负索引，避免与普通写入冲突。各检查点存储器实现应在 `put_writes` 中使用此映射。
"""
WRITES_IDX_MAP = {ERROR: -1, SCHEDULED: -2, INTERRUPT: -3, RESUME: -4}

EXCLUDED_METADATA_KEYS = {
    "thread_id",
    "checkpoint_id",
    "checkpoint_ns",
    "checkpoint_map",
    "langgraph_step",
    "langgraph_node",
    "langgraph_triggers",
    "langgraph_path",
    "langgraph_checkpoint_ns",
}

# --- 以下为旧版 LangGraph 使用的已弃用工具 ---

LATEST_VERSION = 2


def empty_checkpoint() -> Checkpoint:
    from datetime import datetime, timezone

    return Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
        pending_sends=[],
        updated_channels=None,
    )


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, ChannelProtocol] | None,
    step: int,
    *,
    id: str | None = None,
) -> Checkpoint:
    """为给定通道创建检查点。"""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()
    if channels is None:
        values = checkpoint["channel_values"]
    else:
        values = {}
        for k, v in channels.items():
            if k not in checkpoint["channel_versions"]:
                continue
            try:
                values[k] = v.checkpoint()
            except EmptyChannelError:
                pass
    return Checkpoint(
        v=LATEST_VERSION,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=checkpoint["channel_versions"],
        versions_seen=checkpoint["versions_seen"],
        pending_sends=checkpoint.get("pending_sends", []),
        updated_channels=None,
    )
