from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from inspect import signature
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointTuple,
    DeltaChannelHistory,
    PendingWrite,
)
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langgraph._internal._config import (
    DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT,
    patch_configurable,
)
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    NS_END,
    NS_SEP,
    NULL_TASK_ID,
    PUSH,
    SNAPSHOT_BUMPS,
)
from langgraph._internal._typing import MISSING
from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import BinaryOperatorAggregate, _get_overwrite
from langgraph.channels.delta import DeltaChannel
from langgraph.managed.base import ManagedValueMapping, ManagedValueSpec

LATEST_VERSION = 4

GetNextVersion = Callable[[Any, None], Any]


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=LATEST_VERSION,
        id=str(uuid6(clock_seq=-2)),
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions={},
        versions_seen={},
    )


def put_writes_accepts_task_path(put_writes: Callable[..., Any]) -> bool:
    """存储器的 `put_writes` 或 `aput_writes` 是否接受 `task_path`。

    较早的实现没有此参数，因此仅在支持时传入。
    """
    return signature(put_writes).parameters.get("task_path") is not None


def exit_delta_task_id(step: int, task_id: str) -> str:
    """退出持久化模式下，增量通道写入使用的合成任务 ID。

    将超步编号编码到 UUID 第一组，使 `ORDER BY task_id, idx` 保持时间顺序，同时符合 RFC UUID 格式，满足 Postgres `checkpoint_writes.task_id uuid` 列要求。不能使用 `NULL_TASK_ID`，因为读取器会将该 ID 下的写入当作锚点检查点自身的待处理写入。
    """
    parts = str(uuid.UUID(task_id)).split("-")
    synthetic = f"{step:08d}-{parts[1]}-{parts[2]}-{parts[3]}-{parts[4]}"
    if synthetic == NULL_TASK_ID:
        return f"{step:08d}-0000-0000-0000-000000000001"
    return synthetic


def exit_delta_late_task_id(step: int, task_id: str) -> str:
    """退出模式下，锚点自身超步之后各超步写入的合成任务 ID。

    按超步顺序排在所有真实任务 ID 之后；无论存储器按任务路径还是 ID 排序，重放时这些写入都位于锚点自身超步之后。
    """
    parts = str(uuid.UUID(task_id)).split("-")
    return f"ffffffff-{step >> 16:04x}-{step & 0xFFFF:04x}-{parts[3]}-{parts[4]}"


def delta_channels_to_snapshot(
    channels: Mapping[str, BaseChannel],
    counters_since_delta_snapshot: Mapping[str, tuple[int, int]],
    channel_versions: ChannelVersions,
) -> set[str]:
    """返回当前应生成快照的增量通道名集合。

    累计更新次数达到 `snapshot_frequency`，或距上次快照的总超步数达到 `DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT` 时生成快照。没有版本的通道从未在此分支上被写入，无需快照。此函数只判断条件，不修改状态。
    """
    result: set[str] = set()
    for name, ch in channels.items():
        if (
            not isinstance(ch, DeltaChannel)
            or not ch.is_available()
            or name not in channel_versions
        ):
            continue
        updates, supersteps = counters_since_delta_snapshot.get(name, (0, 0))
        if (
            updates >= ch.snapshot_frequency
            or supersteps >= DELTA_MAX_SUPERSTEPS_SINCE_SNAPSHOT
        ):
            result.add(name)
    return result


def get_updated_channels_from_tasks(
    run_tasks: Iterable[Any],
) -> set[str]:
    """返回 update_state 超步写入的通道名，不包含 PUSH。"""
    return {c for task in run_tasks for c, _ in task.writes if c != PUSH}


def get_delta_channels_from_all_channels(
    channels: Mapping[str, BaseChannel],
    channel_versions: ChannelVersions,
) -> set[str]:
    """新线程首次 update_state 时需要快照的增量通道。

    仅包含本次写入的通道；其余通道没有版本，也没有值需要保存。
    """
    return {
        k
        for k, ch in channels.items()
        if isinstance(ch, DeltaChannel) and ch.is_available() and k in channel_versions
    }


def delta_channels_with_pending_writes(
    specs: Mapping[str, Any],
    pending_writes: Iterable[PendingWrite] | None,
) -> set[str]:
    """返回从此检查点启动的新分支必须快照的增量通道。

    检查点的待处理写入属于消费它们的子检查点，但未记录具体子检查点。新分支需快照这些写入涉及的全部增量通道，避免祖先回溯时再次重放。
    """
    return {
        ch
        for _, ch, _ in pending_writes or ()
        if isinstance(specs.get(ch), DeltaChannel)
    }


def delta_channels_overwritten(
    specs: Mapping[str, Any], writes: Iterable[tuple[str, Any]]
) -> set[str]:
    """返回 `writes` 中由 `Overwrite` 设置的增量通道名。

    `update_state` 会在新检查点保存这些通道的完整快照，与节点返回 `Overwrite` 时的循环行为一致。否则以后读取会从较旧快照开始，重放本应被 `Overwrite` 丢弃的写入。
    """
    return {
        ch
        for ch, value in writes
        if isinstance(specs.get(ch), DeltaChannel) and _get_overwrite(value)[0]
    }


def checkpoint_superseded(
    saver: BaseCheckpointSaver, config: RunnableConfig, saved: CheckpointTuple
) -> bool:
    """判断线程是否已越过 `saved`，即 `config` 指定的检查点。

    有子检查点的检查点必然不是最近保存项，因此不会漏判。已废弃分支的叶检查点也算作已被越过；区分这种情况需要列出线程并查找子节点，存储器难以低成本完成。
    """
    if not config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
        return False
    latest = saver.get_tuple(
        patch_configurable(config, {CONFIG_KEY_CHECKPOINT_ID: None})
    )
    return latest is not None and latest.checkpoint["id"] != saved.checkpoint["id"]


async def acheckpoint_superseded(
    saver: BaseCheckpointSaver, config: RunnableConfig, saved: CheckpointTuple
) -> bool:
    """`checkpoint_superseded` 的异步版本。"""
    if not config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
        return False
    latest = await saver.aget_tuple(
        patch_configurable(config, {CONFIG_KEY_CHECKPOINT_ID: None})
    )
    return latest is not None and latest.checkpoint["id"] != saved.checkpoint["id"]


def create_metadata_for_update_state_api(
    channels: Mapping[str, BaseChannel],
    updated_channels: set[str],
    *,
    prev_metadata: Mapping[str, Any] | None,
) -> dict[str, tuple[int, int]]:
    """为非新线程的 update_state 推进 `counters_since_delta_snapshot`。

    与 `_loop._put_checkpoint` 中每个超步推进计数的方式一致。
    """
    prev_counters = dict(
        (prev_metadata or {}).get("counters_since_delta_snapshot") or {}
    )
    new_counters: dict[str, tuple[int, int]] = {}
    for ch_name, ch in channels.items():
        if not isinstance(ch, DeltaChannel):
            continue
        u, s = prev_counters.get(ch_name, (0, 0))
        s += 1
        if ch_name in updated_channels:
            u += 1
        new_counters[ch_name] = (u, s)
    return new_counters


def create_checkpoint_plan_for_update_state_api(
    channels: Mapping[str, BaseChannel],
    updated_channels: set[str],
    *,
    source: Literal["update", "input"],
    step: int,
    parents: dict[str, Any],
    saved_metadata: Mapping[str, Any] | None,
    is_fresh_thread: bool,
    fork_channels: set[str],
    channel_versions: ChannelVersions,
) -> tuple[set[str], dict[str, Any]]:
    """返回 update_state 头检查点所需的 `(channels_to_snapshot, metadata)`。"""
    metadata: dict[str, Any] = {
        "source": source,
        "step": step,
        "parents": parents,
    }
    if is_fresh_thread:
        return get_delta_channels_from_all_channels(
            channels, channel_versions
        ), metadata

    new_counters = create_metadata_for_update_state_api(
        channels,
        updated_channels,
        prev_metadata=saved_metadata,
    )
    channels_to_snapshot = (
        delta_channels_to_snapshot(channels, new_counters, channel_versions)
        | fork_channels
    )
    for k in channels_to_snapshot:
        new_counters[k] = (0, 0)
    non_zero = {k: v for k, v in new_counters.items() if v != (0, 0)}
    if non_zero:
        metadata["counters_since_delta_snapshot"] = non_zero
    return channels_to_snapshot, metadata


def create_checkpoint(
    checkpoint: Checkpoint,
    channels: Mapping[str, BaseChannel] | None,
    step: int,
    *,
    id: str | None = None,
    updated_channels: set[str] | None = None,
    get_next_version: GetNextVersion | None = None,
    channels_to_snapshot: set[str] | None = None,
    stored_versions: ChannelVersions | None = None,
    trigger_to_nodes: Mapping[str, Sequence[str]] | None = None,
) -> Checkpoint:
    """根据前一检查点和当前通道状态构建新检查点。

    对 `channels_to_snapshot` 中每个通道，在 `channel_values[k]` 写入 `_DeltaSnapshot(value)`。其他增量通道不写入 `channel_values`，其状态由祖先回溯从 `checkpoint_writes` 重建。调用方通过 `delta_channels_to_snapshot(channels, counters)` 计算集合；未提供时默认为空，即不生成快照。

    `stored_versions` 为存储器上次保存的通道版本。某个通道需要快照但版本尚未改变时，须提升版本以确保快照写入。
    """
    ts = datetime.now(timezone.utc).isoformat()
    channels_to_snapshot = channels_to_snapshot or set()
    bumped: dict[str, tuple[Any, Any]] = {}
    if channels is None:
        values = checkpoint["channel_values"]
        channel_versions = checkpoint["channel_versions"]
    else:
        values = {}
        channel_versions = dict(checkpoint["channel_versions"])
        for k in channels:
            ch = channels[k]
            if k not in channel_versions:
                # 即使通道从未被写入，强制快照也必须落盘，
                # 才能终止祖先回溯；而 put 只会保存
                # 已经具有版本的通道快照。
                if k in channels_to_snapshot and get_next_version is not None:
                    channel_versions[k] = get_next_version(None, None)
                    bumped[k] = (None, channel_versions[k])
                    values[k] = _DeltaSnapshot(ch.get())
                continue
            if k in channels_to_snapshot:
                # put 仅为版本变化的通道保存快照，因此
                # 本超步未写入的通道若需要快照，也必须提升版本：
                # 例如退出模式在跳过该通道的超步达到快照阈值，
                # 或分叉分支的第一个检查点。
                if (
                    get_next_version is not None
                    and stored_versions is not None
                    and channel_versions[k] == stored_versions.get(k)
                ):
                    old = channel_versions[k]
                    channel_versions[k] = get_next_version(old, None)
                    bumped[k] = (old, channel_versions[k])
                values[k] = _DeltaSnapshot(ch.get())
            else:
                v = ch.checkpoint()
                if v is not MISSING:
                    values[k] = v
    return Checkpoint(
        v=LATEST_VERSION,
        ts=ts,
        id=id or str(uuid6(clock_seq=step)),
        channel_values=values,
        channel_versions=channel_versions,
        versions_seen=_mark_bumps_seen(
            checkpoint["versions_seen"], bumped, trigger_to_nodes or {}
        ),
        updated_channels=None if updated_channels is None else sorted(updated_channels),
    )


def _mark_bumps_seen(
    versions_seen: dict[str, ChannelVersions],
    bumped: Mapping[str, tuple[Any, Any]],
    trigger_to_nodes: Mapping[str, Sequence[str]],
) -> dict[str, ChannelVersions]:
    """将已观察到通道旧版本的节点同步推进至新版本。

    仅为保存快照而提升版本不属于真正写入；若不推进已读版本，会再次触发 `interrupt_before` 并重跑订阅者。从无版本提升的通道从未被写入，还需处理从未运行的订阅者，否则它们会因版本提升而启动。对于被推进的每个条目，`SNAPSHOT_BUMPS` 保存新版本和节点真正读取的版本，供 `versions_seen_without_bumps` 恢复。
    """
    if not bumped:
        return versions_seen
    out = dict(versions_seen)
    for k, (old, _) in bumped.items():
        if old is None:
            for node in trigger_to_nodes.get(k, ()):
                out.setdefault(node, {})
    marks = dict(versions_seen.get(SNAPSHOT_BUMPS, {}))
    for node, seen in out.items():
        if node == SNAPSHOT_BUMPS:
            continue
        for k, (old, new) in bumped.items():
            if seen.get(k) != old:
                continue
            advanced, read = _bump_keys(node, k)
            # 若之前的版本提升已经设置 old，则真实读取版本已有记录。
            if old is not None and marks.get(advanced) != old:
                marks[read] = old
            marks[advanced] = new
            out[node] = {**out[node], k: new}
    if marks:
        out[SNAPSHOT_BUMPS] = marks
    return out


def _bump_keys(node: str, channel: str) -> tuple[str, str]:
    # 节点名不能包含这两种分隔符，因此键不会冲突。
    return f"{node}{NS_SEP}{channel}", f"{node}{NS_END}{channel}"


def versions_seen_without_bumps(
    versions_seen: dict[str, ChannelVersions],
) -> dict[str, ChannelVersions]:
    """恢复节点实际读取时的 `versions_seen`。

    被快照版本提升推进的条目恢复为真实读取版本；若从未读取过，则移除该条目。
    """
    if not (marks := versions_seen.get(SNAPSHOT_BUMPS)):
        return versions_seen
    out: dict[str, ChannelVersions] = {}
    for node, seen in versions_seen.items():
        if node == SNAPSHOT_BUMPS:
            continue
        out[node] = {}
        for k, v in seen.items():
            advanced, read = _bump_keys(node, k)
            if marks.get(advanced) != v:
                out[node][k] = v
            elif read in marks:
                out[node][k] = marks[read]
    return out


def _delta_channels_to_replay(
    specs: Mapping[str, BaseChannel], checkpoint: Checkpoint
) -> list[str]:
    """返回需要通过祖先回溯重建检查点值的通道。

    `_DeltaSnapshot` 或迁移前的普通值可通过 `from_checkpoint` 直接恢复，只有此处没有存储值的通道需要回溯。没有版本意味着从未写入，其状态直接为空；若仍回溯，就无法找到快照终点，导致每次加载都遍历全部祖先。

    `BinaryOperatorAggregate` 会在每个版本保存值，因此其缺值意味着注解变更前曾是 `DeltaChannel`，也应按增量方式重建。
    """
    return [
        k
        for k, spec in specs.items()
        if isinstance(spec, (DeltaChannel, BinaryOperatorAggregate))
        and k in checkpoint["channel_versions"]
        and checkpoint["channel_values"].get(k, MISSING) is MISSING
    ]


def _replay(spec: BaseChannel, history: DeltaChannelHistory) -> BaseChannel:
    ch = spec.from_checkpoint(history.get("seed", MISSING))
    if isinstance(ch, DeltaChannel):
        ch.replay_writes(history["writes"])
    else:
        # 每条写入单独更新一次：这些写入跨越多个超步，若一次性更新，
        # 会拒绝第二个 Overwrite，并丢弃第一个 Overwrite 之后的写入。
        for _, _, value in history["writes"]:
            ch.update([value])
    return ch


def _require_saver_for_history(
    delta_channels: list[str],
    saver: BaseCheckpointSaver | None,
    config: RunnableConfig | None,
) -> None:
    if delta_channels and (saver is None or config is None):
        raise ValueError(
            f"DeltaChannel {delta_channels} has history to replay but no "
            "checkpointer or config was passed to read it"
        )


def channels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
    *,
    saver: BaseCheckpointSaver | None = None,
    config: RunnableConfig | None = None,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    """从检查点恢复通道。

    大多数通道通过 `spec.from_checkpoint(checkpoint["channel_values"][k])` 即可恢复。`DeltaChannel` 若不在 `channel_values` 中，则必须调用 `saver.get_delta_channel_history` 回溯祖先，找到最近初始值（`_DeltaSnapshot` 或迁移前的普通值），并收集它与目标检查点之间的写入。所有需要重放的增量通道合并为一次存储器调用。
    """
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        else:
            managed_specs[k] = v

    delta_channels = _delta_channels_to_replay(channel_specs, checkpoint)
    _require_saver_for_history(delta_channels, saver, config)
    histories: Mapping[str, DeltaChannelHistory] = {}
    if delta_channels and saver is not None and config is not None:
        histories = saver.get_delta_channel_history(
            config=config, channels=delta_channels
        )

    channels: dict[str, BaseChannel] = {}
    for k, spec in channel_specs.items():
        ch: BaseChannel
        if k in histories:
            ch = _replay(spec, histories[k])
        else:
            ch = spec.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
        channels[k] = ch
    return channels, managed_specs


async def achannels_from_checkpoint(
    specs: Mapping[str, BaseChannel | ManagedValueSpec],
    checkpoint: Checkpoint,
    *,
    saver: BaseCheckpointSaver | None = None,
    config: RunnableConfig | None = None,
) -> tuple[Mapping[str, BaseChannel], ManagedValueMapping]:
    """`channels_from_checkpoint` 的异步版本，说明见该方法。"""
    channel_specs: dict[str, BaseChannel] = {}
    managed_specs: dict[str, ManagedValueSpec] = {}
    for k, v in specs.items():
        if isinstance(v, BaseChannel):
            channel_specs[k] = v
        else:
            managed_specs[k] = v

    delta_channels = _delta_channels_to_replay(channel_specs, checkpoint)
    _require_saver_for_history(delta_channels, saver, config)
    histories: Mapping[str, DeltaChannelHistory] = {}
    if delta_channels and saver is not None and config is not None:
        histories = await saver.aget_delta_channel_history(
            config=config, channels=delta_channels
        )

    channels: dict[str, BaseChannel] = {}
    for k, spec in channel_specs.items():
        ch: BaseChannel
        if k in histories:
            ch = _replay(spec, histories[k])
        else:
            ch = spec.from_checkpoint(checkpoint["channel_values"].get(k, MISSING))
        channels[k] = ch
    return channels, managed_specs


def copy_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    return Checkpoint(
        v=checkpoint["v"],
        ts=checkpoint["ts"],
        id=checkpoint["id"],
        channel_values=checkpoint["channel_values"].copy(),
        channel_versions=checkpoint["channel_versions"].copy(),
        versions_seen={k: v.copy() for k, v in checkpoint["versions_seen"].items()},
        updated_channels=checkpoint.get("updated_channels", None),
    )
