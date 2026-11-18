from __future__ import annotations

import asyncio
import binascii
import concurrent.futures
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    ExitStack,
)
from datetime import datetime, timezone
from types import TracebackType
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)
from uuid import UUID, uuid5

from langchain_core.callbacks import AsyncParentRunManager, ParentRunManager
from langchain_core.runnables import RunnableConfig
from langgraph.cache.base import BaseCache
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
)
from langgraph.store.base import BaseStore
from typing_extensions import ParamSpec, Self

from langgraph._internal._config import patch_configurable
from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CHECKPOINT_ID,
    CONFIG_KEY_CHECKPOINT_MAP,
    CONFIG_KEY_CHECKPOINT_NS,
    CONFIG_KEY_REPLAY_STATE,
    CONFIG_KEY_RESUME_MAP,
    CONFIG_KEY_RESUMING,
    CONFIG_KEY_RUNTIME,
    CONFIG_KEY_SCRATCHPAD,
    CONFIG_KEY_STREAM,
    CONFIG_KEY_STRICT_RESUME,
    CONFIG_KEY_TASK_ID,
    CONFIG_KEY_THREAD_ID,
    ERROR,
    ERROR_SOURCE_NODE,
    INPUT,
    INTERRUPT,
    NS_END,
    NS_SEP,
    NULL_TASK_ID,
    PUSH,
    RESUME,
    TASKS,
)
from langgraph._internal._replay import ReplayState
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph._internal._typing import EMPTY_SEQ, MISSING
from langgraph.callbacks import (
    GraphInterruptEvent,
    GraphLifecycleEvent,
    GraphResumeEvent,
)
from langgraph.channels.base import BaseChannel
from langgraph.channels.binop import _get_overwrite
from langgraph.channels.delta import DeltaChannel
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.constants import TAG_HIDDEN
from langgraph.errors import (
    EmptyInputError,
    GraphInterrupt,
    UnknownInterruptError,
)
from langgraph.managed.base import (
    ManagedValueMapping,
    ManagedValueSpec,
)
from langgraph.pregel._algo import (
    Call,
    GetNextVersion,
    PregelTaskWrites,
    apply_writes,
    checkpoint_null_version,
    increment,
    prepare_next_tasks,
    prepare_node_error_handler_task,
    prepare_single_task,
    sanitize_untracked_values_in_send,
    should_interrupt,
    task_path_str,
)
from langgraph.pregel._checkpoint import (
    achannels_from_checkpoint,
    acheckpoint_superseded,
    channels_from_checkpoint,
    checkpoint_superseded,
    copy_checkpoint,
    create_checkpoint,
    delta_channels_to_snapshot,
    delta_channels_with_pending_writes,
    empty_checkpoint,
    exit_delta_late_task_id,
    exit_delta_task_id,
    put_writes_accepts_task_path,
)
from langgraph.pregel._executor import (
    AsyncBackgroundExecutor,
    BackgroundExecutor,
    Submit,
)
from langgraph.pregel._io import (
    map_command,
    map_input,
    map_output_updates,
    map_output_values,
    read_channels,
)
from langgraph.pregel._messages import ensure_message_ids
from langgraph.pregel._read import PregelNode
from langgraph.pregel._task_status import read_task_statuses
from langgraph.pregel._utils import get_new_channel_versions, is_xxh3_128_hexdigest
from langgraph.pregel.debug import (
    map_debug_checkpoint,
    map_debug_task_results,
    map_debug_tasks,
)
from langgraph.pregel.protocol import StreamChunk, StreamProtocol
from langgraph.runtime import RunControl, Runtime
from langgraph.types import (
    All,
    CachePolicy,
    Command,
    Durability,
    Interrupt,
    PregelExecutableTask,
    RetryPolicy,
    Send,
    StreamMode,
)

V = TypeVar("V")
P = ParamSpec("P")


WritesT = Sequence[tuple[str, Any]]


def DuplexStream(*streams: StreamProtocol) -> StreamProtocol:
    def __call__(value: StreamChunk) -> None:
        for stream in streams:
            if value[1] in stream.modes:
                stream(value)

    return StreamProtocol(__call__, {mode for s in streams for mode in s.modes})


def _cacheable(writes: WritesT) -> bool:
    # 命中缓存会跳过节点；若缓存错误，就会跳过节点且不再抛出异常，
    # 因此只缓存成功完成的节点。
    return not any(c in (INTERRUPT, ERROR) for c, _ in writes)


class PregelLoop:
    config: RunnableConfig
    store: BaseStore | None
    stream: StreamProtocol | None
    step: int
    stop: int

    input: Any | None
    cache: BaseCache[WritesT] | None
    checkpointer: BaseCheckpointSaver | None
    nodes: Mapping[str, PregelNode]
    specs: Mapping[str, BaseChannel | ManagedValueSpec]
    input_keys: str | Sequence[str]
    output_keys: str | Sequence[str]
    stream_keys: str | Sequence[str]
    is_replaying: bool
    is_nested: bool
    manager: None | AsyncParentRunManager | ParentRunManager
    interrupt_after: All | Sequence[str]
    interrupt_before: All | Sequence[str]
    durability: Durability
    retry_policy: Sequence[RetryPolicy]
    cache_policy: CachePolicy | None

    checkpointer_get_next_version: GetNextVersion
    checkpointer_put_writes: Callable[[RunnableConfig, WritesT, str], Any] | None
    checkpointer_put_writes_accepts_task_path: bool
    _checkpointer_put_after_previous: (
        Callable[
            [
                concurrent.futures.Future | None,
                Sequence[Any],
                RunnableConfig,
                Checkpoint,
                str,
                ChannelVersions,
            ],
            Any,
        ]
        | None
    )
    _migrate_checkpoint: Callable[[Checkpoint], None] | None
    submit: Submit
    channels: Mapping[str, BaseChannel]
    # 保存产生增量通道写入的 checkpointer.put_writes 调用所返回的 Future。
    # _put_checkpoint 将此列表交给提交的保存任务，
    # 保存任务先等待这些写入完成，确保检查点不会早于
    # 产生它的写入持久化。若写入或前一次保存失败，
    # 本次保存也必须失败：DeltaChannel 依赖父链上的写入重建，
    # 若跨过缺口保存检查点，后续读取会永久缺失数据。
    # 同步和异步 __enter__ 都会将其初始化为 []；
    # 只有未配置检查点存储器时才保持为 None。
    _delta_write_futs: list[Any] | None = None

    # 与 _delta_write_futs 相同，但用于错误处理器相关写入。
    # put_writes 持久化 ERROR_SOURCE_NODE 标记时，
    # 将其 Future 加入此列表；schedule_error_handler / aschedule_error_handler
    # 会等待列表中的任务，确保启动处理器前写入已持久化。
    _error_handler_write_futs: list[Any] | None = None

    # 退出模式的累积器：收集本次运行产生的全部增量通道写入，
    # 包括 _first 中的输入写入，以及
    # after_tick 捕获的各超步写入。退出时，_put_exit_delta_writes
    # 排除即将生成快照的通道，再将其余写入保存在锚点父检查点下。
    # 非退出模式时为 None，因此各捕获位置不执行操作。
    # 每项为 (step, task_id, task_path, channel, value)；
    # 排序方式见 _put_exit_delta_writes。
    _exit_delta_writes: list[tuple[int, str, str, str, Any]] | None = None

    # 记录已加载检查点中保存了增量写入的 (task_id, channel) 对，
    # 通过 checkpoint_id 恢复时可能重跑这些任务；同时记录
    # 本次运行的 Command 增量写入，与从检查点加载的 NULL_TASK_ID 写入分开；
    # 还记录检查点自身的超步编号，
    # 也就是本次运行执行的第一个超步。
    _stored_delta_writes: set[tuple[str, str]]
    _exit_command_writes: list[tuple[str, Any]]
    _exit_first_step: int | None = None

    # 无论快照周期计数如何，下一检查点都必须快照的增量通道：
    # 包括以下两种情况：
    # * 上次检查点之后发生 Overwrite，稀疏重放必须
    #   从覆盖后的值开始；
    # * 本次运行的起始检查点包含这些通道的待处理写入，参见
    #   delta_channels_with_pending_writes。
    _delta_channels_forced_snapshot: set[str]
    # 恢复时由 _first 设置，保存从检查点加载的写入；
    # after_tick 将其与重新分配回任务的写入进行比较。
    _resume_loaded_writes: Sequence[PendingWrite] = ()
    # _reapply_writes_to_succeeded_nodes 将已加载写入归还给的任务。
    _reapplied_task_ids: set[str]
    # 本次运行实际产生写入的任务；退出模式只在退出时保存这些任务。
    _written_task_ids: set[str]
    # 由 checkpoint_id 指定且线程已经越过的检查点，
    # 因此它可能已经有子检查点。
    _addressed_checkpoint_superseded: bool = False

    # 指向 __enter__ 加载的父检查点的 checkpoint_config，
    # 首次运行时则指向合成的空检查点。这里提前保存，
    # 因为每次 _put_checkpoint 都会将 self.checkpoint_config
    # 推进到新保存检查点的 ID，否则退出时就会丢失
    # 最初的父配置。_put_exit_delta_writes 在恢复运行时
    # 将其用作退出增量写入的锚点；首次运行时，
    # 用其派生按需创建的占位检查点配置，
    # 其中的 checkpoint_id 正是占位检查点需要使用的合成空 ID。
    _initial_checkpoint_config: RunnableConfig

    # 仅当 __enter__ 确实从存储器获取到元组时为 True；
    # 线程首次运行且尚无持久化父检查点时为 False。
    # _put_exit_delta_writes 据此决定使用
    # 现有父检查点作锚点（True），还是按需创建占位检查点（False）。
    _has_persisted_parent: bool = False
    # 仅当 __enter__ 加载线程最新检查点而非 checkpoint_id 指定项时为 True，
    # 此时尚未在该检查点上构建后继。
    _loaded_latest: bool = False

    managed: ManagedValueMapping
    checkpoint: Checkpoint
    checkpoint_id_saved: str
    checkpoint_ns: tuple[str, ...]
    checkpoint_config: RunnableConfig
    checkpoint_metadata: CheckpointMetadata
    checkpoint_pending_writes: list[PendingWrite]
    checkpoint_previous_versions: dict[str, str | float | int]
    prev_checkpoint_config: RunnableConfig | None

    status: Literal[
        "input",
        "pending",
        "done",
        "draining",
        "interrupt_before",
        "interrupt_after",
        "out_of_steps",
    ]
    control: RunControl | None
    tasks: dict[str, PregelExecutableTask]
    output: None | dict[str, Any] | Any = None
    updated_channels: set[str] | None = None
    _graph_lifecycle_events: deque[GraphLifecycleEvent]
    _has_graph_lifecycle_callbacks: bool

    # 公开接口。

    def __init__(
        self,
        input: Any | None,
        *,
        stream: StreamProtocol | None,
        config: RunnableConfig,
        store: BaseStore | None,
        cache: BaseCache | None,
        checkpointer: BaseCheckpointSaver | None,
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, BaseChannel | ManagedValueSpec],
        input_keys: str | Sequence[str],
        output_keys: str | Sequence[str],
        stream_keys: str | Sequence[str],
        trigger_to_nodes: Mapping[str, Sequence[str]],
        durability: Durability,
        interrupt_after: All | Sequence[str] = EMPTY_SEQ,
        interrupt_before: All | Sequence[str] = EMPTY_SEQ,
        manager: None | AsyncParentRunManager | ParentRunManager = None,
        migrate_checkpoint: Callable[[Checkpoint], None] | None = None,
        retry_policy: Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        has_graph_lifecycle_callbacks: bool = False,
    ) -> None:
        self.stream = stream
        self.config = config
        self.store = store
        self.step = 0
        self.stop = 0
        self.input = input
        self.checkpointer = checkpointer
        self.cache = cache
        self.nodes = nodes
        self.specs = specs
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.stream_keys = stream_keys
        self.interrupt_after = interrupt_after
        self.interrupt_before = interrupt_before
        self.manager = manager
        self.is_nested = CONFIG_KEY_TASK_ID in self.config.get(CONF, {})
        self._migrate_checkpoint = migrate_checkpoint
        self.trigger_to_nodes = trigger_to_nodes
        self.retry_policy = retry_policy
        self.cache_policy = cache_policy
        self.durability = durability
        # 通道类型在一次运行中固定，因此各超步读取这些集合，
        # 无需反复扫描所有通道。
        self._has_untracked_channels = any(
            isinstance(spec, UntrackedValue) for spec in specs.values()
        )
        self._delta_channel_names = [
            k for k, spec in specs.items() if isinstance(spec, DeltaChannel)
        ]
        self._has_graph_lifecycle_callbacks = has_graph_lifecycle_callbacks
        self._graph_lifecycle_events = deque()
        if self.stream is not None and CONFIG_KEY_STREAM in config[CONF]:
            self.stream = DuplexStream(self.stream, config[CONF][CONFIG_KEY_STREAM])
        scratchpad: PregelScratchpad | None = config[CONF].get(CONFIG_KEY_SCRATCHPAD)
        if isinstance(scratchpad, PregelScratchpad):
            # 计数大于 0 时，追加到 checkpoint_ns。
            # 计数为 0 时，保持原样。
            if cnt := scratchpad.subgraph_counter():
                self.config = patch_configurable(
                    self.config,
                    {
                        CONFIG_KEY_CHECKPOINT_NS: NS_SEP.join(
                            (
                                config[CONF][CONFIG_KEY_CHECKPOINT_NS],
                                str(cnt),
                            )
                        )
                    },
                )
        if not self.is_nested and config[CONF].get(CONFIG_KEY_CHECKPOINT_NS):
            self.config = patch_configurable(
                self.config,
                {CONFIG_KEY_CHECKPOINT_NS: "", CONFIG_KEY_CHECKPOINT_ID: None},
            )
        if (
            CONFIG_KEY_CHECKPOINT_MAP in self.config[CONF]
            and self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS)
            in self.config[CONF][CONFIG_KEY_CHECKPOINT_MAP]
        ):
            self.checkpoint_config = patch_configurable(
                self.config,
                {
                    CONFIG_KEY_CHECKPOINT_ID: self.config[CONF][
                        CONFIG_KEY_CHECKPOINT_MAP
                    ][self.config[CONF][CONFIG_KEY_CHECKPOINT_NS]]
                },
            )
        else:
            self.checkpoint_config = self.config
        if thread_id := self.checkpoint_config[CONF].get(CONFIG_KEY_THREAD_ID):
            if not isinstance(thread_id, str):
                self.checkpoint_config = patch_configurable(
                    self.checkpoint_config,
                    {CONFIG_KEY_THREAD_ID: str(thread_id)},
                )
        self.checkpoint_ns = (
            tuple(cast(str, self.config[CONF][CONFIG_KEY_CHECKPOINT_NS]).split(NS_SEP))
            if self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS)
            else ()
        )
        self.prev_checkpoint_config = None
        # 判断本次运行是否将循环指向特定检查点：包括非空
        # checkpoint_id、本图在 checkpoint_map 中的条目，或正在重放的
        # 父图。每个任务配置都含 checkpoint_id: None，因此不能仅检查键存在，
        # 否则会把所有子图都算进去。
        self._addressed = bool(
            self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID)
            or self.config[CONF].get(CONFIG_KEY_REPLAY_STATE)
        )
        runtime = self.config[CONF].get(CONFIG_KEY_RUNTIME)
        self.control = runtime.control if isinstance(runtime, Runtime) else None

    def _push_graph_lifecycle_event(
        self,
        kind: Literal["resume", "interrupt"],
        *,
        interrupts: tuple[Interrupt, ...] = (),
    ) -> None:
        # 排空状态不会进入生命周期事件：tick() 在推送前返回 False，
        # 而中断通过 GraphInterrupt 抛出。
        if self.status == "draining":
            raise RuntimeError("Draining status cannot emit lifecycle events")
        status = self.status
        if kind == "resume":
            self._graph_lifecycle_events.append(
                GraphResumeEvent(
                    run_id=None,
                    status=status,
                    checkpoint_id=self.checkpoint["id"],
                    checkpoint_ns=self.checkpoint_ns,
                )
            )
        elif kind == "interrupt":
            self._graph_lifecycle_events.append(
                GraphInterruptEvent(
                    run_id=None,
                    status=status,
                    checkpoint_id=self.checkpoint["id"],
                    checkpoint_ns=self.checkpoint_ns,
                    interrupts=interrupts,
                )
            )
        else:
            msg = f"Unknown graph lifecycle event type: {kind}"
            raise AssertionError(msg)

    def _pop_lifecycle_event(self) -> GraphLifecycleEvent | None:
        if not self._graph_lifecycle_events:
            return None
        return self._graph_lifecycle_events.popleft()

    def put_writes(
        self, task_id: str, writes: WritesT, *, cached: bool = False
    ) -> None:
        """记录任务写入，供下一次 tick 读取。"""
        if not writes:
            return
        self._written_task_ids.add(task_id)
        # 对特殊通道的写入去重，保留最后一次写入。
        if all(w[0] in WRITES_IDX_MAP for w in writes):
            writes = list({w[0]: w for w in writes}.values())
        # 在保存、放入 checkpoint_pending_writes 或应用写入之前处理，
        # 确保这些位置使用相同的消息 ID。
        ensure_message_ids(self.specs, writes)
        if task_id == NULL_TASK_ID:
            # 空任务的写入采用累积方式保存。
            self.checkpoint_pending_writes = [
                w
                for w in self.checkpoint_pending_writes
                if w[0] != task_id or w[1] not in WRITES_IDX_MAP
            ]
            # 存储器以列表位置作为普通写入的键，而上面去掉的
            # 特殊写入下次可能不再出现，因此将其放到末尾。
            # 否则先于更新保存的恢复写入在再次保存时会移动索引，
            # 从而被应用两次。
            writes = sorted(writes, key=lambda w: w[0] in WRITES_IDX_MAP)
            writes_to_save: WritesT = [
                w[1:] for w in self.checkpoint_pending_writes if w[0] == task_id
            ] + list(writes)
        else:
            # 移除此任务已有的写入。
            self.checkpoint_pending_writes = [
                w for w in self.checkpoint_pending_writes if w[0] != task_id
            ]
            writes_to_save = writes

        # 检查是否有写入指向 UntrackedValue 通道。
        if self._has_untracked_channels:
            # 检查点不持久化未跟踪的值。
            writes_to_save = [
                # 清理 Send 数据包中嵌套的 UntrackedValue。
                (
                    (c, sanitize_untracked_values_in_send(v, self.channels))
                    if c == TASKS and isinstance(v, Send)
                    else (c, v)
                )
                for c, v in writes_to_save
                # 不持久化 UntrackedValue 通道写入。
                if not isinstance(self.specs.get(c), UntrackedValue)
            ]

        # 保存写入。
        self.checkpoint_pending_writes.extend((task_id, c, v) for c, v in writes)
        if self.durability != "exit" and self.checkpointer_put_writes is not None:
            config = patch_configurable(
                self.checkpoint_config,
                {
                    CONFIG_KEY_CHECKPOINT_NS: self.config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_NS, ""
                    ),
                    CONFIG_KEY_CHECKPOINT_ID: self.checkpoint["id"],
                },
            )
            if self.checkpointer_put_writes_accepts_task_path:
                if hasattr(self, "tasks"):
                    task = self.tasks.get(task_id)
                else:
                    task = None
                fut = self.submit(
                    self.checkpointer_put_writes,
                    config,
                    writes_to_save,
                    task_id,
                    task_path_str(task.path) if task else "",
                )
            else:
                fut = self.submit(
                    self.checkpointer_put_writes,
                    config,
                    writes_to_save,
                    task_id,
                )
            if self._delta_write_futs is not None and any(
                isinstance(self.specs.get(c), DeltaChannel) for c, _ in writes_to_save
            ):
                self._delta_write_futs.append(fut)
            # 只有任务配置了错误处理器时，commit() 才会依据
            # _should_route_to_error_handler 添加 ERROR_SOURCE_NODE，
            # 因此此检查自然将 Future 收集限制在这些任务中。
            if self._error_handler_write_futs is not None and any(
                c == ERROR_SOURCE_NODE for c, _ in writes
            ):
                self._error_handler_write_futs.append(fut)
        # 输出写入。
        if hasattr(self, "tasks"):
            self.output_writes(task_id, writes, cached=cached)

    def _put_pending_writes(self) -> None:
        if self.checkpointer_put_writes is None:
            return
        if not self.checkpoint_pending_writes:
            return
        # 修补配置。
        config = patch_configurable(
            self.checkpoint_config,
            {
                CONFIG_KEY_CHECKPOINT_NS: self.config[CONF].get(
                    CONFIG_KEY_CHECKPOINT_NS, ""
                ),
                CONFIG_KEY_CHECKPOINT_ID: self.checkpoint["id"],
            },
        )
        # 随检查点加载的写入已经保存在检查点中；再次保存
        # 可能将其放到新索引下，导致重复应用。
        by_task = defaultdict(list)
        for task_id, channel, value in self.checkpoint_pending_writes:
            if task_id in self._written_task_ids:
                by_task[task_id].append((channel, value))
        # 将写入提交给检查点存储器。
        for task_id, writes in by_task.items():
            if self.checkpointer_put_writes_accepts_task_path and hasattr(
                self, "tasks"
            ):
                task = self.tasks.get(task_id)
                self.submit(
                    self.checkpointer_put_writes,
                    config,
                    writes,
                    task_id,
                    task_path_str(task.path) if task else "",
                )
            else:
                self.submit(
                    self.checkpointer_put_writes,
                    config,
                    writes,
                    task_id,
                )

    def accept_push(
        self, task: PregelExecutableTask, write_idx: int, call: Call | None = None
    ) -> PregelExecutableTask | None:
        """接收任务发出的 PUSH，必要时返回一个待启动的新任务。"""
        checkpoint_id_bytes = binascii.unhexlify(self.checkpoint["id"].replace("-", ""))
        null_version = checkpoint_null_version(self.checkpoint)
        if pushed := cast(
            PregelExecutableTask | None,
            prepare_single_task(
                (PUSH, task.path, write_idx, task.id, call),
                None,
                checkpoint=self.checkpoint,
                checkpoint_id_bytes=checkpoint_id_bytes,
                checkpoint_null_version=null_version,
                pending_writes=self.checkpoint_pending_writes,
                processes=self.nodes,
                channels=self.channels,
                managed=self.managed,
                config=task.config,
                step=self.step,
                stop=self.stop,
                for_execution=True,
                store=self.store,
                checkpointer=self.checkpointer,
                manager=self.manager,
                retry_policy=self.retry_policy,
                cache_policy=self.cache_policy,
            ),
        ):
            # 产生调试输出。
            self._emit("tasks", map_debug_tasks, [pushed])
            # 保存新任务。
            self.tasks[pushed.id] = pushed
            # 将待处理写入匹配到新任务。
            if self._reapplies_pending_writes:
                self._reapply_writes_to_succeeded_nodes({pushed.id: pushed})
            # 返回新任务；仅在此前未运行时启动。
            return pushed

    def schedule_error_handler(
        self, failed_task: PregelExecutableTask, error: BaseException
    ) -> PregelExecutableTask | None:
        raise NotImplementedError

    async def aschedule_error_handler(
        self, failed_task: PregelExecutableTask, error: BaseException
    ) -> PregelExecutableTask | None:
        raise NotImplementedError

    def tick(self) -> bool:
        """执行 Pregel 循环的一次迭代。

        Returns:
            若仍需继续迭代则返回 True。
        """

        # 检查是否达到迭代上限。
        if self.step > self.stop:
            self.status = "out_of_steps"
            # 流式输出上一步保存的检查点，与正常执行一次 tick 的行为一致。
            self._prepare_tasks_and_emit_checkpoint()
            return False

        self._prepare_tasks_and_emit_checkpoint()

        # 没有更多任务时，执行结束。
        if not self.tasks:
            self.status = "done"
            return False

        # 若前一轮留下待处理写入，则应用这些写入。
        if self._reapplies_pending_writes and self.checkpoint_pending_writes:
            self._reapply_writes_to_succeeded_nodes(self.tasks)
            self._resume_error_handlers_if_applicable()

        # 执行前检查是否需要中断。
        if self.interrupt_before and should_interrupt(
            self.checkpoint, self.interrupt_before, self.tasks.values()
        ):
            self.status = "interrupt_before"
            raise GraphInterrupt()

        # 若收到排空请求，则在运行下一批任务前停止。此检查位于
        # 断点检查之后，因为恢复操作会越过下一步断点，
        # 若提前排空，就会跳过这个断点。
        if self.control is not None and self.control.drain_requested:
            self.status = "draining"
            return False

        # 产生调试输出。
        self._emit("tasks", map_debug_tasks, self.tasks.values())

        # 输出已应用历史写入的任务结果。
        for task in self.tasks.values():
            if task.writes:
                self.output_writes(task.id, task.writes, cached=True)

        return True

    def after_tick(self) -> None:
        # 完成当前超步。
        writes = [w for t in self.tasks.values() for w in t.writes]
        self._delta_channels_forced_snapshot.update(
            ch
            for ch, v in writes
            if isinstance(self.specs.get(ch), DeltaChannel) and _get_overwrite(v)[0]
        )
        if self._resume_loaded_writes:
            # 加载后未被重新分配回任务的写入，属于本次运行
            # 丢弃或重跑的任务：例如被 Command(goto=...) 替换的 Send，
            # 或需要再次执行的错误处理器。
            self._delta_channels_forced_snapshot.update(
                delta_channels_with_pending_writes(
                    self.specs,
                    [
                        w
                        for w in self._resume_loaded_writes
                        if w[0] not in self._reapplied_task_ids
                    ],
                )
            )
            self._resume_loaded_writes = ()
        # 全部任务已完成。
        self.updated_channels = apply_writes(
            self.checkpoint,
            self.channels,
            self.tasks.values(),
            self.checkpointer_get_next_version,
            self.trigger_to_nodes,
        )
        # 输出 values 数据。
        if not self.updated_channels.isdisjoint(
            (self.output_keys,)
            if isinstance(self.output_keys, str)
            else self.output_keys
        ):
            self._emit(
                "values", map_output_values, self.output_keys, writes, self.channels
            )
        # 清空前，为退出模式累积器捕获增量通道写入。
        if self._exit_delta_writes is not None:
            # 第一个 tick 的待处理写入仍包含随检查点加载的条目，
            # 这些条目已经保存在该检查点中。
            first = self._exit_first_step is None
            if first:
                self._exit_first_step = self.step
                self._exit_delta_writes.extend(
                    (self.step, NULL_TASK_ID, "", ch, v)
                    for ch, v in self._exit_command_writes
                )
            for tid, ch, v in self.checkpoint_pending_writes:
                if not isinstance(self.specs.get(ch), DeltaChannel):
                    continue
                if first and (
                    tid == NULL_TASK_ID or (tid, ch) in self._stored_delta_writes
                ):
                    continue
                task = self.tasks.get(tid)
                path = task_path_str(task.path) if task else ""
                self._exit_delta_writes.append((self.step, tid, path, ch, v))
        # 清空待处理写入。
        self.checkpoint_pending_writes.clear()
        # 仅在第一个 tick 重放，也就是重新执行已完成任务。
        self.is_replaying = False
        # 保存检查点。
        self._put_checkpoint({"source": "loop"})
        # 执行后检查是否需要中断。
        if self.interrupt_after and should_interrupt(
            self.checkpoint, self.interrupt_after, self.tasks.values()
        ):
            self.status = "interrupt_after"
            # 负责流式输出此检查点的下一次 tick 将不会执行。
            self._prepare_tasks_and_emit_checkpoint()
            raise GraphInterrupt()
        # 清除恢复标志。
        self.config[CONF].pop(CONFIG_KEY_RESUMING, None)

    def _prepare_tasks_and_emit_checkpoint(self) -> None:
        self.tasks = prepare_next_tasks(
            self.checkpoint,
            self.checkpoint_pending_writes,
            self.nodes,
            self.channels,
            self.managed,
            self.config,
            self.step,
            self.stop,
            for_execution=True,
            manager=self.manager,
            store=self.store,
            checkpointer=self.checkpointer,
            trigger_to_nodes=self.trigger_to_nodes,
            updated_channels=self.updated_channels,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
        )

        # 产生调试输出。
        if self._checkpointer_put_after_previous is not None:
            self._emit(
                "checkpoints",
                map_debug_checkpoint,
                {
                    **self.checkpoint_config,
                    CONF: {
                        **self.checkpoint_config[CONF],
                        CONFIG_KEY_CHECKPOINT_ID: self.checkpoint["id"],
                    },
                },
                self.channels,
                self.stream_keys,
                self.checkpoint_metadata,
                self.tasks.values(),
                self.checkpoint_pending_writes,
                self.prev_checkpoint_config,
                self.output_keys,
            )

    def match_cached_writes(self) -> Sequence[PregelExecutableTask]:
        raise NotImplementedError

    async def amatch_cached_writes(self) -> Sequence[PregelExecutableTask]:
        raise NotImplementedError

    # 内部实现。

    @property
    def _reapplies_pending_writes(self) -> bool:
        """判断随检查点加载的写入是否应归还给原任务，而不是重新运行这些任务。"""
        return not self.is_replaying

    def _reapply_writes_to_succeeded_nodes(
        self, tasks: Mapping[str, PregelExecutableTask]
    ) -> None:
        """将检查点中已完成任务的输出恢复到内存任务。

        未完成、失败或被中断的任务保持空写入，以便运行器重新执行，或将其路由到错误处理器。
        """
        # 先筛选：每次任务调用都会执行此处，若读取所有产生过写入的任务状态，
        # 任务调用较多时就会出现二次复杂度。
        writes = (w for w in self.checkpoint_pending_writes if w[0] in tasks)
        for tid, status in read_task_statuses(writes).items():
            tasks[tid].writes.extend(status.output)
            self._reapplied_task_ids.add(tid)

    def _resume_error_handlers_if_applicable(self) -> None:
        """恢复运行时，为先前失败的任务调度错误处理器。

        在 `tick()` 中紧接 `_reapply_writes_to_succeeded_nodes` 调用。该方法已跳过 ERROR / ERROR_SOURCE_NODE 写入，所以先前失败任务的 `writes` 仍为空；运行器只执行 `not t.writes` 的任务，若不干预就会重跑原节点。对配置了错误处理器的节点，按以下步骤避免重跑：

        1. 扫描 `checkpoint_pending_writes`，寻找先前 `commit()` 持久化的 ERROR_SOURCE_NODE 标记，表示任务已失败并已路由到处理器。
        2. 将 `(ERROR, error)` 写入对应 `task.writes`，使其不再为空，运行器便会跳过。
        3. 准备新的错误处理任务并加入 `self.tasks`；该任务初始写入为空，因此会被运行器执行。
        """
        # 阶段一：收集同时具有 ERROR_SOURCE_NODE 和 ERROR 的任务 ID。
        failed: dict[str, BaseException] = {}
        for tid, chan, val in self.checkpoint_pending_writes:
            if chan == ERROR_SOURCE_NODE:
                error = next(
                    (
                        v
                        for t, c, v in self.checkpoint_pending_writes
                        if t == tid and c == ERROR
                    ),
                    None,
                )
                if error is not None:
                    failed[tid] = error
        # 阶段二：将原任务标为完成，并调度处理器任务。
        for task_id, error in failed.items():
            task = self.tasks.get(task_id)
            if task is None:
                continue
            handler_node = self.nodes[task.name].error_handler_node
            if not handler_node:
                continue
            # 写入非空，运行器的 not t.writes 筛选会跳过此任务。
            task.writes.append((ERROR, error))
            # 处理器任务初始写入为空，因此运行器会执行它。
            handler_task = prepare_node_error_handler_task(
                task,
                handler_node_name=handler_node,
                failed_error=error,
                checkpoint=self.checkpoint,
                pending_writes=self.checkpoint_pending_writes,
                processes=self.nodes,
                channels=self.channels,
                managed=self.managed,
                config=task.config,
                step=self.step,
                stop=self.stop,
                store=self.store,
                checkpointer=self.checkpointer,
                manager=self.manager,
                retry_policy=self.retry_policy,
                cache_policy=self.cache_policy,
            )
            if handler_task is not None:
                self.tasks[handler_task.id] = handler_task

    def _pending_interrupts(self) -> set[str]:
        """返回仍在等待回答的中断 ID。"""
        return {
            interrupt.id
            for status in read_task_statuses(self.checkpoint_pending_writes).values()
            for interrupt in status.pending_interrupts
        }

    def _first(
        self, *, input_keys: str | Sequence[str], updated_channels: set[str] | None
    ) -> set[str] | None:
        self._stored_delta_writes = {
            (tid, ch)
            for tid, ch, _ in self.checkpoint_pending_writes
            if tid != NULL_TASK_ID and isinstance(self.specs.get(ch), DeltaChannel)
        }
        self._exit_command_writes = []
        self._written_task_ids = set()
        # 从先前检查点恢复需要同时满足两项条件：
        # 1. 存在历史检查点，即 channel_versions 非空；
        # 2. 输入表示继续执行，而不是携带新输入启动新运行。
        # 子图由父图显式设置 CONFIG_KEY_RESUMING。
        # 外层图则从输入推断：
        #   - 输入为 None：从中断恢复，例如 invoke(None, config)；
        #   - 输入为 Command：任何 Command 都作用于已有状态；
        #   - run_id 相同：重新进入尚在进行的运行，例如流式连接重连。
        configurable = self.config.get(CONF, {})
        input_is_command = isinstance(self.input, Command)
        is_resuming = bool(self.checkpoint["channel_versions"]) and bool(
            configurable.get(
                CONFIG_KEY_RESUMING,
                self.input is None
                or input_is_command
                or (
                    not self.is_nested
                    and self.config.get("metadata", {}).get("run_id")
                    == self.checkpoint_metadata.get("run_id", MISSING)
                ),
            )
        )

        # 从特定检查点重放时，丢弃已缓存的 RESUME 写入，
        # 使 interrupt() 重新触发，而不是返回过期值。
        # 但主动恢复执行时必须保留这些写入，
        # 因为多中断场景需要保留已经解决的中断值。
        is_time_traveling = self._addressed and (
            # 时间旅行到子图检查点时，父图设置 RESUMING=True，
            # 因为它无法区分时间旅行和恢复；因此检查
            # checkpoint_map 中是否包含此子图自身的命名空间。
            # 通常映射只含祖先条目，见 _algo.py；
            # 子图自身条目仅由 get_state(subgraphs=True) 引入。
            (
                self.is_nested
                and configurable.get(CONFIG_KEY_CHECKPOINT_NS, "")
                in configurable.get(CONFIG_KEY_CHECKPOINT_MAP, {})
            )
            or not (
                # 外层图通过 Command(resume=...) 接收恢复请求。
                (input_is_command and cast(Command, self.input).resume is not None)
                # 子图通过父图传入的配置标志接收恢复请求，
                # 因为子图输入是 Send 的参数，而非 Command。
                or configurable.get(CONFIG_KEY_RESUMING, False)
            )
        )
        # 只有时间旅行或恢复到线程已经越过的检查点时，
        # 才重跑已完成任务。恢复到线程最新检查点时，
        # 与未指定 ID 时一样，将已有写入归还给原任务。
        self.is_replaying = is_time_traveling or self._addressed_checkpoint_superseded
        if is_time_traveling:
            self.checkpoint_pending_writes = [
                w for w in self.checkpoint_pending_writes if w[1] != RESUME
            ]
        # 子图重放所依据的父检查点；下方分叉操作
        # 原本可能改变这个引用。
        replay_bound: RunnableConfig | None = None
        # 判断本次运行是否在下方保存分叉检查点来启动新分支。
        # 分叉检查点自行携带 Command 写入，保证本次运行
        # 不将任何写入存到原来指定的检查点上，
        # 因为该检查点的子检查点可能会重放这些写入。
        forks = (
            is_resuming
            and is_time_traveling
            and (
                self.checkpoint_metadata.get("source") not in ("update", "fork")
                or self._addressed_checkpoint_superseded
            )
        )
        carried: list[PendingWrite] = []
        # 恢复并重新应用头检查点的待处理写入时，只有任务调度后
        # 才知道哪些写入应归还原任务，因此由 after_tick
        # 固化其余写入；它们与本次新增写入分开保存。
        reapplies = is_resuming and self._reapplies_pending_writes
        self._resume_loaded_writes = (
            [w for w in self.checkpoint_pending_writes if w[0] != NULL_TASK_ID]
            if reapplies
            else ()
        )
        self._reapplied_task_ids = set()
        self._delta_channels_forced_snapshot = (
            set()
            if reapplies
            else delta_channels_with_pending_writes(
                self.specs, self.checkpoint_pending_writes
            )
        )

        # 将 Command 映射为写入。
        if input_is_command:
            if (resume := cast(Command, self.input).resume) is not None:
                if not self.checkpointer:
                    raise RuntimeError(
                        "Cannot use Command(resume=...) without checkpointer"
                    )

                strict = (
                    self.config[CONF].get(CONFIG_KEY_STRICT_RESUME, False)
                    # 嵌套调用继承父级 configurable 配置。
                    and not self.is_nested
                    # 重放时，已完成任务可能再次运行并中断，
                    # 但先前输出仍可能使其中断看起来已被回答。
                    and not self._addressed_checkpoint_superseded
                )
                if resume_is_map := (
                    isinstance(resume, dict)
                    and all(is_xxh3_128_hexdigest(k) for k in resume)
                ):
                    if strict:
                        pending = self._pending_interrupts()
                        if not resume:
                            raise UnknownInterruptError(
                                "The resume map names no interrupt",
                                reason="empty_map",
                                pending_ids=sorted(pending),
                            )
                        if unknown := sorted(resume.keys() - pending):
                            raise UnknownInterruptError(
                                f"Unknown or already-consumed interrupt: {unknown[0]}",
                                reason="not_pending",
                                interrupt_ids=unknown,
                                pending_ids=sorted(pending),
                            )
                    self.config[CONF][CONFIG_KEY_RESUME_MAP] = resume
                else:
                    if len(pending := self._pending_interrupts()) > 1:
                        message = (
                            "When there are multiple pending interrupts, you must specify the interrupt id when resuming. "
                            "Docs: https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts."
                        )
                        if strict:
                            raise UnknownInterruptError(
                                message, reason="ambiguous", pending_ids=sorted(pending)
                            )
                        raise RuntimeError(message)

            writes: defaultdict[str, list[tuple[str, Any]]] = defaultdict(list)
            # 按任务 ID 分组写入。
            for tid, c, v in map_command(cmd=cast(Command, self.input)):
                if not (c == RESUME and resume_is_map):
                    writes[tid].append((c, v))
            if not writes and not resume_is_map:
                raise EmptyInputError("Received empty Command input")
            # 保存写入。
            for tid, ws in writes.items():
                if forks:
                    # 这些写入不经过分配消息 ID 的 put_writes。
                    ensure_message_ids(self.specs, ws)
                    carried.extend((tid, c, v) for c, v in ws)
                else:
                    self.put_writes(tid, ws)
                    if self._exit_delta_writes is not None and tid == NULL_TASK_ID:
                        self._exit_command_writes.extend(
                            (c, v)
                            for c, v in ws
                            if isinstance(self.specs.get(c), DeltaChannel)
                        )
            self._delta_channels_forced_snapshot.update(
                delta_channels_with_pending_writes(self.specs, carried)
            )
        # 应用 NULL_TASK_ID 写入。
        if null_writes := [
            w[1:]
            for w in (*self.checkpoint_pending_writes, *carried)
            if w[0] == NULL_TASK_ID
        ]:
            null_updated_channels = apply_writes(
                self.checkpoint,
                self.channels,
                [PregelTaskWrites((), INPUT, null_writes, [])],
                self.checkpointer_get_next_version,
                self.trigger_to_nodes,
            )
            if updated_channels is not None:
                updated_channels.update(null_updated_channels)
        # 推进到先前检查点之后。
        if is_resuming:
            self.checkpoint["versions_seen"].setdefault(INTERRUPT, {})
            for k in self.channels:
                if k in self.checkpoint["channel_versions"]:
                    version = self.checkpoint["channel_versions"][k]
                    self.checkpoint["versions_seen"][INTERRUPT][k] = version
            # 时间旅行，即从特定检查点重放时，
            # 先保存分叉检查点，使重放在新分支中执行。
            # 否则若在 after_tick() 之前触发中断，
            # 就不会创建新检查点；父图的最新检查点
            # 仍是原来的检查点，后续恢复便会
            # 加载错误的状态。
            # update_state 产生的分叉（source=update/fork）已有自己的
            # 分叉检查点，通常无需再建；但若线程已越过它，
            # 仍需新建，避免其他子检查点重放
            # 本次运行写入它的数据。
            if forks:
                if (
                    self.checkpoint_metadata.get("source") in ("update", "fork")
                    and self.prev_checkpoint_config
                ):
                    replay_bound = self.prev_checkpoint_config
                # 清除已加载检查点中的旧 INTERRUPT 写入。
                # 分叉会产生新的 checkpoint_id，进而改变
                # 任务 ID；旧中断写入若继续累积，
                # 会干扰未来恢复时的多中断检查。
                self.checkpoint_pending_writes = [
                    w for w in self.checkpoint_pending_writes if w[1] != INTERRUPT
                ]
                self._put_checkpoint({"source": "fork"})
            # 输出 values 数据。
            self._emit(
                "values", map_output_values, self.output_keys, True, self.channels
            )
        # 将输入映射为通道更新。
        elif input_writes := deque(map_input(input_keys, self.input)):
            # 在应用输入前处理，使本次运行读到的消息 ID
            # 与保存的输入消息 ID 相同。
            ensure_message_ids(self.specs, input_writes)
            # 丢弃先前检查点中的未完成任务。
            discard_tasks = prepare_next_tasks(
                self.checkpoint,
                self.checkpoint_pending_writes,
                self.nodes,
                self.channels,
                self.managed,
                self.config,
                self.step,
                self.stop,
                for_execution=True,
                store=None,
                checkpointer=None,
                manager=None,
                updated_channels=updated_channels,
            )
            self._delta_channels_forced_snapshot.update(
                c
                for c, v in input_writes
                if isinstance(self.specs.get(c), DeltaChannel) and _get_overwrite(v)[0]
            )
            # 应用输入写入。
            updated_channels = apply_writes(
                self.checkpoint,
                self.channels,
                [
                    *discard_tasks.values(),
                    PregelTaskWrites((), INPUT, input_writes, []),
                ],
                self.checkpointer_get_next_version,
                self.trigger_to_nodes,
            )
            # 输入写入直接经过上方的 apply_writes，
            # 不会进入 checkpoint_pending_writes，因此 after_tick
            # 无法捕获。退出模式需在此捕获，
            # 使 _exit_delta_writes 同时包含输入增量
            # 和各超步写入；否则读取时会丢失输入，
            # 因为未到快照频率的通道不会出现在 final_checkpoint.channel_values 中，
            # 而祖先遍历也不会读取目标检查点自身的 pending_writes。
            if self._exit_delta_writes is not None:
                for c, v in input_writes:
                    if isinstance(self.specs.get(c), DeltaChannel):
                        self._exit_delta_writes.append(
                            (self.step, NULL_TASK_ID, "", c, v)
                        )
            # DeltaChannel 从本次运行起始检查点中
            # 独立任务 ID 下的写入读取输入；
            # 读取器会将 NULL_TASK_ID 写入作为检查点自身的状态应用。
            # 新线程尚无可保存这些写入的检查点；由 checkpoint_id
            # 指定的检查点也可能已有子检查点，会读到这些写入，
            # 因此这些情况下改由输入检查点保存通道快照。
            if self.durability != "exit":
                delta_input = [
                    (c, v)
                    for c, v in input_writes
                    if isinstance(self.specs.get(c), DeltaChannel)
                ]
                if delta_input and self._has_persisted_parent and self._loaded_latest:
                    self.put_writes(
                        str(uuid5(UUID(self.checkpoint["id"]), INPUT)), delta_input
                    )
                else:
                    self._delta_channels_forced_snapshot.update(
                        c for c, _ in delta_input
                    )
            # 保存输入检查点。
            self.updated_channels = updated_channels
            self._put_checkpoint({"source": "input"})
        elif CONFIG_KEY_RESUMING not in configurable:
            raise EmptyInputError(f"Received no input for {input_keys}")
        # 将恢复和重放标志传给子图。
        if not self.is_nested:
            # 传入已解析的上界检查点 ID，使子图无需
            # 重新获取父图，也能找到对应检查点。
            # 对于 source=update/fork 的分叉，使用分叉的父检查点 ID，
            # 因为分叉创建时间晚于原始执行中
            # 子图检查点的创建时间。
            #
            # 这里只判断 is_time_traveling，而非 _addressed。
            # 客户端可能带着显式 checkpoint_id 恢复，
            # 但该 ID 恰好指向当前头检查点，例如 LangGraph Studio
            # 同时发送 checkpoint: {checkpoint_id} 和 Command(resume=...)。
            # 这种情况指定了检查点，却没有进行时间旅行；
            # 子图应正常加载其最新检查点，
            # 而不应走 ReplayState 的上界查找，
            # 否则会漏掉处理父图当前步骤期间
            # 创建的子图检查点。
            replay_state: ReplayState | None = None
            if is_time_traveling:
                replay_checkpoint_id = self.checkpoint["id"]
                if replay_bound is not None:
                    replay_checkpoint_id = replay_bound[CONF].get(
                        CONFIG_KEY_CHECKPOINT_ID, replay_checkpoint_id
                    )
                elif (
                    self.checkpoint_metadata.get("source")
                    in (
                        "update",
                        "fork",
                    )
                    and self.prev_checkpoint_config
                ):
                    replay_checkpoint_id = self.prev_checkpoint_config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_ID, replay_checkpoint_id
                    )
                replay_state = ReplayState(replay_checkpoint_id)
            self.config = patch_configurable(
                self.config,
                {
                    CONFIG_KEY_RESUMING: is_resuming,
                    CONFIG_KEY_REPLAY_STATE: replay_state,
                },
            )
        # 设置标志。
        self.status = "pending"
        if is_resuming:
            self._push_graph_lifecycle_event("resume")
        return updated_channels

    def _put_checkpoint(self, metadata: CheckpointMetadata) -> None:
        # 此处使用 is 判断对象身份，而不是 ==。四处调用中有三处传入
        # 新的字典（source 为 input、loop 或 fork）；只有
        # 退出时的 _suppress_interrupt（之后计划改名为 _on_loop_exit）
        # 复用已有的 self.checkpoint_metadata 实例。因此
        # metadata is self.checkpoint_metadata 仅在退出调用中为 True，
        # 可用于限定退出专属行为，例如跳过计数递增、
        # 保留原元数据。也可以改成显式
        # exiting: bool = False 参数；目前保持原样，
        # 以延续本文件已有用法。
        # TODO：替换为显式 exiting: bool = False 参数。
        exiting = metadata is self.checkpoint_metadata
        if exiting and self.checkpoint["id"] == self.checkpoint_id_saved:
            # 检查点已经保存。
            return
        # 维护每个增量通道的计数。
        #
        # 各增量通道记录一个 (updates, supersteps) 元组：
        # - updates 仅在本超步写入该通道时递增。
        # - supersteps 无论是否写入，每个超步都递增。
        #
        # 每个超步以新的 metadata 字典调用一次 _put_checkpoint，
        # source 为 input、loop 或 fork；这些中间调用
        # 负责递增计数。退出模式下，
        # _suppress_interrupt（之后计划改名为 _on_loop_exit）
        # 还会在退出时调用 _put_checkpoint(self.checkpoint_metadata)
        # 以提交最终检查点；此调用发生在
        # 最后一次中间调用已经计算过该超步之后。
        # 因此退出调用绝不能再次递增，
        # 否则最后一个超步会被重复计数。
        if not exiting:
            prev_counters = dict(
                self.checkpoint_metadata.get("counters_since_delta_snapshot") or {}
            )
            new_counters: dict[str, tuple[int, int]] = {}
            updated = self.updated_channels or set()
            for ch_name in self._delta_channel_names:
                u, s = prev_counters.get(ch_name, (0, 0))
                s += 1
                if ch_name in updated:
                    u += 1
                new_counters[ch_name] = (u, s)
            metadata["step"] = self.step
            metadata["parents"] = self.config[CONF].get(CONFIG_KEY_CHECKPOINT_MAP, {})
            self.checkpoint_metadata = metadata
        else:
            new_counters = dict(
                self.checkpoint_metadata.get("counters_since_delta_snapshot") or {}
            )
        # 判断是否需要保存检查点。
        do_checkpoint = self._checkpointer_put_after_previous is not None and (
            exiting or self.durability != "exit"
        )
        # 创建新检查点。
        channels_to_snapshot = (
            delta_channels_to_snapshot(
                {k: self.channels[k] for k in self._delta_channel_names},
                new_counters,
                self.checkpoint["channel_versions"],
            )
            | self._delta_channels_forced_snapshot
            if do_checkpoint
            else set()
        )
        self.checkpoint = create_checkpoint(
            self.checkpoint,
            self.channels if do_checkpoint else None,
            self.step,
            id=self.checkpoint["id"] if exiting else None,
            updated_channels=self.updated_channels,
            get_next_version=self.checkpointer_get_next_version
            if do_checkpoint
            else None,
            channels_to_snapshot=channels_to_snapshot,
            stored_versions=self.checkpoint_previous_versions,
            trigger_to_nodes=self.trigger_to_nodes,
        )
        for k in channels_to_snapshot:
            new_counters[k] = (0, 0)
        if do_checkpoint:
            self._delta_channels_forced_snapshot.difference_update(channels_to_snapshot)
        non_zero = {k: v for k, v in new_counters.items() if v != (0, 0)}
        if non_zero:
            self.checkpoint_metadata["counters_since_delta_snapshot"] = non_zero
        elif "counters_since_delta_snapshot" in self.checkpoint_metadata:
            del self.checkpoint_metadata["counters_since_delta_snapshot"]
        # 保存前清理检查点中的 TASK 通道，适用于 durability=="exit"。
        if TASKS in self.checkpoint["channel_values"] and self._has_untracked_channels:
            sanitized_tasks = [
                sanitize_untracked_values_in_send(value, self.channels)
                if isinstance(value, Send)
                else value
                for value in self.checkpoint["channel_values"][TASKS]
            ]
            self.checkpoint["channel_values"][TASKS] = sanitized_tasks
        # 未配置存储器时直接返回。

        if do_checkpoint and self._checkpointer_put_after_previous is not None:
            self.prev_checkpoint_config = (
                self.checkpoint_config
                if CONFIG_KEY_CHECKPOINT_ID in self.checkpoint_config[CONF]
                and self.checkpoint_config[CONF][CONFIG_KEY_CHECKPOINT_ID]
                else None
            )
            self.checkpoint_config = {
                **self.checkpoint_config,
                CONF: {
                    **self.checkpoint_config[CONF],
                    CONFIG_KEY_CHECKPOINT_NS: self.config[CONF].get(
                        CONFIG_KEY_CHECKPOINT_NS, ""
                    ),
                },
            }

            channel_versions = self.checkpoint["channel_versions"].copy()
            new_versions = get_new_channel_versions(
                self.checkpoint_previous_versions, channel_versions
            )
            self.checkpoint_previous_versions = channel_versions

            # 在此捕获当前检查点的写入：保存任务在后台运行，
            # 启动顺序可能不同；若由保存任务自行读取，
            # 可能误取另一个检查点的写入。
            delta_write_futs, self._delta_write_futs = self._delta_write_futs, []
            # 以非阻塞方式保存。
            # 若前一个检查点仍在保存，则先等待它完成，
            # 确保检查点存储器按顺序接收检查点。
            self._put_checkpoint_fut = self.submit(
                self._checkpointer_put_after_previous,
                getattr(self, "_put_checkpoint_fut", None),
                delta_write_futs,
                self.checkpoint_config,
                copy_checkpoint(self.checkpoint),
                self.checkpoint_metadata,
                new_versions,
            )
            self.checkpoint_config = {
                **self.checkpoint_config,
                CONF: {
                    **self.checkpoint_config[CONF],
                    CONFIG_KEY_CHECKPOINT_ID: self.checkpoint["id"],
                },
            }
        if not exiting:
            # 递增步骤编号。
            self.step += 1

    def _put_exit_delta_writes(self) -> None:
        """准备占位检查点和累积增量写入，使最终检查点的 put 等待它们完成。

        可见性要求：这两者必须先持久化，最终检查点才能对读取者可见。仅当没有持久化父检查点，且至少一个增量通道存在不会被快照覆盖的写入时，才按需创建占位检查点。
        """
        if (
            not self._exit_delta_writes
            or self.checkpointer is None
            or self._checkpointer_put_after_previous is None
            or self.checkpointer_put_writes is None
        ):
            return

        # checkpoint_id 指定的检查点可能已有子检查点，
        # 它们会读取存放于此的写入；因此这种情况下，
        # 本次运行写入的通道改为保存快照。
        if self._has_persisted_parent and not self._loaded_latest:
            self._delta_channels_forced_snapshot.update(
                w[3] for w in self._exit_delta_writes
            )

        counters = dict(
            self.checkpoint_metadata.get("counters_since_delta_snapshot") or {}
        )
        channels_to_snapshot = (
            delta_channels_to_snapshot(
                self.channels, counters, self.checkpoint["channel_versions"]
            )
            | self._delta_channels_forced_snapshot
        )

        pending = [
            w for w in self._exit_delta_writes if w[3] not in channels_to_snapshot
        ]
        if not pending:
            return

        if self._has_persisted_parent:
            # _initial_checkpoint_config 的 checkpoint_id 是已保存父检查点的 ID，
            # 因为存储器在 __enter__ 中返回了真实元组。
            anchor_config = self._initial_checkpoint_config
        else:
            stub_cp = empty_checkpoint()
            stub_cp["id"] = self.checkpoint_id_saved
            stub_cp["ts"] = datetime.now(timezone.utc).isoformat()
            # 占位检查点没有父节点，因此配置中的 checkpoint_id=None。
            stub_put_config = patch_configurable(
                self._initial_checkpoint_config,
                {CONFIG_KEY_CHECKPOINT_ID: None},
            )
            # 供 put_writes 使用的锚点配置，checkpoint_id 为占位检查点的 ID。
            anchor_config = patch_configurable(
                self._initial_checkpoint_config,
                {CONFIG_KEY_CHECKPOINT_ID: stub_cp["id"]},
            )
            self._put_checkpoint_fut = self.submit(
                self._checkpointer_put_after_previous,
                getattr(self, "_put_checkpoint_fut", None),
                (),
                stub_put_config,
                stub_cp,
                {"step": -2},
                {},
            )
            # 设置 checkpoint_config，使最终检查点的 _put_checkpoint
            # 将占位检查点视作父节点。
            self.checkpoint_config = anchor_config

        # 检查点自身超步保留真实任务路径，以便与恢复时
        # 从该检查点加载的写入交错排序；任务 ID 则仍使用
        # 合成 ID，否则若最终检查点保存失败，下一次恢复
        # 会将真实 ID 下的恢复任务误判为已完成。
        # 后续超步按步骤顺序排在所有真实任务路径和 ID 之后，
        # 保证存储器无论按路径还是 ID 排序都保持一致。
        grouped: dict[tuple[str, str], list[tuple[str, Any]]] = {}
        for step, tid, path, ch, v in pending:
            if tid == NULL_TASK_ID:
                key = (exit_delta_task_id(step, tid), "")
            elif step == self._exit_first_step:
                key = (exit_delta_task_id(step, tid), path)
            else:
                key = (exit_delta_late_task_id(step, tid), f"~~{step:010d}{path}")
            grouped.setdefault(key, []).append((ch, v))
        anchor_write_config = patch_configurable(
            anchor_config,
            {
                CONFIG_KEY_CHECKPOINT_NS: self.config[CONF].get(
                    CONFIG_KEY_CHECKPOINT_NS, ""
                ),
                CONFIG_KEY_CHECKPOINT_ID: anchor_config[CONF][CONFIG_KEY_CHECKPOINT_ID],
            },
        )
        for (tid, path), entries in grouped.items():
            if self.checkpointer_put_writes_accepts_task_path:
                fut = self.submit(
                    self.checkpointer_put_writes,
                    anchor_write_config,
                    entries,
                    tid,
                    path,
                )
            else:
                fut = self.submit(
                    self.checkpointer_put_writes,
                    anchor_write_config,
                    entries,
                    tid,
                )
            if self._delta_write_futs is not None:
                self._delta_write_futs.append(fut)

    def _suppress_interrupt(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        # 持久化当前检查点及写入。
        if self.durability == "exit" and (
            # 如果是顶层图，
            not self.is_nested
            # 或出现错误或中断的嵌套图，
            or exc_value is not None
            # 或设置了 checkpointer=True 的嵌套图。
            or all(NS_END not in part for part in self.checkpoint_ns)
        ):
            self._put_exit_delta_writes()
            self._put_checkpoint(self.checkpoint_metadata)
            self._put_pending_writes()
        # 抑制中断。
        if isinstance(exc_value, GraphInterrupt) and not self.is_nested:
            interrupt = exc_value
            interrupts = tuple(interrupt.args[0]) if interrupt.args else ()
            self._push_graph_lifecycle_event("interrupt", interrupts=interrupts)
            # 应用待处理写入后，再发出最后一个 values 事件。
            if (
                hasattr(self, "tasks")
                and self.checkpoint_pending_writes
                and any(task.writes for task in self.tasks.values())
            ):
                updated_channels = apply_writes(
                    self.checkpoint,
                    self.channels,
                    self.tasks.values(),
                    self.checkpointer_get_next_version,
                    self.trigger_to_nodes,
                )
                if not updated_channels.isdisjoint(
                    (self.output_keys,)
                    if isinstance(self.output_keys, str)
                    else self.output_keys
                ):
                    self._emit(
                        "values",
                        map_output_values,
                        self.output_keys,
                        [w for t in self.tasks.values() for w in t.writes],
                        self.channels,
                    )
            # 异常内容为空时发出 INTERRUPT，否则由 put_writes 发出。
            if not interrupt.args or not interrupt.args[0]:
                interrupt_payload = interrupt.args[0] if interrupt.args else ()
                self._emit(
                    "updates",
                    lambda: iter([{INTERRUPT: interrupt_payload}]),
                )
            # 保存最终输出。
            self.output = read_channels(self.channels, self.output_keys)
            # 抑制中断。
            return True
        elif exc_type is None:
            # 保存最终输出。
            self.output = read_channels(self.channels, self.output_keys)

    def _emit(
        self,
        mode: StreamMode,
        values: Callable[P, Iterator[Any]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        if self.stream is None:
            return
        debug_remap = mode in ("checkpoints", "tasks") and "debug" in self.stream.modes
        if mode not in self.stream.modes and not debug_remap:
            return
        for v in values(*args, **kwargs):
            if mode in self.stream.modes:
                self.stream((self.checkpoint_ns, mode, v))
            # debug 模式是带有包装字典的 checkpoints 或 tasks 模式。
            if debug_remap:
                self.stream(
                    (
                        self.checkpoint_ns,
                        "debug",
                        {
                            "step": self.step - 1
                            if mode == "checkpoints"
                            else self.step,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "type": "checkpoint"
                            if mode == "checkpoints"
                            else "task_result"
                            if "result" in v
                            else "task",
                            "payload": v,
                        },
                    )
                )

    def output_writes(
        self, task_id: str, writes: WritesT, *, cached: bool = False
    ) -> None:
        if task := self.tasks.get(task_id):
            if task.config is not None and TAG_HIDDEN in task.config.get(
                "tags", EMPTY_SEQ
            ):
                return
            if writes[0][0] == INTERRUPT:
                # loop.py 会在 PUSH 任务路径末尾追加布尔值，
                # 用于表示是否存在 call；若存在，
                # 此处不发出中断，由父级负责发出。
                if task.path[0] == PUSH and task.path[-1] is True:
                    return
                interrupts = [
                    {
                        INTERRUPT: tuple(
                            v
                            for w in writes
                            if w[0] == INTERRUPT
                            for v in (w[1] if isinstance(w[1], Sequence) else (w[1],))
                        )
                    }
                ]
                stream_modes = self.stream.modes if self.stream else []
                if "updates" in stream_modes:
                    self._emit("updates", lambda: iter(interrupts))
                if "values" in stream_modes:
                    current_values = read_channels(self.channels, self.output_keys)
                    # self.output_keys 为序列时，流片段包含完整状态和中断。
                    if isinstance(current_values, dict):
                        current_values[INTERRUPT] = interrupts[0][INTERRUPT]
                        self._emit("values", lambda: iter([current_values]))
                    # self.output_keys 为字符串时，流片段仅包含中断。
                    else:
                        self._emit("values", lambda: iter(interrupts))
            elif writes[0][0] != ERROR:
                self._emit(
                    "updates",
                    map_output_updates,
                    self.output_keys,
                    [(task, writes)],
                    cached,
                )
            if not cached:
                self._emit(
                    "tasks",
                    map_debug_task_results,
                    (task, writes),
                    self.stream_keys,
                )


class SyncPregelLoop(PregelLoop, AbstractContextManager):
    def __init__(
        self,
        input: Any | None,
        *,
        stream: StreamProtocol | None,
        config: RunnableConfig,
        store: BaseStore | None,
        cache: BaseCache | None,
        checkpointer: BaseCheckpointSaver | None,
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, BaseChannel | ManagedValueSpec],
        trigger_to_nodes: Mapping[str, Sequence[str]],
        durability: Durability,
        manager: None | AsyncParentRunManager | ParentRunManager = None,
        interrupt_after: All | Sequence[str] = EMPTY_SEQ,
        interrupt_before: All | Sequence[str] = EMPTY_SEQ,
        input_keys: str | Sequence[str] = EMPTY_SEQ,
        output_keys: str | Sequence[str] = EMPTY_SEQ,
        stream_keys: str | Sequence[str] = EMPTY_SEQ,
        migrate_checkpoint: Callable[[Checkpoint], None] | None = None,
        retry_policy: Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        has_graph_lifecycle_callbacks: bool = False,
    ) -> None:
        super().__init__(
            input,
            stream=stream,
            config=config,
            checkpointer=checkpointer,
            cache=cache,
            store=store,
            nodes=nodes,
            specs=specs,
            input_keys=input_keys,
            output_keys=output_keys,
            stream_keys=stream_keys,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
            manager=manager,
            migrate_checkpoint=migrate_checkpoint,
            trigger_to_nodes=trigger_to_nodes,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            durability=durability,
            has_graph_lifecycle_callbacks=has_graph_lifecycle_callbacks,
        )
        self.stack = ExitStack()
        if checkpointer:
            self.checkpointer_get_next_version = checkpointer.get_next_version
            self.checkpointer_put_writes = checkpointer.put_writes
            self.checkpointer_put_writes_accepts_task_path = (
                put_writes_accepts_task_path(checkpointer.put_writes)
            )
        else:
            self.checkpointer_get_next_version = increment
            self._checkpointer_put_after_previous = None  # type: ignore[assignment]
            self.checkpointer_put_writes = None
            self.checkpointer_put_writes_accepts_task_path = False

    def _checkpointer_put_after_previous(
        self,
        prev: concurrent.futures.Future | None,
        delta_write_futs: Sequence[concurrent.futures.Future],
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        for fut in delta_write_futs:
            fut.result()
        if prev is not None:
            prev.result()
        cast(BaseCheckpointSaver, self.checkpointer).put(
            config, checkpoint, metadata, new_versions
        )

    def match_cached_writes(self) -> Sequence[PregelExecutableTask]:
        if self.cache is None:
            return ()
        matched: list[PregelExecutableTask] = []
        if cached := {
            (t.cache_key.ns, t.cache_key.key): t
            for t in self.tasks.values()
            if t.cache_key and not t.writes
        }:
            for key, values in self.cache.get(tuple(cached)).items():
                task = cached[key]
                task.writes.extend(values)
                matched.append(task)
        return matched

    def accept_push(
        self, task: PregelExecutableTask, write_idx: int, call: Call | None = None
    ) -> PregelExecutableTask | None:
        if pushed := super().accept_push(task, write_idx, call):
            for task in self.match_cached_writes():
                self.put_writes(task.id, task.writes, cached=True)
        return pushed

    def schedule_error_handler(
        self, failed_task: PregelExecutableTask, error: BaseException
    ) -> PregelExecutableTask | None:
        handler_node = self.nodes[failed_task.name].error_handler_node
        if not handler_node:
            return None
        # 确保错误和 ERROR_SOURCE_NODE 写入在处理器运行前已经持久化。
        if self._error_handler_write_futs:
            futs, self._error_handler_write_futs = self._error_handler_write_futs, []
            concurrent.futures.wait(futs)
        handler_task = prepare_node_error_handler_task(
            failed_task,
            handler_node_name=handler_node,
            failed_error=error,
            checkpoint=self.checkpoint,
            pending_writes=self.checkpoint_pending_writes,
            processes=self.nodes,
            channels=self.channels,
            managed=self.managed,
            config=failed_task.config,
            step=self.step,
            stop=self.stop,
            store=self.store,
            checkpointer=self.checkpointer,
            manager=self.manager,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
        )
        if handler_task is None:
            return None
        self.tasks[handler_task.id] = handler_task
        if self._reapplies_pending_writes:
            self._reapply_writes_to_succeeded_nodes({handler_task.id: handler_task})
        for task in self.match_cached_writes():
            self.put_writes(task.id, task.writes, cached=True)
        return handler_task

    def put_writes(
        self, task_id: str, writes: WritesT, *, cached: bool = False
    ) -> None:
        """记录任务写入，供下一次 tick 读取。"""
        super().put_writes(task_id, writes, cached=cached)
        if cached or not writes or self.cache is None or not hasattr(self, "tasks"):
            return
        task = self.tasks.get(task_id)
        if task is None or task.cache_key is None or not _cacheable(writes):
            return
        self.submit(
            self.cache.set,
            {
                (task.cache_key.ns, task.cache_key.key): (
                    task.writes,
                    task.cache_key.ttl,
                )
            },
        )

    # 上下文管理器。

    def __enter__(self) -> Self:
        self._graph_lifecycle_events = deque()
        if not self.checkpointer:
            saved = None
        elif self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
            # 显式指定 checkpoint_id 时，获取该精确检查点。
            # 这同时涵盖普通重放，以及时间旅行期间
            # 通过 checkpoint_map 解析出的子图。
            saved = self.checkpointer.get_tuple(self.checkpoint_config)
            if saved is not None:
                self._addressed_checkpoint_superseded = checkpoint_superseded(
                    self.checkpointer, self.checkpoint_config, saved
                )
        elif replay_state := self.config[CONF].get(CONFIG_KEY_REPLAY_STATE):
            # 子图重放：父图正在重放，并通过 replay_state
            # 传入自身的 checkpoint_id。应从父图的
            # checkpoint_map 查找对应检查点，而非获取最新项。
            saved = replay_state.get_checkpoint(
                self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, ""),
                self.checkpointer,
                self.checkpoint_config,
            )
            # 清除 RESUMING，让 _first 重新应用输入而非执行恢复。
            # 这样会重建临时路由通道，使节点
            # 通过版本比较自然触发。
            self.config[CONF].pop(CONFIG_KEY_RESUMING, None)
        else:
            # 普通情况：获取此图和线程的最新检查点；
            # 首次调用时返回 None。
            saved = self.checkpointer.get_tuple(self.checkpoint_config)
            self._loaded_latest = True

        # 在下方合成空检查点覆盖 saved 之前记录此状态。
        # 首次运行时没有持久化父检查点，_put_exit_delta_writes
        # 据此按需创建占位检查点，而不使用父检查点作锚点。
        self._has_persisted_parent = saved is not None
        if saved is None:
            saved = CheckpointTuple(
                self.checkpoint_config, empty_checkpoint(), {"step": -2}, None, []
            )
        elif self._migrate_checkpoint is not None:
            self._migrate_checkpoint(saved.checkpoint)
        self.checkpoint_config = {
            **self.checkpoint_config,
            **saved.config,
            CONF: {
                CONFIG_KEY_CHECKPOINT_NS: "",
                **self.checkpoint_config.get(CONF, {}),
                **saved.config.get(CONF, {}),
            },
        }
        self._initial_checkpoint_config = self.checkpoint_config
        self.prev_checkpoint_config = saved.parent_config
        self.checkpoint_id_saved = saved.checkpoint["id"]
        self.checkpoint = saved.checkpoint
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = (
            [(str(tid), k, v) for tid, k, v in saved.pending_writes]
            if saved.pending_writes is not None
            else []
        )
        self._delta_write_futs = []
        self._error_handler_write_futs = []
        self._exit_delta_writes = (
            [] if self.durability == "exit" and self.checkpointer is not None else None
        )
        self.submit = self.stack.enter_context(BackgroundExecutor(self.config))
        self.channels, self.managed = channels_from_checkpoint(
            self.specs,
            self.checkpoint,
            saver=self.checkpointer,
            config=self.checkpoint_config,
        )
        self.stack.push(self._suppress_interrupt)
        self.status = "input"
        self.step = self.checkpoint_metadata["step"] + 1
        self.stop = self.step + self.config["recursion_limit"] + 1
        self.checkpoint_previous_versions = self.checkpoint["channel_versions"].copy()
        self.updated_channels = self._first(
            input_keys=self.input_keys,
            updated_channels=set(self.checkpoint.get("updated_channels"))  # type: ignore[arg-type]
            if self.checkpoint.get("updated_channels")
            else None,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        # 展开并退出上下文栈。
        return self.stack.__exit__(exc_type, exc_value, traceback)


class AsyncPregelLoop(PregelLoop, AbstractAsyncContextManager):
    def __init__(
        self,
        input: Any | None,
        *,
        stream: StreamProtocol | None,
        config: RunnableConfig,
        store: BaseStore | None,
        cache: BaseCache | None,
        checkpointer: BaseCheckpointSaver | None,
        nodes: Mapping[str, PregelNode],
        specs: Mapping[str, BaseChannel | ManagedValueSpec],
        trigger_to_nodes: Mapping[str, Sequence[str]],
        durability: Durability,
        interrupt_after: All | Sequence[str] = EMPTY_SEQ,
        interrupt_before: All | Sequence[str] = EMPTY_SEQ,
        manager: None | AsyncParentRunManager | ParentRunManager = None,
        input_keys: str | Sequence[str] = EMPTY_SEQ,
        output_keys: str | Sequence[str] = EMPTY_SEQ,
        stream_keys: str | Sequence[str] = EMPTY_SEQ,
        migrate_checkpoint: Callable[[Checkpoint], None] | None = None,
        retry_policy: Sequence[RetryPolicy] = (),
        cache_policy: CachePolicy | None = None,
        has_graph_lifecycle_callbacks: bool = False,
    ) -> None:
        super().__init__(
            input,
            stream=stream,
            config=config,
            checkpointer=checkpointer,
            cache=cache,
            store=store,
            nodes=nodes,
            specs=specs,
            input_keys=input_keys,
            output_keys=output_keys,
            stream_keys=stream_keys,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
            manager=manager,
            migrate_checkpoint=migrate_checkpoint,
            trigger_to_nodes=trigger_to_nodes,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            durability=durability,
            has_graph_lifecycle_callbacks=has_graph_lifecycle_callbacks,
        )
        self.stack = AsyncExitStack()
        if checkpointer:
            self.checkpointer_get_next_version = checkpointer.get_next_version
            self.checkpointer_put_writes = checkpointer.aput_writes
            self.checkpointer_put_writes_accepts_task_path = (
                put_writes_accepts_task_path(checkpointer.aput_writes)
            )
        else:
            self.checkpointer_get_next_version = increment
            self._checkpointer_put_after_previous = None  # type: ignore[assignment]
            self.checkpointer_put_writes = None
            self.checkpointer_put_writes_accepts_task_path = False

    async def _checkpointer_put_after_previous(
        self,
        prev: asyncio.Task | None,
        delta_write_futs: Sequence[asyncio.Future],
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        if delta_write_futs:
            await asyncio.gather(*delta_write_futs)
        if prev is not None:
            await prev
        await cast(BaseCheckpointSaver, self.checkpointer).aput(
            config, checkpoint, metadata, new_versions
        )

    async def amatch_cached_writes(self) -> Sequence[PregelExecutableTask]:
        if self.cache is None:
            return []
        matched: list[PregelExecutableTask] = []
        if cached := {
            (t.cache_key.ns, t.cache_key.key): t
            for t in self.tasks.values()
            if t.cache_key and not t.writes
        }:
            for key, values in (await self.cache.aget(tuple(cached))).items():
                task = cached[key]
                task.writes.extend(values)
                matched.append(task)
        return matched

    async def aaccept_push(
        self, task: PregelExecutableTask, write_idx: int, call: Call | None = None
    ) -> PregelExecutableTask | None:
        if pushed := super().accept_push(task, write_idx, call):
            for task in await self.amatch_cached_writes():
                self.put_writes(task.id, task.writes, cached=True)
        return pushed

    async def aschedule_error_handler(
        self, failed_task: PregelExecutableTask, error: BaseException
    ) -> PregelExecutableTask | None:
        handler_node = self.nodes[failed_task.name].error_handler_node
        if not handler_node:
            return None
        # 确保错误和 ERROR_SOURCE_NODE 写入在处理器运行前已经持久化。
        if self._error_handler_write_futs:
            futs, self._error_handler_write_futs = self._error_handler_write_futs, []
            await asyncio.gather(*futs)
        handler_task = prepare_node_error_handler_task(
            failed_task,
            handler_node_name=handler_node,
            failed_error=error,
            checkpoint=self.checkpoint,
            pending_writes=self.checkpoint_pending_writes,
            processes=self.nodes,
            channels=self.channels,
            managed=self.managed,
            config=failed_task.config,
            step=self.step,
            stop=self.stop,
            store=self.store,
            checkpointer=self.checkpointer,
            manager=self.manager,
            retry_policy=self.retry_policy,
            cache_policy=self.cache_policy,
        )
        if handler_task is None:
            return None
        self.tasks[handler_task.id] = handler_task
        if self._reapplies_pending_writes:
            self._reapply_writes_to_succeeded_nodes({handler_task.id: handler_task})
        for task in await self.amatch_cached_writes():
            self.put_writes(task.id, task.writes, cached=True)
        return handler_task

    def put_writes(
        self, task_id: str, writes: WritesT, *, cached: bool = False
    ) -> None:
        """记录任务写入，供下一次 tick 读取。"""
        super().put_writes(task_id, writes, cached=cached)
        if cached or not writes or self.cache is None or not hasattr(self, "tasks"):
            return
        task = self.tasks.get(task_id)
        if task is None or task.cache_key is None or not _cacheable(writes):
            return
        self.submit(
            self.cache.aset,
            {
                (task.cache_key.ns, task.cache_key.key): (
                    task.writes,
                    task.cache_key.ttl,
                )
            },
        )

    # 上下文管理器。

    async def __aenter__(self) -> Self:
        self._graph_lifecycle_events = deque()
        if not self.checkpointer:
            saved = None
        elif self.checkpoint_config[CONF].get(CONFIG_KEY_CHECKPOINT_ID):
            # 显式指定 checkpoint_id 时，获取该精确检查点。
            # 这同时涵盖普通重放，以及时间旅行期间
            # 通过 checkpoint_map 解析出的子图。
            saved = await self.checkpointer.aget_tuple(self.checkpoint_config)
            if saved is not None:
                self._addressed_checkpoint_superseded = await acheckpoint_superseded(
                    self.checkpointer, self.checkpoint_config, saved
                )
        elif replay_state := self.config[CONF].get(CONFIG_KEY_REPLAY_STATE):
            # 子图重放：父图正在重放，并通过 replay_state
            # 传入自身的 checkpoint_id。应从父图的
            # checkpoint_map 查找对应检查点，而非获取最新项。
            saved = await replay_state.aget_checkpoint(
                self.config[CONF].get(CONFIG_KEY_CHECKPOINT_NS, ""),
                self.checkpointer,
                self.checkpoint_config,
            )
            # 清除 RESUMING，让 _first 重新应用输入而非执行恢复。
            # 这样会重建临时路由通道，使节点
            # 通过版本比较自然触发。
            self.config[CONF].pop(CONFIG_KEY_RESUMING, None)
        else:
            # 普通情况：获取此图和线程的最新检查点；
            # 首次调用时返回 None。
            saved = await self.checkpointer.aget_tuple(self.checkpoint_config)
            self._loaded_latest = True

        # 在下方合成空检查点覆盖 saved 之前记录此状态。
        # 首次运行时没有持久化父检查点，_put_exit_delta_writes
        # 据此按需创建占位检查点，而不使用父检查点作锚点。
        self._has_persisted_parent = saved is not None
        if saved is None:
            saved = CheckpointTuple(
                self.checkpoint_config, empty_checkpoint(), {"step": -2}, None, []
            )
        elif self._migrate_checkpoint is not None:
            self._migrate_checkpoint(saved.checkpoint)
        self.checkpoint_config = {
            **self.checkpoint_config,
            **saved.config,
            CONF: {
                CONFIG_KEY_CHECKPOINT_NS: "",
                **self.checkpoint_config.get(CONF, {}),
                **saved.config.get(CONF, {}),
            },
        }
        self._initial_checkpoint_config = self.checkpoint_config
        self.prev_checkpoint_config = saved.parent_config
        self.checkpoint_id_saved = saved.checkpoint["id"]
        self.checkpoint = saved.checkpoint
        self.checkpoint_metadata = saved.metadata
        self.checkpoint_pending_writes = (
            [(str(tid), k, v) for tid, k, v in saved.pending_writes]
            if saved.pending_writes is not None
            else []
        )
        self._delta_write_futs = []
        self._error_handler_write_futs = []
        self._exit_delta_writes = (
            [] if self.durability == "exit" and self.checkpointer is not None else None
        )
        self.submit = await self.stack.enter_async_context(
            AsyncBackgroundExecutor(self.config)
        )
        self.channels, self.managed = await achannels_from_checkpoint(
            self.specs,
            self.checkpoint,
            saver=self.checkpointer,
            config=self.checkpoint_config,
        )
        self.stack.push(self._suppress_interrupt)
        self.status = "input"
        self.step = self.checkpoint_metadata["step"] + 1
        self.stop = self.step + self.config["recursion_limit"] + 1
        self.checkpoint_previous_versions = self.checkpoint["channel_versions"].copy()
        self.updated_channels = self._first(
            input_keys=self.input_keys,
            updated_channels=set(self.checkpoint.get("updated_channels"))  # type: ignore[arg-type]
            if self.checkpoint.get("updated_channels")
            else None,
        )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        # 展开并退出上下文栈。
        exit_task = asyncio.create_task(
            self.stack.__aexit__(exc_type, exc_value, traceback)
        )
        try:
            return await exit_task
        except asyncio.CancelledError as e:
            # 取消时向上传递退出任务，使 API 调用方能够
            # 在复用数据库连接等操作前，先等待退出任务完成。
            e.args = (*e.args, exit_task)
            raise
