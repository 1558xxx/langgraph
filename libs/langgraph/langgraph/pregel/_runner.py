from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import threading
import time
import weakref
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from functools import partial
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
)

from langchain_core.callbacks import Callbacks

from langgraph._internal._constants import (
    CONF,
    CONFIG_KEY_CALL,
    CONFIG_KEY_SCRATCHPAD,
    ERROR,
    ERROR_SOURCE_NODE,
    INTERRUPT,
    NO_WRITES,
    RESUME,
    RETURN,
)
from langgraph._internal._future import chain_future, run_coroutine_threadsafe
from langgraph._internal._scratchpad import PregelScratchpad
from langgraph._internal._typing import MISSING
from langgraph.constants import TAG_HIDDEN
from langgraph.errors import GraphBubbleUp, GraphInterrupt, is_invalid_resume
from langgraph.pregel._algo import Call
from langgraph.pregel._executor import SKIP_RERAISE_SET, Submit
from langgraph.pregel._retry import arun_with_retry, run_with_retry
from langgraph.pregel._task_status import CONTROL_WRITES
from langgraph.types import (
    CachePolicy,
    PregelExecutableTask,
    RetryPolicy,
    TimeoutPolicy,
)

F = TypeVar("F", concurrent.futures.Future, asyncio.Future)
E = TypeVar("E", threading.Event, asyncio.Event)

# 从异常回溯中排除的文件名列表。
# 注意：仅当这些帧位于回溯末尾时才移除，并递归处理。
EXCLUDED_FRAME_FNAMES = (
    "langgraph/pregel/retry.py",
    "langgraph/pregel/runner.py",
    "langgraph/pregel/executor.py",
    "langgraph/utils/runnable.py",
    "langchain_core/runnables/config.py",
    "concurrent/futures/thread.py",
    "concurrent/futures/_base.py",
)


class FuturesDict(Generic[F, E], dict[F, PregelExecutableTask | None]):
    event: E
    callback: weakref.ref[Callable[[PregelExecutableTask, BaseException | None], None]]
    # 停止条件由 PregelRunner 注入，而非在此写死。
    # 这样运行器可将已由图级错误处理器接管的异常视为非致命错误，
    # 避免 on_done 因这些 Future 提前停止。
    should_stop: Callable[[set[F]], bool]
    counter: int
    done: set[F]
    failed: set[F]
    lock: threading.Lock

    def __init__(
        self,
        event: E,
        callback: weakref.ref[
            Callable[[PregelExecutableTask, BaseException | None], None]
        ],
        should_stop: Callable[[set[F]], bool],
        future_type: type[F],
        # 用于泛型类型标注；较新 Python 支持 FutureDict[...](...)。
    ) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.event = event
        self.callback = callback
        self.should_stop = should_stop
        self.counter = 0
        self.done: set[F] = set()
        self.failed: set[F] = set()

    def __setitem__(
        self,
        key: F,
        value: PregelExecutableTask | None,
    ) -> None:
        super().__setitem__(key, value)  # type: ignore[index]
        if value is not None:
            with self.lock:
                self.event.clear()
                self.counter += 1
            key.add_done_callback(partial(self.on_done, value))

    def on_done(
        self,
        task: PregelExecutableTask,
        fut: F,
    ) -> None:
        # 由 __setitem__ 注册的 future.add_done_callback 自动调用。
        try:
            if cb := self.callback():
                cb(task, _exception(fut))
        finally:
            with self.lock:
                self.done.add(fut)
                self.counter -= 1
                # 只有抛出异常的 Future 才可能满足停止条件，因此只检查它们，
                # 避免扫描所有已完成 Future，导致宽超步中出现二次复杂度。
                if not fut.cancelled() and fut.exception() is not None:
                    self.failed.add(fut)
                # 所有跟踪的 Future 完成，或满足运行器级停止条件时唤醒等待者，
                # 例如出现未处理的致命异常。
                if self.counter == 0 or self.should_stop(self.failed):
                    self.event.set()


class PregelRunner:
    """并发执行一组 Pregel 任务并提交其写入。

    有输出可发送时将控制权交还调用方，必要时中断其他任务。
    """

    def __init__(
        self,
        *,
        submit: weakref.ref[Submit],
        put_writes: weakref.ref[Callable[[str, Sequence[tuple[str, Any]]], None]],
        use_astream: bool = False,
        node_finished: Callable[[str], None] | None = None,
        node_error_handler_map: Mapping[str, str] | None = None,
        schedule_error_handler: Callable[
            [PregelExecutableTask, BaseException], PregelExecutableTask | None
        ]
        | None = None,
        aschedule_error_handler: Callable[
            [PregelExecutableTask, BaseException],
            Awaitable[PregelExecutableTask | None],
        ]
        | None = None,
    ) -> None:
        self.submit = submit
        self.put_writes = put_writes
        self.use_astream = use_astream
        self.node_finished = node_finished
        self.node_error_handler_map = dict(node_error_handler_map or {})
        self.error_handler_nodes = set(self.node_error_handler_map.values())
        self.schedule_error_handler = schedule_error_handler
        self.aschedule_error_handler = aschedule_error_handler
        # 已经路由到图级错误处理器的异常对象 ID。
        # 停止和异常传播检查会查询这些 ID，避免在同一次运行中
        # 再次通过普通致命错误路径抛出已处理的异常。
        self._handled_exception_ids: set[int] = set()

    def _should_route_to_error_handler(self, task: PregelExecutableTask) -> bool:
        if task.name in self.error_handler_nodes:
            return False
        return task.name in self.node_error_handler_map

    def tick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: float | None = None,
        retry_policy: Sequence[RetryPolicy] | None = None,
        get_waiter: Callable[[], concurrent.futures.Future[None]] | None = None,
        schedule_task: Callable[
            [PregelExecutableTask, int, Call | None],
            PregelExecutableTask | None,
        ],
    ) -> Iterator[None]:
        tasks = tuple(tasks)
        futures = FuturesDict(
            callback=weakref.WeakMethod(self.commit),
            event=threading.Event(),
            should_stop=partial(
                _should_stop_others, handled_exception_ids=self._handled_exception_ids
            ),
            future_type=concurrent.futures.Future,
        )
        # 将控制权交还调用方。
        yield
        # 仅有一个任务且没有超时或等待者时，使用快速路径。
        if len(tasks) == 0:
            return
        elif len(tasks) == 1 and timeout is None and get_waiter is None:
            t = tasks[0]
            scheduled_error_handler = False
            try:
                run_with_retry(
                    t,
                    retry_policy,
                    configurable={
                        CONFIG_KEY_CALL: partial(
                            _call,
                            weakref.ref(t),
                            retry_policy=retry_policy,
                            futures=weakref.ref(futures),
                            schedule_task=schedule_task,
                            submit=self.submit,
                        ),
                    },
                )
                self.commit(t, None)
            except Exception as exc:
                self.commit(t, exc)
                if (
                    not isinstance(exc, GraphBubbleUp)
                    and self._should_route_to_error_handler(t)
                    and self.schedule_error_handler is not None
                ):
                    self._handled_exception_ids.add(id(exc))
                    if handler_task := self.schedule_error_handler(t, exc):
                        tasks = (handler_task,)
                        scheduled_error_handler = True
                        # 转入常规调度路径，以执行错误处理器。
                if reraise and futures:
                    if id(exc) not in self._handled_exception_ids:
                        # 待所有 Future 完成后重新抛出。
                        fut: concurrent.futures.Future = concurrent.futures.Future()
                        fut.set_exception(exc)
                        futures.done.add(fut)
                        futures.failed.add(fut)
                elif reraise and id(exc) not in self._handled_exception_ids:
                    # 抛出异常前，先让调用方流式输出此任务结果。
                    yield
                    if tb := exc.__traceback__:
                        while tb.tb_next is not None and any(
                            tb.tb_frame.f_code.co_filename.endswith(name)
                            for name in EXCLUDED_FRAME_FNAMES
                        ):
                            tb = tb.tb_next
                        exc.__traceback__ = tb
                    raise
            if not futures and not scheduled_error_handler:
                # t 可能调度了另一个任务。
                return
            else:
                if not scheduled_error_handler:
                    tasks = ()  # 不重复调度此任务。
        # 按需添加等待任务。
        if get_waiter is not None:
            futures[get_waiter()] = None
        # 调度任务。
        for t in tasks:
            fut = self.submit()(  # type: ignore[misc]
                run_with_retry,
                t,
                retry_policy,
                configurable={
                    CONFIG_KEY_CALL: partial(
                        _call,
                        weakref.ref(t),
                        retry_policy=retry_policy,
                        futures=weakref.ref(futures),
                        schedule_task=schedule_task,
                        submit=self.submit,
                    ),
                },
                __reraise_on_exit__=reraise,
            )
            futures[fut] = t
        # 执行任务，等待任一失败或全部完成。
        # 每个任务都独立于其他并发任务。
        # 各任务完成时，逐一输出更新和调试信息。
        end_time = timeout + time.monotonic() if timeout else None
        handled_futures: set[concurrent.futures.Future[Any]] = set()
        while len(futures) > (1 if get_waiter is not None else 0):
            done, inflight = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
                timeout=(max(0, end_time - time.monotonic()) if end_time else None),
            )
            if not done:
                break  # 已超时。
            done_for_stop: set[concurrent.futures.Future[Any]] = set()
            for fut in done:
                task = futures.pop(fut)
                if task is None:
                    # 等待任务已完成，再调度一个等待任务。
                    if inflight and get_waiter is not None:
                        futures[get_waiter()] = None
                elif (
                    (task_exc := _exception(fut))
                    and self._should_route_to_error_handler(task)
                    and not isinstance(task_exc, GraphBubbleUp)
                ):
                    self._handled_exception_ids.add(id(task_exc))
                    SKIP_RERAISE_SET.add(fut)
                    handled_futures.add(fut)
                    if self.schedule_error_handler is not None:
                        if handler_task := self.schedule_error_handler(task, task_exc):
                            handler_fut = self.submit()(  # type: ignore[misc]
                                run_with_retry,
                                handler_task,
                                retry_policy,
                                configurable={
                                    CONFIG_KEY_CALL: partial(
                                        _call,
                                        weakref.ref(handler_task),
                                        retry_policy=retry_policy,
                                        futures=weakref.ref(futures),
                                        schedule_task=schedule_task,
                                        submit=self.submit,
                                    ),
                                },
                                __reraise_on_exit__=reraise,
                            )
                            futures[handler_fut] = handler_task
                else:
                    done_for_stop.add(fut)
            else:
                # 移除对循环变量的引用。
                del fut, task
            # 按需停止其他任务。
            if _should_stop_others(
                done_for_stop, handled_exception_ids=self._handled_exception_ids
            ):
                break
            # 将控制权交还调用方。
            yield
        # 等待完成回调执行完毕。
        futures.event.wait(
            timeout=(max(0, end_time - time.monotonic()) if end_time else None)
        )
        # 将控制权交还调用方。
        yield
        # 失败或超时时传播异常。
        try:
            _panic_or_proceed(
                futures.done.union(f for f, t in futures.items() if t is not None),
                panic=reraise,
                handled_exception_ids=self._handled_exception_ids,
                handled_futures=handled_futures,
            )
        except Exception as exc:
            if tb := exc.__traceback__:
                while tb.tb_next is not None and any(
                    tb.tb_frame.f_code.co_filename.endswith(name)
                    for name in EXCLUDED_FRAME_FNAMES
                ):
                    tb = tb.tb_next
                exc.__traceback__ = tb
            raise

    async def atick(
        self,
        tasks: Iterable[PregelExecutableTask],
        *,
        reraise: bool = True,
        timeout: float | None = None,
        retry_policy: Sequence[RetryPolicy] | None = None,
        get_waiter: Callable[[], asyncio.Future[None]] | None = None,
        schedule_task: Callable[
            [PregelExecutableTask, int, Call | None],
            Awaitable[PregelExecutableTask | None],
        ],
    ) -> AsyncIterator[None]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = tuple(tasks)
        futures = FuturesDict(
            callback=weakref.WeakMethod(self.commit),
            event=asyncio.Event(),
            should_stop=partial(
                _should_stop_others, handled_exception_ids=self._handled_exception_ids
            ),
            future_type=asyncio.Future,
        )
        # 将控制权交还调用方。
        yield
        # 仅有一个任务且没有等待者或超时时，使用快速路径。
        if len(tasks) == 0:
            return
        elif len(tasks) == 1 and get_waiter is None and timeout is None:
            t = tasks[0]
            scheduled_error_handler = False
            try:
                await arun_with_retry(
                    t,
                    retry_policy,
                    stream=self.use_astream,
                    configurable={
                        CONFIG_KEY_CALL: partial(
                            _acall,
                            weakref.ref(t),
                            stream=self.use_astream,
                            retry_policy=retry_policy,
                            futures=weakref.ref(futures),
                            schedule_task=schedule_task,
                            submit=self.submit,
                            loop=loop,
                        ),
                    },
                )
                self.commit(t, None)
            except Exception as exc:
                self.commit(t, exc)
                if (
                    not isinstance(exc, GraphBubbleUp)
                    and self._should_route_to_error_handler(t)
                    and self.aschedule_error_handler is not None
                ):
                    self._handled_exception_ids.add(id(exc))
                    if handler_task := await self.aschedule_error_handler(t, exc):
                        tasks = (handler_task,)
                        scheduled_error_handler = True
                if reraise and futures:
                    if id(exc) not in self._handled_exception_ids:
                        # 待所有 Future 完成后重新抛出。
                        fut: asyncio.Future = loop.create_future()
                        fut.set_exception(exc)
                        futures.done.add(fut)
                        futures.failed.add(fut)
                elif reraise and id(exc) not in self._handled_exception_ids:
                    # 抛出异常前，先让调用方流式输出此任务结果。
                    yield
                    if tb := exc.__traceback__:
                        while tb.tb_next is not None and any(
                            tb.tb_frame.f_code.co_filename.endswith(name)
                            for name in EXCLUDED_FRAME_FNAMES
                        ):
                            tb = tb.tb_next
                        exc.__traceback__ = tb
                    raise
            if not futures and not scheduled_error_handler:
                # t 可能调度了另一个任务。
                return
            else:
                if not scheduled_error_handler:
                    tasks = ()  # 不重复调度此任务。
        # 按需添加等待任务。
        if get_waiter is not None:
            futures[get_waiter()] = None
        # 调度任务。
        for t in tasks:
            fut = cast(
                asyncio.Future,
                self.submit()(  # type: ignore[misc]
                    arun_with_retry,
                    t,
                    retry_policy,
                    stream=self.use_astream,
                    configurable={
                        CONFIG_KEY_CALL: partial(
                            _acall,
                            weakref.ref(t),
                            retry_policy=retry_policy,
                            stream=self.use_astream,
                            futures=weakref.ref(futures),
                            schedule_task=schedule_task,
                            submit=self.submit,
                            loop=loop,
                        ),
                    },
                    __name__=t.name,
                    __cancel_on_exit__=True,
                    __reraise_on_exit__=reraise,
                ),
            )
            futures[fut] = t
        # 执行任务，等待任一失败或全部完成。
        # 每个任务都独立于其他并发任务。
        # 各任务完成时，逐一输出更新和调试信息。
        end_time = timeout + loop.time() if timeout else None
        handled_futures: set[asyncio.Future[Any]] = set()
        while len(futures) > (1 if get_waiter is not None else 0):
            done, inflight = await asyncio.wait(
                futures,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=(max(0, end_time - loop.time()) if end_time else None),
            )
            if not done:
                break  # 已超时。
            done_for_stop: set[asyncio.Future[Any]] = set()
            for fut in done:
                task = futures.pop(fut)
                if task is None:
                    # 等待任务已完成，再调度一个等待任务。
                    if inflight and get_waiter is not None:
                        futures[get_waiter()] = None
                elif (
                    (task_exc := _exception(fut))
                    and self._should_route_to_error_handler(task)
                    and not isinstance(task_exc, GraphBubbleUp)
                ):
                    self._handled_exception_ids.add(id(task_exc))
                    SKIP_RERAISE_SET.add(fut)
                    handled_futures.add(fut)
                    if self.aschedule_error_handler is not None:
                        if handler_task := await self.aschedule_error_handler(
                            task, task_exc
                        ):
                            handler_fut = cast(
                                asyncio.Future,
                                self.submit()(  # type: ignore[misc]
                                    arun_with_retry,
                                    handler_task,
                                    retry_policy,
                                    stream=self.use_astream,
                                    configurable={
                                        CONFIG_KEY_CALL: partial(
                                            _acall,
                                            weakref.ref(handler_task),
                                            retry_policy=retry_policy,
                                            stream=self.use_astream,
                                            futures=weakref.ref(futures),
                                            schedule_task=schedule_task,
                                            submit=self.submit,
                                            loop=loop,
                                        ),
                                    },
                                    __name__=handler_task.name,
                                    __cancel_on_exit__=True,
                                    __reraise_on_exit__=reraise,
                                ),
                            )
                            futures[handler_fut] = handler_task
                else:
                    done_for_stop.add(fut)
            else:
                # 移除对循环变量的引用。
                del fut, task
            # 按需停止其他任务。
            if _should_stop_others(
                done_for_stop, handled_exception_ids=self._handled_exception_ids
            ):
                break
            # 将控制权交还调用方。
            yield
        # 等待完成回调执行完毕。
        await asyncio.wait_for(
            futures.event.wait(),
            timeout=(max(0, end_time - loop.time()) if end_time else None),
        )
        # 将控制权交还调用方。
        yield
        # 取消等待任务。
        for fut in futures:
            fut.cancel()
        # 失败或超时时传播异常。
        try:
            _panic_or_proceed(
                futures.done.union(f for f, t in futures.items() if t is not None),
                timeout_exc_cls=asyncio.TimeoutError,
                panic=reraise,
                handled_exception_ids=self._handled_exception_ids,
                handled_futures=handled_futures,
            )
        except Exception as exc:
            if tb := exc.__traceback__:
                while tb.tb_next is not None and any(
                    tb.tb_frame.f_code.co_filename.endswith(name)
                    for name in EXCLUDED_FRAME_FNAMES
                ):
                    tb = tb.tb_next
                exc.__traceback__ = tb
            raise

    def commit(
        self,
        task: PregelExecutableTask,
        exception: BaseException | None,
    ) -> None:
        if isinstance(exception, asyncio.CancelledError):
            # 任务被取消时，也要将错误保存到任务中，
            # 使循环能够完成当前超步。
            task.writes.append((ERROR, exception))
            self.put_writes()(task.id, task.writes)  # type: ignore[misc]
        elif exception:
            if isinstance(exception, GraphInterrupt):
                # 将中断保存到检查点存储器。
                if exception.args[0]:
                    writes = [(INTERRUPT, exception.args[0])]
                    if resumes := [w for w in task.writes if w[0] == RESUME]:
                        writes.extend(resumes)
                    self.put_writes()(task.id, writes)  # type: ignore[misc]
            elif isinstance(exception, GraphBubbleUp):
                # 异常会在 _panic_or_proceed 中抛出。
                pass
            else:
                # 将错误保存到检查点存储器。
                task.writes.append((ERROR, exception))
                if self._should_route_to_error_handler(task) and not isinstance(
                    exception, GraphBubbleUp
                ):
                    task.writes.append((ERROR_SOURCE_NODE, task.name))
                    self._handled_exception_ids.add(id(exception))
                self.put_writes()(task.id, task.writes)  # type: ignore[misc]
        else:
            if self.node_finished and (
                task.config is None or TAG_HIDDEN not in task.config.get("tags", [])
            ):
                self.node_finished(task.name)
            if all(chan in CONTROL_WRITES for chan, _ in task.writes):
                # 即使没有输出，也记录任务已完成，
                # 参见 langgraph.pregel._task_status。
                task.writes.append((NO_WRITES, None))
            # 将任务写入保存到检查点存储器。
            self.put_writes()(task.id, task.writes)  # type: ignore[misc]


def _should_stop_others(
    done: set[F],
    *,
    handled_exception_ids: set[int] | None = None,
) -> bool:
    """检查是否有任务失败；若有则取消其余任务。

    `GraphInterrupt` 不视为失败。无效的恢复值也不视为失败：等待其他任务完成后再抛出，避免同次恢复中已被回答的任务在产生副作用后被截断，随后又使用已保存的回答重新运行。
    """
    for fut in done:
        if fut.cancelled():
            continue
        elif exc := fut.exception():
            if (
                id(exc) not in (handled_exception_ids or set())
                and not isinstance(exc, GraphBubbleUp)
                and not is_invalid_resume(exc)
                and fut not in SKIP_RERAISE_SET
            ):
                return True

    return False


def _exception(
    fut: concurrent.futures.Future[Any] | asyncio.Future[Any],
) -> BaseException | None:
    """获取 Future 的异常，且不抛出 CancelledError。"""
    if fut.cancelled():
        if isinstance(fut, asyncio.Future):
            return asyncio.CancelledError()
        else:
            return concurrent.futures.CancelledError()
    else:
        return fut.exception()


def _panic_or_proceed(
    futs: set[concurrent.futures.Future] | set[asyncio.Future],
    *,
    timeout_exc_cls: type[Exception] = TimeoutError,
    panic: bool = True,
    handled_exception_ids: set[int] | None = None,
    handled_futures: Collection[concurrent.futures.Future[Any] | asyncio.Future[Any]]
    | None = None,
) -> None:
    """有任务失败时取消其余任务；panic 为 True 时重新抛出异常。"""
    done: set[concurrent.futures.Future[Any] | asyncio.Future[Any]] = set()
    inflight: set[concurrent.futures.Future[Any] | asyncio.Future[Any]] = set()
    for fut in futs:
        if fut.cancelled():
            continue
        elif fut.done():
            done.add(fut)
        else:
            inflight.add(fut)
    interrupts: list[GraphInterrupt] = []
    while done:
        # 如果有任务失败。
        fut = done.pop()
        if exc := _exception(fut):
            if fut in (handled_futures or set()):
                continue
            if id(exc) in (handled_exception_ids or set()):
                continue
            # 取消全部待完成任务。
            while inflight:
                inflight.pop().cancel()
            # 抛出异常。
            if panic:
                if isinstance(exc, GraphInterrupt):
                    # 收集中断。
                    interrupts.append(exc)
                elif fut not in SKIP_RERAISE_SET:
                    raise exc
    # 抛出合并后的中断。
    if interrupts:
        raise GraphInterrupt(tuple(i for exc in interrupts for i in exc.args[0]))
    if inflight:
        # 执行到此处说明已超时。
        while inflight:
            # 取消全部待完成任务。
            inflight.pop().cancel()
        # 抛出超时错误。
        raise timeout_exc_cls("Timed out")


def _call(
    task: weakref.ref[PregelExecutableTask],
    func: Callable[[Any], Awaitable[Any] | Any],
    input: Any,
    *,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    timeout: TimeoutPolicy | None = None,
    callbacks: Callbacks = None,
    futures: weakref.ref[FuturesDict],
    schedule_task: Callable[
        [PregelExecutableTask, int, Call | None], PregelExecutableTask | None
    ],
    submit: weakref.ref[Submit],
) -> concurrent.futures.Future[Any]:
    if inspect.iscoroutinefunction(func):
        raise RuntimeError("In an sync context async tasks cannot be called")

    fut: concurrent.futures.Future | None = None
    # 调度 PUSH 任务并收集 Future。
    scratchpad: PregelScratchpad = task().config[CONF][CONFIG_KEY_SCRATCHPAD]  # type: ignore[union-attr]
    # 回调返回新任务时，调度该任务。
    if next_task := schedule_task(
        task(),  # type: ignore[arg-type]
        scratchpad.call_counter(),
        Call(
            func,
            input,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            callbacks=callbacks,
            timeout=timeout,
        ),
    ):
        if fut := next(
            (
                f
                for f, t in list(futures().items())  # type: ignore[union-attr]
                if t is not None and t == next_task.id
            ),
            None,
        ):
            # 若父任务已重试，
            # 下一个任务可能已经在运行。
            pass
        elif next_task.writes:
            # 若已执行完成，则返回结果。
            fut = concurrent.futures.Future()
            ret = next((v for c, v in next_task.writes if c == RETURN), MISSING)
            if ret is not MISSING:
                fut.set_result(ret)
            elif exc := next((v for c, v in next_task.writes if c == ERROR), None):
                fut.set_exception(
                    exc if isinstance(exc, BaseException) else Exception(exc)
                )
            else:
                fut.set_result(None)
        else:
            # 调度下一个任务。
            fut = submit()(  # type: ignore[misc]
                run_with_retry,
                next_task,
                retry_policy,
                configurable={
                    CONFIG_KEY_CALL: partial(
                        _call,
                        weakref.ref(next_task),
                        futures=futures,
                        retry_policy=retry_policy,
                        callbacks=callbacks,
                        schedule_task=schedule_task,
                        submit=submit,
                    ),
                },
                __reraise_on_exit__=False,
                # 下一次 tick 再启动新任务，确保本次 tick 的更新
                # 先完成提交和流式输出。
                __next_tick__=True,
            )
            # call() 任务的异常会抛入父任务，
            # 因此不应在 tick 末尾再次抛出。
            SKIP_RERAISE_SET.add(fut)
            futures()[fut] = next_task  # type: ignore[index]
    fut = cast(asyncio.Future | concurrent.futures.Future, fut)
    # 返回链式 Future，确保返回的 Future 完成前
    # 先调用 commit() 回调，以保证流式事件顺序等约束。
    return chain_future(fut, concurrent.futures.Future())


def _acall(
    task: weakref.ref[PregelExecutableTask],
    func: Callable[[Any], Awaitable[Any] | Any],
    input: Any,
    *,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    timeout: TimeoutPolicy | None = None,
    callbacks: Callbacks = None,
    # 注入的依赖。
    futures: weakref.ref[FuturesDict],
    schedule_task: Callable[
        [PregelExecutableTask, int, Call | None],
        Awaitable[PregelExecutableTask | None],
    ],
    submit: weakref.ref[Submit],
    loop: asyncio.AbstractEventLoop,
    stream: bool = False,
) -> asyncio.Future[Any] | concurrent.futures.Future[Any]:
    # 返回链式 Future，确保返回的 Future 完成前
    # 先调用 commit() 回调，以保证流式事件顺序等约束。
    try:
        in_async = asyncio.current_task() is not None
    except RuntimeError:
        in_async = False
    # 异步上下文返回异步 Future，否则返回同步 Future。
    if in_async:
        fut: asyncio.Future[Any] | concurrent.futures.Future[Any] = asyncio.Future(
            loop=loop
        )
    else:
        fut = concurrent.futures.Future()
    # 调度下一个任务。
    run_coroutine_threadsafe(
        _acall_impl(
            fut,
            task,
            func,
            input,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            timeout=timeout,
            callbacks=callbacks,
            futures=futures,
            schedule_task=schedule_task,
            submit=submit,
            loop=loop,
            stream=stream,
        ),
        loop,
        lazy=False,
    )
    return fut


async def _acall_impl(
    destination: asyncio.Future[Any] | concurrent.futures.Future[Any],
    task: weakref.ref[PregelExecutableTask],
    func: Callable[[Any], Awaitable[Any] | Any],
    input: Any,
    *,
    retry_policy: Sequence[RetryPolicy] | None = None,
    cache_policy: CachePolicy | None = None,
    timeout: TimeoutPolicy | None = None,
    callbacks: Callbacks = None,
    # 注入的依赖。
    futures: weakref.ref[FuturesDict[asyncio.Future, asyncio.Event]],
    schedule_task: Callable[
        [PregelExecutableTask, int, Call | None],
        Awaitable[PregelExecutableTask | None],
    ],
    submit: weakref.ref[Submit],
    loop: asyncio.AbstractEventLoop,
    stream: bool = False,
) -> None:
    try:
        fut: asyncio.Future | None = None
        # 调度 PUSH 任务并收集 Future。
        scratchpad: PregelScratchpad = task().config[CONF][CONFIG_KEY_SCRATCHPAD]  # type: ignore[union-attr]
        # 回调返回新任务时，调度该任务。
        if next_task := await schedule_task(
            task(),  # type: ignore[arg-type]
            scratchpad.call_counter(),
            Call(
                func,
                input,
                retry_policy=retry_policy,
                cache_policy=cache_policy,
                callbacks=callbacks,
                timeout=timeout,
            ),
        ):
            if fut := next(
                (
                    f
                    for f, t in list(futures().items())  # type: ignore[union-attr]
                    if t is not None and t == next_task.id
                ),
                None,
            ):
                # 若父任务已重试，
                # 下一个任务可能已经在运行。
                pass
            elif next_task.writes:
                # 若已执行完成，则返回结果。
                fut = asyncio.Future(loop=loop)
                ret = next((v for c, v in next_task.writes if c == RETURN), MISSING)
                if ret is not MISSING:
                    fut.set_result(ret)
                elif exc := next((v for c, v in next_task.writes if c == ERROR), None):
                    fut.set_exception(
                        exc if isinstance(exc, BaseException) else Exception(exc)
                    )
                else:
                    fut.set_result(None)
            else:
                # 调度下一个任务。
                fut = cast(
                    asyncio.Future,
                    submit()(  # type: ignore[misc]
                        arun_with_retry,
                        next_task,
                        retry_policy,
                        stream=stream,
                        configurable={
                            CONFIG_KEY_CALL: partial(
                                _acall,
                                weakref.ref(next_task),
                                stream=stream,
                                futures=futures,
                                schedule_task=schedule_task,
                                submit=submit,
                                loop=loop,
                            ),
                        },
                        __name__=next_task.name,
                        __cancel_on_exit__=True,
                        __reraise_on_exit__=False,
                        # 下一次 tick 再启动新任务，确保本次 tick 的更新
                        # 先完成提交和流式输出。
                        __next_tick__=True,
                    ),
                )
                # call() 任务的异常会抛入父任务，
                # 因此不应在 tick 末尾再次抛出。
                SKIP_RERAISE_SET.add(fut)
                futures()[fut] = next_task  # type: ignore[index]
        if fut is not None:
            chain_future(fut, destination)
        else:
            destination.set_exception(RuntimeError("Task not scheduled"))
    except Exception as exc:
        destination.set_exception(exc)
