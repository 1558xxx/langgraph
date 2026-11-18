"""改编自：
https://github.com/oittaa/uuid6-python/blob/main/src/uuid6/__init__.py#L95
内置此实现以避免 uuid6 包的安装问题。
"""

from __future__ import annotations

import random
import time
import uuid

_last_v6_timestamp = None


class UUID(uuid.UUID):
    r"""UUID 草案版本对象。"""

    __slots__ = ()

    def __init__(
        self,
        hex: str | None = None,
        bytes: bytes | None = None,
        bytes_le: bytes | None = None,
        fields: tuple[int, int, int, int, int, int] | None = None,
        int: int | None = None,
        version: int | None = None,
        *,
        is_safe: uuid.SafeUUID = uuid.SafeUUID.unknown,
    ) -> None:
        r"""创建 UUID。"""

        if int is None or [hex, bytes, bytes_le, fields].count(None) != 4:
            return super().__init__(
                hex=hex,
                bytes=bytes,
                bytes_le=bytes_le,
                fields=fields,
                int=int,
                version=version,
                is_safe=is_safe,
            )
        if not 0 <= int < 1 << 128:
            raise ValueError("int is out of range (need a 128-bit value)")
        if version is not None:
            if not 6 <= version <= 8:
                raise ValueError("illegal version number")
            # 将变体设置为 RFC 4122。
            int &= ~(0xC000 << 48)
            int |= 0x8000 << 48
            # 设置版本号。
            int &= ~(0xF000 << 64)
            int |= version << 76
        super().__init__(int=int, is_safe=is_safe)

    @property
    def subsec(self) -> int:
        return ((self.int >> 64) & 0x0FFF) << 8 | ((self.int >> 54) & 0xFF)

    @property
    def time(self) -> int:
        if self.version == 6:
            return (
                (self.time_low << 28)
                | (self.time_mid << 12)
                | (self.time_hi_version & 0x0FFF)
            )
        if self.version == 7:
            return self.int >> 80
        if self.version == 8:
            return (self.int >> 80) * 10**6 + _subsec_decode(self.subsec)
        return super().time


def _subsec_decode(value: int) -> int:
    return -(-value * 10**6 // 2**20)


def uuid6(node: int | None = None, clock_seq: int | None = None) -> UUID:
    r"""UUIDv6 与 UUIDv1 的字段兼容，通过重新排列字段改善数据库中的数据局部性。UUIDv6 主要用于已有 UUIDv1 的场景；没有 UUIDv1 历史兼容需求的系统应考虑使用 UUIDv7。

    未提供 `node` 时，随机选择一个 48 位数字。提供 `clock_seq` 时将其用作序列号，否则随机选择一个 14 位序列号。
    """

    global _last_v6_timestamp

    nanoseconds = time.time_ns()
    # 0x01b21dd213814000 表示 UUID 纪元起点
    # 1582-10-15 00:00:00 与 Unix 纪元起点 1970-01-01 00:00:00 之间的 100 纳秒间隔数。
    timestamp = nanoseconds // 100 + 0x01B21DD213814000
    if _last_v6_timestamp is not None and timestamp <= _last_v6_timestamp:
        timestamp = _last_v6_timestamp + 1
    _last_v6_timestamp = timestamp
    if clock_seq is None:
        clock_seq = random.getrandbits(14)  # 使用随机值代替稳定存储。
    if node is None:
        node = random.getrandbits(48)
    time_high_and_time_mid = (timestamp >> 12) & 0xFFFFFFFFFFFF
    time_low_and_version = timestamp & 0x0FFF
    uuid_int = time_high_and_time_mid << 80
    uuid_int |= time_low_and_version << 64
    uuid_int |= (clock_seq & 0x3FFF) << 48
    uuid_int |= node & 0xFFFFFFFFFFFF
    return UUID(int=uuid_int, version=6)
