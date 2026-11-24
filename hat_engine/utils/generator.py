# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import os
from collections.abc import Iterable
from typing import Any, Generator, Tuple

from .traceback_monitor import dataloader_traceback

__all__ = ["prefetch_iterator"]


def prefetch_iterator(
    iterable: Iterable,
) -> Generator[Tuple[Any, bool], None, None]:
    """Return an iterator that pre-fetches and caches the next item."""  # noqa: E501
    it = iter(iterable)
    try:
        # the iterator may be empty from the beginning
        last = next(it)
    except StopIteration:
        return

    dataloader_traceback_interval = int(
        os.environ.get("HAT_DATALOADER_TRACEBACK_INTERVAL", 600)
    )
    if int(dataloader_traceback_interval) > 0:
        dataloader_traceback(it, dataloader_traceback_interval)
    yield last
    for val in it:
        last = val
        yield last
