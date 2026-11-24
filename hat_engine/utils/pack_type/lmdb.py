# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Project RoboOrchard
#
# Copyright (c) 2024-2026 Beijing Horizon Information Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import logging
from typing import Any, List, Tuple, Union

import lmdb
from timeout_decorator import TimeoutError

from hat_engine.utils.apply_func import _as_list
from hat_engine.utils.timer import AlarmTimerDecorator, maybetimeout
from .base import PackType

__all__ = ["Lmdb"]

logger = logging.getLogger(__name__)


def alarm_timer_callback(ret: Any, duration: float) -> Tuple[Any, str]:
    """
    Define a callback function for the alarm timer.

    Args:
        ret: Return value of function.
        duration: Duration of function.
    """
    res, uri = ret
    msg = f"LMDB open [{uri}] file cost {duration:.2f}s."
    return res, msg


class Lmdb(PackType):
    """
    Abstact class of LMDB, which include all operators.

    Args:
        uri: Path to lmdb file.
        writable: Writable flag for opening LMDB.
        commit_step: The step for commit.
        map_size:
            Maximum size database may grow to, used to size the memory mapping.
            If map_size is None, map_size will set to 10M while reading,
            set to 1T while writing.
        kwargs: Kwargs for open lmdb file and perf read data.

    """

    def __init__(
        self,
        uri: str,
        writable: bool = True,
        commit_step: int = 1,
        map_size: int = None,
        **kwargs,
    ):
        super(Lmdb, self).__init__(**kwargs)
        self.uri = uri
        self.writable = writable
        self.kwargs = kwargs
        self.lmdb_kwargs = {}
        self.lmdb_kwargs["map_size"] = map_size
        # default lmdb settings

        self.lmdb_kwargs["meminit"] = self.kwargs.get("meminit", False)
        self.lmdb_kwargs["map_async"] = self.kwargs.get("map_async", True)
        self.lmdb_kwargs["sync"] = self.kwargs.get("sync", False)
        if not writable:
            self.lmdb_kwargs["readonly"] = True
            self.lmdb_kwargs["lock"] = False
            # set map_size to 10M while reading.
            if self.lmdb_kwargs.get("map_size") is None:
                self.lmdb_kwargs["map_size"] = 10485760
        else:
            # set map_size to 1T while writing.
            if self.lmdb_kwargs.get("map_size") is None:
                self.lmdb_kwargs["map_size"] = 1024 ** 4
        # LMDB env
        self.env = None
        self.txn = None
        self.open()
        if not self.writable:
            self._create_txn()
        # pack settings
        self.commit_step = commit_step
        self.put_idx = 0

    def read_idx(self, idx: Union[int, str]) -> bytes:
        """Read data by idx."""
        idx = "{}".format(idx).encode("ascii")
        try:
            return self.get(idx)
        except TimeoutError as exception:
            logger.error(
                f"Time out when reading data with index of "
                f"{idx} from {self.uri}"
            )
            raise exception

    @maybetimeout(seconds=1800)
    def get(self, idx: Union[int, str]) -> bytes:
        if self.txn is None:
            self._create_txn()
        try:
            return self.txn.get(idx)
        except ValueError as exception:
            logging.error(f"key:{idx} not in {self.uri}")
            raise exception

    def write(self, idx: Union[int, str], record: bytes):
        """Write data into lmdb file."""
        if self.env is None:
            self.open()
        if self.txn is None:
            self._create_txn()
        self.txn.put("{}".format(idx).encode("ascii"), record)
        self.put_idx += 1
        if self.put_idx % self.commit_step == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=self.writable)

    @maybetimeout(seconds=1800)
    @AlarmTimerDecorator(900, callback_func=alarm_timer_callback)
    def open_lmdb(self):
        return (lmdb.open(self.uri, **self.lmdb_kwargs), self.uri)

    def open(self):
        """Open lmdb file."""
        if self.env is None:
            try:
                self.env = self.open_lmdb()
            except TimeoutError as exception:
                logger.error(f"Time out when opening {self.uri}")
                raise exception

    def _create_txn(self):
        """Create lmdb transaction."""
        if self.env is None:
            self.open()
        if self.txn is None:
            self.txn = self.env.begin(write=self.writable)

    def close(self):
        """Close lmdb file."""
        if self.env is not None:
            if self.writable and self.txn is not None:
                if self.put_idx % self.commit_step != 0:
                    self.txn.commit()
                self.put_idx = 0
                self.env.sync()
            self.env.close()
            self.env = None
            self.txn = None

    def reset(self):
        """Reset open file."""
        if self.env is None and self.txn is None:
            self.open()
        else:
            self.close()
            self.open()

    def get_keys(self):
        """Get all keys."""
        if self.txn is None:
            self._create_txn()
        try:
            idx = "{}".format("__len__").encode("ascii")
            return range(int(self.txn.get(idx)))
        except Exception:
            # traversal may be slow while too much keys
            keys = []
            for key, _value in self.txn.cursor():
                keys.append(key)
            return keys

    def __len__(self):
        """Get the length."""
        if self.txn is None:
            self._create_txn()
        try:
            idx = "{}".format("__len__").encode("ascii")
            return int(self.txn.get(idx))
        except Exception:
            return self.txn.stat()["entries"]

    def __getstate__(self):
        state = self.__dict__
        self.close()
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.open()


class LmdbReadList(PackType):
    """Class of LMDB reader, used to read from list of lmdbs.

    Args:
        uri: Path to lmdb file, support str and list[str].
        kwargs: Kwargs for open lmdb file and perf read. refer to Lmdb.

    """

    def __init__(
        self,
        uris: Union[str, List[str]],
        **kwargs,
    ):
        super(LmdbReadList, self).__init__(**kwargs)
        self.uris = _as_list(uris)
        self.txns = None
        if self.uris:
            self.txns = [
                Lmdb(
                    uri,
                    writable=False,
                    **kwargs,
                )
                for uri in self.uris
            ]
        self.kwargs = kwargs

    def __len__(self):
        """Get the length."""
        return sum([len(txn) for txn in self.txns])

    def read_idx(self, idx: Union[int, str]) -> bytes:
        """Read data by idx."""
        for txn_idx, txn in enumerate(self.txns):
            try:
                res = txn.read(idx)
                if res is not None:
                    return res
            except TimeoutError as exception:
                logger.error(
                    f"Time out when reading data with index of "
                    f"{idx} from {self.uris[txn_idx]}"
                )
                raise exception
        raise IndexError(
            f"Index Error when reading data with index of {idx} from {self.uris}"  # noqa [E501]
        )
