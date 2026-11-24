# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import contextlib
import datetime
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from typing import AnyStr, Union

import hat_engine.utils.filestream.io as file_io
from hat_engine.utils.filestream.bucket import (
    BucketClient,
    is_dmp_url,
    url_to_local_path,
)

__all__ = [
    "FileHelper",
]

FILECHUNK_PREFIX = "FILE_CHUNK://"


def is_file_chunk(url: str):
    return url.startswith(FILECHUNK_PREFIX)


def parse_file_chunk(chunk_url: str):
    assert is_file_chunk(chunk_url)
    path_with_chunk = chunk_url[len(FILECHUNK_PREFIX) :]
    splited = path_with_chunk.split(":")
    offset, length = int(splited[0]), int(splited[1])
    path = ":".join(splited[2:])
    return path, int(offset), int(length)


def get_file_chunk_url(url: str, offset: int, length: int):
    chunk_url = "%s%d:%d:%s" % (FILECHUNK_PREFIX, offset, length, url)
    return chunk_url


class FileHelper(object):
    def __init__(self, tmp_dir="tmp/", remove_at_exit=True):
        self._rel_tmp_dir = tmp_dir
        self._tmp_dir = os.path.abspath(tmp_dir)
        self._remove_at_exit = remove_at_exit
        self._created_cache = set()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_created_cache"] = set()
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self._tmp_dir = os.path.abspath(self._rel_tmp_dir)

    def get_tmp_dir(self, with_time_suffix=False):
        if with_time_suffix:
            new_dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            new_tmp_dir = os.path.join(self._tmp_dir, new_dir_name)
            if not os.path.exists(new_tmp_dir):
                self._created_cache.add(new_tmp_dir)
            os.makedirs(new_tmp_dir, exist_ok=True)
            return new_tmp_dir
        else:
            if not os.path.exists(self._tmp_dir):
                self._created_cache.add(self._tmp_dir)
            os.makedirs(self._tmp_dir, exist_ok=True)
            return self._tmp_dir

    def read_buf_from_file_url(
        self, dmp_client: BucketClient, file_url: str
    ) -> bytes:
        if is_file_chunk(file_url):
            file_url, offset, length = self.parse_file_chunk(file_url)
            if is_dmp_url(file_url):
                file_path = url_to_local_path(file_url, dmp_client)
            else:
                file_path = file_url
            with open(file_path, "rb") as fin:
                fin.seek(offset)
                buf = fin.read(length)
        else:
            if is_dmp_url(file_url):
                file_path = url_to_local_path(file_url, dmp_client)
            else:
                file_path = file_url
            with open(file_path, "rb") as fin:
                buf = fin.read()
        return buf

    def write_buf_to_file_url(
        self, dmp_client: BucketClient, file_url: str, buf: bytes
    ):
        if is_file_chunk(file_url):
            raise ValueError(
                f"Writing buf to file chunk is not supported: {file_url}"
            )
        else:
            if is_dmp_url(file_url):
                file_path = url_to_local_path(file_url, dmp_client)
            else:
                file_path = file_url
            with open(file_path, "wb") as fout:
                fout.write(buf)

    def parse_file_chunk(self, chunk_url: str):
        return parse_file_chunk(chunk_url)

    def __enter__(self):
        return self

    def __exit__(self, ptype, value, trace):
        self._cleanup()

    def _cleanup(self):
        if self._remove_at_exit:
            for dirname in self._created_cache:
                if os.path.exists(dirname):
                    try:
                        shutil.rmtree(dirname)
                    except Exception:
                        pass

    def __del__(self):
        self._cleanup()


def _union_hash(data: Union[bytes, str, dict]):
    assert isinstance(
        data, (bytes, str, dict)
    ), f"invalid input data type: {type(data)}"
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True).encode("utf-8")
    elif isinstance(data, str):
        data = data.encode(encoding="utf-8")
    return hashlib.md5(data).hexdigest()


@contextlib.contextmanager
def make_temp_directory(
    prefix: str = None,
    key: Union[bytes, str, dict] = None,
    exist_ok: bool = False,
    clear_on_exp: bool = True,
) -> AnyStr:
    """Make a temporary directory.

    For Examples
    --------

    >>> with make_temp_directory(clear_on_exp=False) as dir_path:
    ...     print(dir_path)
    /tmp/93805474a98c4be9b39dcdcb46b5bb5c

    Args:
        prefix : string
            If prefix is not None, the directory path will
            starts with that prefix; otherwise, the system
            temperory directory is used.
        key : Union[bytes, str, dict]
            If key is not None, the directory name is hashed by the key;
            otherwise, the directory name is provided from uuid.
        exist_ok : bool
            Weather to raise exception while the directory already exist.
        clear_on_exp : bool
            Weather to clear the directory when exception occurs.

    Returns:
        dir_path: The temporary directory path
    """
    if prefix is None:
        prefix = tempfile.gettempdir()
    if key is not None:
        dir_name = _union_hash(key)
    else:
        dir_name = "".join(str(uuid.uuid4()).split("-"))
    dir_name = os.path.join(prefix, dir_name)
    if file_io.exists(dir_name):
        if not exist_ok:
            raise IOError(f"file or director {dir_name} exist")
    else:
        file_io.makedirs(dir_name)
    try:
        yield dir_name
    except:  # noqa
        if clear_on_exp:
            file_io.delete(dir_name, recursive=True)
        raise
    else:
        file_io.delete(dir_name, recursive=True)
