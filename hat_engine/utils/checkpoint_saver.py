# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Copyright 2023 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

# This file has been modified by Beijing Horizon Information Technology Co., Ltd. on 2026/6/9.
# Use AsyncCheckpointSaver to save checkpoint into shared memory, which
# can be loaded by a new training process when the original training process fails. # noqa: E501

# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import mmap
import os
import pickle
import struct
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Callable, List, Mapping, Sequence, Tuple

import _posixshmem
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SharedMemory(shared_memory.SharedMemory):
    """
    Customization of the SharedMemory is necessary.

    The 'resource_tracker.ResourceTracker' in Python will unlink and remove the
    file if one process fails. Our objective is to ensure that the training
    process does not unlink the shared memory upon failure,
    hereby allowing a new training process to commence utilizing
    the existing shared memory to load checkpoint.
    """

    _name = None
    _fd = -1
    _mmap = None
    _buf = None
    _flags = os.O_RDWR
    _mode = 0o600
    _prepend_leading_slash = True

    def __init__(self, name=None, create=False, size=0):
        self.close_in_del = True
        if not size >= 0:
            raise ValueError("'size' must be a positive integer")
        if create:
            self._flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            if size == 0:
                raise ValueError(
                    "'size' must be a positive number different from zero"
                )
        if name is None and not self._flags & os.O_EXCL:
            raise ValueError("'name' can only be None if create=True")

        if name is None:
            while True:
                name = shared_memory._make_filename()
                try:
                    self._fd = _posixshmem.shm_open(
                        name, self._flags, mode=self._mode
                    )
                except FileExistsError:
                    continue
                self._name = name
                break
        else:
            name = "/" + name if self._prepend_leading_slash else name
            self._fd = _posixshmem.shm_open(name, self._flags, mode=self._mode)
            self._name = name
        try:
            if create and size:
                os.ftruncate(self._fd, size)
            stats = os.fstat(self._fd)
            size = stats.st_size
            self._mmap = mmap.mmap(self._fd, size)
        except OSError:
            self.unlink()
            raise

        self._size = size
        self._buf = memoryview(self._mmap)

    def unlink(self):
        if self._name:
            try:
                _posixshmem.shm_unlink(self._name)
            except FileNotFoundError:
                pass

    def __del__(self):
        if self.close_in_del:
            super().__del__()


@dataclass
class TensorMeta:
    shape: Tuple[int] = None  # type: ignore
    dtype: torch.dtype = None  # type: ignore
    element_size: int = 0
    numel: int = 0
    offset: int = 0


@dataclass
class ScalarMeta:
    element_size: int = 0
    offset: int = 0


def _traverse_state_dict(value: object, visitor: Callable[[object], None]):
    """Invoke ``visitor`` for each value recursively in ``state_dict``."""
    elem_type = type(value)
    if isinstance(value, Mapping):
        return elem_type(
            {k: _traverse_state_dict(v, visitor) for k, v in value.items()}
        )
    elif isinstance(value, Sequence) and not isinstance(value, str):
        return elem_type([_traverse_state_dict(d, visitor) for d in value])
    else:
        return visitor(value)


def _read_state_dict_from_shm(meta_dict, tensor_shm):
    state_dict = _traverse_state_dict(
        meta_dict,
        lambda x: _read_tensor_from_buf(x, tensor_shm),
    )
    return state_dict


def _read_tensor_from_buf(value, shm_tensor_buffer):
    if isinstance(value, TensorMeta):
        if value.numel == 0:
            return torch.tensor([], dtype=value.dtype)
        else:
            shm_tensor = torch.frombuffer(
                buffer=shm_tensor_buffer.buf,
                dtype=value.dtype,
                offset=value.offset,
                count=value.numel,
            )
            ori_data = shm_tensor.reshape(value.shape)
            return ori_data
    elif isinstance(value, ScalarMeta):
        ori_data = bytes(
            shm_tensor_buffer.buf[
                value.offset : value.offset + value.element_size
            ]
        )
        if value.element_size == 8:
            ori_data = struct.unpack("d", ori_data)[0]
        elif value.element_size == 4:
            ori_data = struct.unpack("i", ori_data)[0]
        return ori_data
    else:
        return value


def _create_shared_memory(name, create, size=0):
    if not create:
        try:
            return SharedMemory(name=name)
        except FileNotFoundError:
            return None

    if size == 0:
        logger.warning("Cannot create the shared memory with size = 0.")
        return None

    kwargs = {"name": name, "create": create, "size": size}
    try:
        return SharedMemory(**kwargs)
    except FileExistsError:
        shm = SharedMemory(name=name)
        if shm.size == size:
            return shm

        logger.info(
            f"The old size is {shm.size} and "
            f"create a new memory buffer with size {size}."
        )
        shm.unlink()
        return SharedMemory(**kwargs)


def _traverse_copy_to_shm(value, meta, buffer):
    if isinstance(value, Mapping):
        for k, v in value.items():
            m = meta[k]
            if isinstance(v, (Mapping, List, Tuple)):
                _traverse_copy_to_shm(v, m, buffer)
            elif torch.is_tensor(v):
                _write_shared_memory_tensor(v, m, buffer)
            elif isinstance(v, (int, float)):
                _write_shared_memory_scalar(v, m, buffer)
    elif isinstance(value, (List, Tuple)):
        for i, v in enumerate(value):
            m = meta[i]
            if isinstance(v, (Mapping, List, Tuple)):
                _traverse_copy_to_shm(v, m, buffer)
            elif torch.is_tensor(v):
                _write_shared_memory_tensor(v, m, buffer)
            elif isinstance(v, (int, float)):
                _write_shared_memory_scalar(v, m, buffer)


def _write_shared_memory_tensor(value: torch.Tensor, meta: TensorMeta, buffer):
    if value.numel() == 0:
        return
    with torch.no_grad():
        shm_tensor = torch.frombuffer(
            buffer, dtype=value.dtype, count=value.numel(), offset=meta.offset
        ).reshape(value.shape)
        shm_tensor.copy_(value)


def _write_shared_memory_scalar(value, meta, buffer):
    if isinstance(value, float):
        byte_data = struct.pack("d", value)
    elif isinstance(value, int):
        byte_data = struct.pack("i", value)
    buffer[meta.offset : meta.offset + meta.element_size] = byte_data


class SharedMemoryHandler(object):
    """Handler to write and read the shared memory with the state dict."""

    def __init__(self, shm_name="ckpt_shm"):
        self._buffer_size = 0
        self._ckpt_shm_name = shm_name
        self._meta_dict_shm_name = "meta_dict_" + shm_name
        self._need_creation = True
        self.meta_dict = {}
        self.ckpt_shared_memory = None
        self.meta_dict_shm = None

    def close(self):
        if self.ckpt_shared_memory:
            self.ckpt_shared_memory.close()
            self.ckpt_shared_memory.unlink()
        if self.meta_dict_shm:
            self.meta_dict_shm.close()
            self.meta_dict_shm.unlink()

    def _create_tensor_meta(self, value: torch.Tensor):
        if isinstance(value, (int, float)):
            if isinstance(value, int):
                element_size = 4
            elif isinstance(value, float):
                element_size = 8
            meta = ScalarMeta(
                element_size=element_size, offset=self._buffer_size
            )
            self._buffer_size += element_size
        elif torch.is_tensor(value):
            meta = TensorMeta(
                shape=tuple(value.shape),  # type: ignore
                dtype=value.dtype,
                element_size=value.element_size(),
                numel=value.numel(),
                offset=self._buffer_size,
            )
            self._buffer_size += value.numel() * value.element_size()
        else:
            meta = value
        return meta

    def save_state_dict(self, state_dict, ckpt_file, constant_data=None):
        if not self.ckpt_shared_memory:
            meta_dict = _traverse_state_dict(
                state_dict, self._create_tensor_meta
            )
            meta_dict.update(
                {"ckpt_file": ckpt_file, "constant_data": constant_data}
            )
            serialized_meta_dict = pickle.dumps(meta_dict)
            self.init_shared_memory(
                create=True,
                ckpt_size=self._buffer_size,
                meta_dict_size=len(serialized_meta_dict),
                serialized_meta_dict=serialized_meta_dict,
            )
        else:
            meta_dict = self.meta_dict

        assert self.ckpt_shared_memory is not None
        _traverse_copy_to_shm(
            state_dict, meta_dict, self.ckpt_shared_memory.buf
        )
        self.meta_dict = meta_dict
        self.ckpt_file = ckpt_file
        self.constant_data = constant_data

    def load_state_dict(self, release_shm=True, shm_name=None):
        """
        Load the state dict from the shared memory.

        Returns:
            Tuple(int, dict): The first value is the iteration step,
                the second value is the state dict.
        """
        self._meta_dict_shm_name = "meta_dict_" + shm_name
        meta_dict_shm = _create_shared_memory(
            name=self._meta_dict_shm_name, create=False
        )
        if meta_dict_shm is None:
            return None
        serialized_meta_dict = bytes(meta_dict_shm.buf)
        meta_dict = pickle.loads(serialized_meta_dict)
        self.ckpt_file = meta_dict.get("ckpt_file", None)
        self.constant_data = meta_dict.get("constant_data", None)
        ckpt_shared_memory = _create_shared_memory(
            self._ckpt_shm_name, create=False
        )
        ckpt_shared_memory.close_in_del = release_shm
        state_dict = _read_state_dict_from_shm(meta_dict, ckpt_shared_memory)
        state_dict.pop("ckpt_file", None)
        state_dict.pop("constant_data", None)
        return state_dict

    def init_shared_memory(
        self,
        create=False,
        ckpt_size=0,
        meta_dict_size=0,
        serialized_meta_dict=None,
    ):
        self.ckpt_shared_memory = _create_shared_memory(
            self._ckpt_shm_name, create=create, size=ckpt_size
        )
        self.meta_dict_shm = _create_shared_memory(
            self._meta_dict_shm_name, create=create, size=meta_dict_size
        )
        self.meta_dict_shm.buf[:meta_dict_size] = serialized_meta_dict
        self._need_creation = False


class FlashCheckpointer:
    """
    The FlashCheckpointer writes the state dict into the shared memory.

    this only blocks the training with a little time. If the training
    process failed, the agent in main process can continuously
    save the state dict from the shared memory into the storage.
    """

    def __init__(
        self,
        ckpt_name="flash_ckpt",
    ):
        self.shm_handler = SharedMemoryHandler(shm_name=ckpt_name)

    def __del__(self):
        self.close()

    def close(self):
        """Close the shared memory."""
        self.shm_handler.close()

    def save_state_dict_to_memory(self, state_dict, ckpt_file, constant_data):
        self.shm_handler.save_state_dict(state_dict, ckpt_file, constant_data)
