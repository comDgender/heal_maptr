# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# From PyTorch:
#
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)  # noqa
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)  # noqa
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain
#
# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.
#
# All contributions by Tri Dao:
# Copyright (c) 2024 Tri Dao.
# All rights reserved.
#
# All contributions by Arm:
# Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further  # noqa
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is  # noqa
# committed.
#
# All rights reserved.
#
# See the NOTICE-pytorch file distributed with this work for additional information regarding copyright ownership.  # noqa
#
# This file has been modified by Beijing Horizon Information Technology Co., Ltd. on 2026/06/10.
# diff 1: download with retry;
# diff 2: support to download file on rank 0 with flag;
# diff 3: add barrier to ensure all ranks have downloaded the file;
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America  # noqa
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import errno
import faulthandler
import logging
import os
import sys
import time
import warnings
from distutils.version import LooseVersion
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlparse  # noqa: F401

import torch
import torch.distributed as dist
from torch.hub import (
    HASH_REGEX,
    _is_legacy_zip_format,
    _legacy_zip_load,
    download_url_to_file,
    get_dir,
)
from typing_extensions import TypeAlias

try:
    from torch.serialization import MAP_LOCATION
except ImportError:
    MAP_LOCATION: TypeAlias = Optional[
        Union[
            Callable[[torch.Tensor, str], torch.Tensor],
            torch.device,
            str,
            Dict[str, str],
        ]
    ]

logger = logging.getLogger(__name__)

__all__ = [
    "download_url_to_file_with_retry",
    "load_state_dict_from_url",
]


def download_url_to_file_with_retry(
    url: str,
    dst: str,
    hash_prefix: Optional[str] = None,
    progress: bool = True,
    max_retry: int = 3,
):
    """Download object at the given URL to a local path.

    Args:
        url: URL of the object to download
        dst: Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix: If not None, the SHA256 downloaded file should start
            with ``hash_prefix``. Default: None
        progress: whether or not to display a progress bar to stderr.
            Default: True
        max_retry: Retry times if download failed.

    """

    for i in range(1, max_retry + 1):
        try:
            download_url_to_file(
                url=url,
                dst=dst,
                hash_prefix=hash_prefix,
                progress=progress,
            )
            break
        except Exception as e:
            if i == max_retry:
                logger.warning(
                    f"Download failed and has retried {max_retry} times:{e}"
                )
            else:
                logger.warning(
                    f"Download failed caused by: {e}. "
                    f"Remaining_retry: {max_retry - i}."
                )


def load_state_dict_from_url(
    url: str,
    model_dir: Optional[str] = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    weights_only: bool = False,
    max_retry: int = 3,
    host_rank_zero_only: bool = True,
) -> Dict[str, Any]:
    r"""Overwrite of `torch.hub.load_state_dict_from_url`.

    Note: this function is same to `torch.hub.load_state_dict_from_url`, but
        with added retry download and `host_rank_zero_only` option.

    Args:
        url: URL of the object to download
        model_dir: directory in which to save the object
        map_location: a function or a dict specifying how to remap storage l
            ocations (see torch.load)
        progress: whether or not to display a progress bar to stderr.
            Default: True
        check_hash: If True, the filename part of the URL should follow the
            naming convention ``filename-<sha256>.ext`` where ``<sha256>`` is
            the first eight or more digits of the SHA256 hash of the contents
            of the file. The hash is used to ensure unique names and to verify
            the contents of the file. Default: False
        file_name: name for the downloaded file. Filename from ``url`` will be
            used if not set.
        weights_only: If True, only weights will be loaded and no complex
            pickled objects. Recommended for untrusted sources.
            See :func:`~torch.load` for more details.
        max_retry: Retry times if download failed.
        host_rank_zero_only: If True, only local rank 0 of each host will
            download the file.

    """
    # Issue warning to move data if old env is set
    if os.getenv("TORCH_MODEL_ZOO"):
        warnings.warn(
            "TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead"
        )

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)

    def _download_url_to_file():
        if not os.path.exists(cached_file):
            sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
            hash_prefix = None
            if check_hash:
                r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
                hash_prefix = r.group(1) if r else None
            # diff 1: with retry
            download_url_to_file_with_retry(
                url, cached_file, hash_prefix, progress=progress
            )

    # diff 2: if `host_rank_zero_only=True`, only download file on
    # each host's local rank 0
    host_local_rank = os.getenv("HAT_HOST_LOCAL_RANK", None)
    if host_rank_zero_only and host_local_rank is not None:
        if host_local_rank == "0":
            _download_url_to_file()
        else:
            while not os.path.exists(cached_file):
                sys.stderr.write(
                    f'Waiting for local host local rank 0 to download: "{url}" to {cached_file}\n'  # noqa: E501
                )
                time.sleep(3)
    else:
        _download_url_to_file()

    # diff 3: add barrier to ensure all ranks have downloaded the file
    if dist.is_initialized():
        dist.barrier()

    if LooseVersion(torch.__version__) >= LooseVersion("2.0.0"):
        extra_kwargs = {"weights_only": weights_only}
    else:
        extra_kwargs = {}

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(
            cached_file, model_dir, map_location, **extra_kwargs
        )
    return torch.load(cached_file, map_location=map_location, **extra_kwargs)


def hook_torch_signal_handler():
    """Replace torch dataloader signal handler.

    Note:
        This hook is designed to print stack information when the dataloader
        worker encounters a signal error.
    """
    orig_set_worker_signal_handler = (
        torch.utils.data._utils.signal_handling._set_worker_signal_handlers
    )

    def set_worker_signal_handlers(*arg: Any):
        faulthandler.disable()
        orig_set_worker_signal_handler(*arg)
        faulthandler.enable(all_threads=True)

    torch.utils.data._utils.signal_handling._set_worker_signal_handlers = (
        set_worker_signal_handlers
    )


# hook torch dataloader signal handler
hook_torch_signal_handler()
