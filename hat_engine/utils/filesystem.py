# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import json
import logging
import os
import pathlib
import pickle
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Generator, Tuple, Union

import fsspec
import yaml
from filelock import FileLock

logger = logging.getLogger(__name__)

__all__ = [
    "get_filesystem",
    "join_path",
    "file_load",
    "split_file_path",
    "FileLockExt",
    "load_pyfile",
    "load_json",
    "load_yaml",
    "load_requirements",
]


def get_filesystem(path: Union[str, Path]) -> fsspec.AbstractFileSystem:
    """Get fsspec.filesystem from provided path.

       Filesystem can support file, http, hdfs.

    Args:
        path (str or Path): Provided path for filesystem.
    Returns:
        fsspec.filesystem

    """

    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        # HDFS depends on pyarrow
        # HTTP depends on aiohttp
        return fsspec.filesystem(path.split(":", 1)[0])
    else:
        # use local filesystem
        return fsspec.filesystem("file")


def join_path(
    prefix_path: Union[str, os.PathLike],
    input: Any,
    ignore_key: Union[str, Sequence] = None,
) -> Any:  # noqa: D205,D400
    """
    Join any str objects or os.PathLike objects
    in input with prefix path recursively. When
    ignore_key is not None, stop joining for the
    objects whose key is included in ignore_key.

    Examples::

        >>> import pathlib
        >>> input = dict(
        ...    path1='user1',
        ...    path2=['user2','user3'],
        ...    path3=pathlib.Path('user4'),
        ...    path4={"group1":"user5"},
        ...    v=2)
        ... ignore_key = ["group1"]

        >>> join_path('/home',input,ignore_key)
        OrderedDict([('path1', '/home/user1'),
        ('path2', ['/home/user2', '/home/user3']),
        ('path3', '/home/user4'),
        ('path4', OrderedDict([('group1', 'user5')])),
        ('v', 2)])
    Args:
        prefix_path (str): prefix path.
        input (any): any format path to join.
        ignore_key (Sequence): ignored keys for path join.
    """
    if isinstance(input, (str, os.PathLike)):
        return os.path.join(prefix_path, input)
    elif isinstance(input, Sequence):
        return [join_path(prefix_path, one, ignore_key) for one in input]
    elif isinstance(input, Mapping):
        result = OrderedDict()
        for k, v in input.items():
            if ignore_key is not None and k in ignore_key:
                result[k] = v
            else:
                result[k] = join_path(prefix_path, v, ignore_key)
        return result
    else:
        return input


def pickle_load(data):  # type: ignore
    out = pickle.load(data)
    return out


def file_load(file: str) -> Generator:
    with open(file, "rb") as f:
        while True:
            try:
                yield pickle_load(f)
            except EOFError:
                break


def split_file_path(file_path: str) -> Tuple:
    """Get the dir path, file name and file extension of input file.

    Args:
        file_path: Input file path.

    Returns:
        (dir path, file name and file extension) of input file.
    """
    path = pathlib.Path(file_path)
    return (path.parent, path.name, "".join(path.suffixes))


def check_file_exist_with_warning(
    file_path: str,
    overwrite: bool = True,
):
    """Check whether the file exists and show overwrite result.

    Args:
        file_path: Path of file to check.
        overwrite: Whether to overwrite existed file.
                If True, will delete the old file.
    """
    if os.path.exists(file_path):
        if overwrite:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(
                    f"Remove `{file_path}` failed, make sure you have "
                    "the permission to manipulate the folder where the files are saved"  # noqa E501
                )
                logger.info(str(e))
        else:
            logger.warning(
                f"File `{file_path}` already exists, set `overwrite=True` "  # noqa E501
                "to delete the old file to avoid unexpected results."
            )


def _check_path_exists(path, local_only=False):
    if not os.path.exists(path):
        raise FileNotFoundError("%s does not exists" % path)


class FileLockExt(FileLock):
    """Extend :py:class:`filelock.FileLock` to allow clear lock file at exit."""  # noqa

    def __init__(self, *args, clear_at_exit: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_at_exit = clear_at_exit

    def release(self, *args, **kwargs):
        super().release(*args, **kwargs)
        if self.clear_at_exit and not self.is_locked:
            try:
                os.remove(self.lock_file)
            except FileNotFoundError:
                pass


def load_pyfile(
    filename,
    allow_unsafe=False,
    allow_duplicate_module=False,
    pop_module=True,
):
    """Load python file."""

    _check_path_exists(filename, local_only=True)
    module_name = os.path.basename(filename)
    if os.path.isfile(filename):
        module_name = module_name[:-3]

    lock = FileLockExt(
        module_name + ".hat_easydict_loader.lock", clear_at_exit=True
    )
    with lock:
        # check duplicate names
        if not allow_duplicate_module:
            if module_name in sys.modules:
                raise ValueError(
                    f"module name {module_name} for {filename} is duplicated."
                )

        config_dir = os.path.dirname(filename)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        cfg = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith("__")
        }
        if pop_module:
            sys.modules.pop(module_name)

    return cfg


def load_json(filename, allow_unsafe=False):
    """Load json file."""

    _check_path_exists(filename)
    with open(filename, "r") as f:
        cfg = json.load(f)
    return cfg


def load_yaml(filename, allow_unsafe=False):
    """Load yaml file."""

    _check_path_exists(filename)
    with open(filename, "r") as f:
        if allow_unsafe:
            cfg = yaml.unsafe_load(f)
        else:
            cfg = yaml.safe_load(f)
    return cfg


def load_requirements(filename):
    """Load requirements file."""

    _check_path_exists(filename)
    with open(filename, "r") as f:
        cfg = f.read().splitlines()
    return cfg


ext_to_load_fn_map = {
    ".py": load_pyfile,
    ".json": load_json,
    ".yaml": load_yaml,
    ".yml": load_yaml,
}


def walk_package_py_file(dir, skip_dirs=(), skip_files=()):
    filter_path = ".".join(dir.replace("/", ".").split(".")[:-1]) + "."
    skip_dir_set = set(skip_dirs)
    for (root, dirs, files) in os.walk(dir, topdown=True):
        # Prune skipped directories so the walk does not descend into them.
        dirs[:] = [d for d in dirs if d not in skip_dir_set]
        if os.path.basename(root) in skip_dir_set:
            continue
        for file in files:
            if file in skip_files:
                continue
            if file.endswith(".py"):
                module = os.path.join(root, file).replace("/", ".")[:-3]
                yield module.replace(filter_path, "")
