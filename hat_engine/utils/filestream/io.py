# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from typing import IO, Optional, Tuple, Union

from tqdm import tqdm

try:
    import hdfs
except ImportError:
    hdfs = None

from hat_engine.utils.deprecate import deprecated_warning
from hat_engine.utils.filestream.bucket import (
    DMP_PREFIX,
    BucketClient,
    BucketFileEntry,
    FileCountDict,
    FolderScan,
    split_dmp_url,
)
from hat_engine.utils.package_helper import require_packages
from hat_engine.utils.resource_manager import (
    get_current_resource_manager,
    get_resource,
)

logger = logging.getLogger(__name__)
__all__ = [
    "open_file",
    "reader",
    "exists",
    "normpath",
    "makedirs",
    "delete",
    "listdir",
    "isdir",
    "move",
    "WorkingDir",
]


HDFS_PREFIX = "hdfs://"

HDFS_NODES_DICT = {
    "yz-cpu005.hogpu.cc": ["yz-cpu005.hogpu.cc"],
    "yz-cpu006.hogpu.cc": ["yz-cpu006.hogpu.cc"],
    "hobot-bigdata": ["yz-cpu005.hogpu.cc", "yz-cpu006.hogpu.cc"],
    "alicpu001.hogpu.cc": ["alicpu001.hogpu.cc"],
    "alicpu002.hogpu.cc": ["alicpu002.hogpu.cc"],
    "hobot-bigdata-aliyun": ["alicpu001.hogpu.cc", "alicpu002.hogpu.cc"],
    "hobot-bigdata-ucloud": ["10.10.232.29", "10.10.232.28"],
}

_node2client = {}

_bucket_client = None


class LocalFileSys:
    @staticmethod
    def mv(src: str, dst: str) -> bool:
        cmds = ["mv", src, dst, "-f"]
        if not isdir(dst):
            raise KeyError("dst is not a directory")
        process = subprocess.Popen(
            cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return_code = process.wait()
        stdout, stderr = process.communicate()
        if return_code != 0:
            raise Exception(stderr.decode("utf-8"))

        return True


class AidiInfoCliFileSys:
    @staticmethod
    def aidi_inf_cli_available() -> bool:
        # run subprocess and capture output and return code
        process = subprocess.Popen(
            ["aidi-inf-cli", "bkt-file"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")
        if "level=info" not in stdout:
            logger.error(stdout)
            logger.error(stderr)
        return_code = process.returncode
        return return_code == 0

    @staticmethod
    def mv(src: str, dst: str) -> bool:
        cmds = ["aidi-inf-cli", "bkt-file", "mv", src, dst, "-f"]

        if not exists(src):
            raise FileNotFoundError("src not exists")

        if isdir(src):
            cmds.append("-R")

        if not isdir(dst):
            raise KeyError("dst is not a directory")

        process = subprocess.Popen(
            cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process.wait()
        stdout, stderr = process.communicate()
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")
        if "level=info" in stdout:
            return True
        else:
            raise Exception(stdout)


class BucketFileIO(IO):
    def __init__(self, io: IO, path: str):
        self._clt = BucketClient()
        self._io = io
        self._path = path
        self.readable = io.readable
        self.read = io.read
        self.readline = io.readline
        self.readlines = io.readlines
        self.write = io.write
        self.writable = io.writable
        self.writelines = io.writelines
        self.tell = io.tell
        self.seek = io.seek

    def __enter__(self):
        return self._io.__enter__()

    def __exit__(self, *args, **kwargs):
        need_sync = False
        if self._io.writable():
            need_sync = True
        ret = self._io.__exit__(*args, **kwargs)
        if need_sync:
            self._clt.force_sync(self._path)
        return ret

    def close(self):
        need_sync = False
        if self._io.writable():
            need_sync = True
        ret = self._io.close()
        if need_sync:
            self._clt.force_sync(self._path)
        return ret


@require_packages("hdfs")
def _get_hdfs_client(namenode):
    deprecated_warning(
        author="wujun.wen",
        deprecation_version="v3.3.4",
        removal_version="v4.0.0",
    )
    global _node2client
    assert (
        namenode in HDFS_NODES_DICT
    ), "Only support hdfs {}, given {}".format(
        HDFS_NODES_DICT.keys(), namenode
    )
    for node in HDFS_NODES_DICT[namenode]:
        if node not in _node2client:
            try:
                hdfs_client = hdfs.InsecureClient("http://" + node + ":50070")
                hdfs_client.status("/")

                _node2client[node] = hdfs_client
                return hdfs_client
            except Exception:
                continue
        else:
            return _node2client[node]
    raise IOError("HDFS access error. Nodename: {}".format(namenode))


def _get_bucket_client():
    global _bucket_client
    if _bucket_client is None:
        _bucket_client = (
            get_current_resource_manager()
            and get_resource(BucketClient)
            or BucketClient()
        )
    return _bucket_client


@require_packages("hdfs")
def _split_path(
    path,
) -> Tuple[str, str, Optional[Union["hdfs.InsecureClient", BucketClient]]]:
    """Split path to 2 parts: head + uri.

    Head contains proto and host(port) for hdfs.
    URI is the unique path in the specified filesystem.

    Args:
        path: file path

    Returns:
        head: the head of path.
        URI: the unique path for the file
        fs_client: hdfs.InsecureClient or DMPClient, the fs client
    """

    path_uri = path.strip()
    if isinstance(path_uri, bytes):
        path_uri = path_uri.decode()
    head = ""
    uri = path
    fs_client = None
    if path_uri.startswith("hdfs://"):
        # hdfs
        parts = path_uri[7:].partition("/")
        head = "hdfs://" + parts[0]
        if parts[1] == "" or parts[2] == "":
            uri = "/"
        else:
            uri = parts[1] + parts[2]
            fs_client = _get_hdfs_client(parts[0])
    elif path_uri.startswith(DMP_PREFIX):
        """
        for dmpv2://matrix/my_tmp,
        result:   head: matrix  uri: /my_tmp
        """
        head, uri = split_dmp_url(path_uri)
        fs_client = _get_bucket_client()
    return head, uri, fs_client


def _hdfs_file_exist(client, path):
    return client.status(path, strict=False) is not None


def open_file(path, *args, **kwargs):
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        local_path = path
    elif isinstance(fs_client, BucketClient):
        local_path = fs_client.url_to_local(path)
        return BucketFileIO(io=open(local_path, *args, **kwargs), path=path)
    elif isinstance(fs_client, hdfs.InsecureClient):
        raise NotImplementedError("Not implemented for hdfs path")
    else:
        raise TypeError(f"Not supported head: {head}")
    return open(local_path, *args, **kwargs)


@contextmanager
def reader(
    path,
    encoding=None,
    delimiter=None,
    buffer_size=128 * 1024 * 1024,
    mode="r",
):
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        with open(uri, mode, encoding=encoding, newline=delimiter) as f:
            yield f
    elif isinstance(fs_client, hdfs.InsecureClient):
        with fs_client.read(
            uri,
            encoding=encoding,
            delimiter=delimiter,
            buffer_size=buffer_size,
        ) as f:
            yield f
    else:
        raise TypeError("Not supported head: {}".format(head))


def exists(path):
    """Whether path exists."""
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        return os.path.exists(uri)
    elif isinstance(fs_client, hdfs.InsecureClient):
        return _hdfs_file_exist(fs_client, uri)
    else:
        return _get_bucket_client().exists(path)


def normpath(path):
    """Normalize path, eliminating double slashes, etc."""
    if path.startswith("hdfs://"):
        return "hdfs://" + os.path.normpath(path[7:])
    elif path.startswith(DMP_PREFIX):
        return DMP_PREFIX + os.path.normpath(path[len(DMP_PREFIX) :])
    else:
        return os.path.normpath(path)


def makedirs(path, mode=0o777):
    if path[-1] == "/":
        path = path[:-1]
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        if not os.path.exists(uri):
            os.makedirs(uri, mode=mode, exist_ok=True)
    elif isinstance(fs_client, hdfs.InsecureClient):
        # os.makedirs will convert mode to octal representation
        # but hdfs.client.makedirs will not, so we have to convert
        # it here
        fs_client.makedirs(uri, permission=int(oct(mode)[2:]))
    else:
        _get_bucket_client().mkdir(path, recursive=True)


def bucket_delete_ex(
    clt: BucketClient,
    url: str,
    recursive: bool = False,
    split_num: int = 100,
) -> bool:
    if not recursive:
        return clt.delete(url, recursive=recursive)

    delete_file_num_thres = 1000

    fd = FileCountDict(clt).scan(url, recursive=True)
    deleted_folders = []
    for folder, file_cnt in fd.data.items():
        if file_cnt < delete_file_num_thres:
            deleted_folders.append(folder)

    # delete from the shallowest folder
    deleted_folders.reverse()
    for folder in deleted_folders:
        # note that depper folder may be deleted by previous delete
        if folder in fd.data:
            logger.info(f"try to delete {folder} It may takes a while.")
            fd.delete_cnt(folder)
            clt.delete(folder, recursive=True)
            logger.info(f"delete {folder} successfully.")

    while len(fd.data) > 0:
        folder, file_cnt = next(iter(fd.data.items()))
        scanner = FolderScan(clt)

        if file_cnt >= min(delete_file_num_thres, split_num):
            with tqdm(total=file_cnt) as pbar:
                file_list = []

                def delete_file(entry: BucketFileEntry):
                    file_list.append(entry.absolute_path)
                    if len(file_list) >= split_num:
                        clt.delete_files(file_list, split_num=split_num)
                        pbar.update(len(file_list))
                        file_list.clear()

                logger.info(
                    f"try to delete {folder} which contain {file_cnt} files."
                )
                scanner.scan(folder, file_callback=delete_file)
                if len(file_list) > 0:
                    clt.delete_files(file_list, split_num=split_num)
                    file_list.clear()
        else:
            logger.info(
                f"try to delete {folder} which contain {file_cnt} files."
            )
        # delete folder anyway.
        clt.delete(folder, recursive=True)
        fd.delete_cnt(folder)

    return True


def delete(path, recursive=False):
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        if recursive:
            try:
                shutil.rmtree(uri)
                return True
            except Exception:
                return False
        else:
            try:
                os.remove(uri)
                return True
            except Exception:
                return False
    elif isinstance(fs_client, hdfs.InsecureClient):
        try:
            return fs_client.delete(uri, recursive=recursive)
        except hdfs.HdfsError:
            return fs_client.delete(uri, recursive=False)
    else:
        return bucket_delete_ex(
            _get_bucket_client(), path, recursive=recursive, split_num=100
        )


def listdir(path, force_sync=False, force_sync_timeout=3600):
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        return os.listdir(uri)
    elif isinstance(fs_client, hdfs.InsecureClient):
        return fs_client.list(uri)
    else:
        return list(
            _get_bucket_client().ls_iterator(
                path,
                detail=False,
                full_path=False,
                force_sync=force_sync,
                force_sync_timeout=force_sync_timeout,
            )
        )  # noqa


def isdir(path: str) -> bool:
    """Is the target path a directory.

    Similar to os.path.isdir but works for hdfs and gpfs path.
    If path does not exist, this function return False.

    Args:
        path (str): The path

    Raises:
        Exception: If error happens.

    Returns:
        bool: Whether the target path is a directory.
    """
    head, uri, fs_client = _split_path(path)
    if fs_client is None:
        return os.path.isdir(uri)
    elif isinstance(fs_client, hdfs.InsecureClient):
        try:
            return fs_client.status(uri)["type"] == "DIRECTORY"
        except hdfs.util.HdfsError as e:
            if e.message.startswith("File does not exist"):
                return False
            else:
                raise e
    else:
        assert isinstance(fs_client, BucketClient)
        try:
            return fs_client.file_info(path).folder
        except FileNotFoundError:
            return False


def move(src: str, dst: str) -> bool:
    """Move file from src to dst.

    Args:
        src (str): The source path.
        dst (str): The destination path.

    Raises:
        NotImplementedError: Not implemented for hdfs path.

    Returns:
        bool: Whether the move is successful.
    """
    head_src, uri_src, fs_client_src = _split_path(src)
    head_dst, uri_dst, fs_client_dst = _split_path(dst)
    if fs_client_src is None:
        assert (
            fs_client_dst is None
        ), "Cannot move file between different file systems"
        return LocalFileSys.mv(src, dst)
    elif isinstance(fs_client_src, hdfs.InsecureClient):
        raise NotImplementedError("Not implemented for hdfs path")
    else:
        assert isinstance(fs_client_src, BucketClient)
        assert isinstance(fs_client_dst, BucketClient)
        assert AidiInfoCliFileSys.aidi_inf_cli_available()
        return AidiInfoCliFileSys.mv(src, dst)


class WorkingDir:
    def __init__(self, target: str):
        self._old = None
        self.target = target

    def __enter__(self):
        if self._old is not None:
            return self
        self._old = os.getcwd()
        os.chdir(self.target)

    def __exit__(self, ptype, value, trace):
        assert self._old is not None
        os.chdir(self._old)
        self._old = None
