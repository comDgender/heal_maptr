# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""The client to manipulate files in bucket.

Copied from `hatbc` repo and make modifications,
retaining only the URL path conversion functions.
originally from `hatbc.filestream.bucket.client`.
"""
import json
import logging
import os
import random
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Union,
)

import requests
from dataclasses_json import DataClassJsonMixin
from fsspec.utils import infer_storage_options
from tqdm import tqdm

try:
    import aidisdk
except ImportError:
    aidisdk = None

try:
    from requests_toolbelt.multipart.encoder import (
        MultipartEncoder,
        MultipartEncoderMonitor,
    )
except ImportError:
    MemoryError = None
    MultipartEncoderMonitor = None

from hat_engine.utils.apply_func import _as_list
from hat_engine.utils.http import wait_until_finish
from hat_engine.utils.package_helper import require_packages

__all__ = [
    "BucketClient",
    "BucketFileEntry",
    "FolderScan",
    "FileCountDict",
    "get_bucket_client",
    "url_to_local_path",
    "local_path_to_url",
    "url2ip",
    "get_bucket_mount_root",
    "parse_url",
    "split_dmp_url",
    "is_dmp_url",
    "check_is_dmp_url",
]


logger = logging.getLogger(__name__)

DMP_PREFIX = "dmpv2://"
HTTPS_PREFIX = "https://"
GFPS_PREFIX = "/gpfs/"
HTTPS_PATH = "https://us-data.hobot.cc/"

GPFS_MKDIR_API = "%s/dmp/v1/buckets/%s/files:mkdir"
GPFS_MV_API = "%s/dmp/v1/buckets/files:mv"
GPFS_UPLOAD_FILE_API = "%s/dmpsvc/http-dmp-upload/upload/%s"
GPFS_DOWNLOAD_FILE_API = "%s/dmpsvc/http-dmp-download/download/%s"
GPFS_DELETE_FILE_API = "%s/dmp/v1/buckets/%s/files:batchDelete"
GPFS_FILE_INFO_API = "%s/dmp/v1/buckets/%s/files/detail"
GPFS_EXIST_API = "%s/dmp/v1/buckets/%s/files/isExist"
GPFS_LIST_API = "%s/dmp/v1/buckets/%s/files"
GPFS_FORCE_SYNC_API = "%s/dmp/v1/buckets/%s/files:force_sync"
GPFS_GET_SYNC_TASK_API = "%s/dmp/v1/buckets/%s/files:get-sync-task"
GPFS_RECALL_FROM_TAPE_API = "%s/dmp/v1//buckets/%s/recall"
GPFS_GET_RECALL_FROM_TAPE_TASK_API = "%s/dmp/v1/buckets/tape-task/%s"


B_PER_MB = 2 ** 20


class KThread(threading.Thread):
    """A subclass of threading.Thread, with a kill() method.

    Come from:

    Kill a thread in Python:

    http://mail.python.org/pipermail/python-list/2004-May/260937.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.killed = False
        self._run_backup = None

    def start(self):
        """Start the thread."""
        self._run_backup = self.run
        self.run = self._run_with_trace
        super().start()

    def _run_with_trace(self):
        """Install the trace hook before executing the original run."""
        sys.settrace(self._trace_calls)
        try:
            self._run_backup()
        finally:
            self.run = self._run_backup

    def _trace_calls(self, frame, event, arg):
        if event == "call":
            return self._trace_lines
        return None

    def _trace_lines(self, frame, event, arg):
        if self.killed and event == "line":
            raise SystemExit()
        return self._trace_lines

    def kill(self):
        self.killed = True


def timeout_call(
    seconds: float,
    func: Callable,
    args=tuple(),  # noqa C408
    kwargs: Dict = None,
    message: str = None,
):  # noqa D400
    """Call a function and raise a TimeoutError when executing time
        exceed timeout duration.

    Args:
        seconds: In seconds.
        func : Function to exec.
        args : Function args, by default tuple().
        kwargs : Function kwargs, by default None.
        message : TimeoutError message, by default None.
    """

    if message is None:
        message = f"Calling {func} take time exceed {seconds}s"

    if kwargs is None:
        kwargs = dict()  # noqa C408

    result = []
    exception = None

    def _new_func(oldfunc, result, oldfunc_args, oldfunc_kwargs):
        try:
            result.append(oldfunc(*oldfunc_args, **oldfunc_kwargs))
        except BaseException as e:
            nonlocal exception
            exception = e

    # create new args for _new_func, because we want to get the func return val to result list  # noqa
    new_kwargs = {
        "oldfunc": func,
        "result": result,
        "oldfunc_args": args,
        "oldfunc_kwargs": kwargs,
    }

    thd = KThread(target=_new_func, args=(), kwargs=new_kwargs)

    thd.start()

    thd.join(seconds)

    alive = thd.is_alive()

    thd.kill()  # kill the child thread

    if alive:
        raise TimeoutError(message)
    elif exception is not None:
        raise exception
    else:
        return result[0]


def _parse_mount_root(msg, check_access=True):
    mount_keys = [
        "hogpu.cc",
        "JuiceFS",
        "gpfs/plat_dmp",
        "gpfs2/plat_dmp",
        "gpfs-test/plat_dmp",
        "gpfs",
    ]

    bucket2root = dict()  # noqa C408

    for result_i in msg.strip().split("\n"):
        parts = result_i.strip().split()
        if len(parts) >= 6:
            device = parts[0]
            mount_point = parts[2]
            for k in mount_keys:
                if k in device:
                    bucket_name = mount_point.split("/")[-1]
                    if check_access:
                        if os.access(mount_point, os.R_OK):
                            bucket2root[bucket_name] = mount_point
                    else:
                        bucket2root[bucket_name] = mount_point

    return bucket2root


def is_running_on_aidi():
    """Whether is running on aidi."""
    return (
        os.environ.get("WORKING_PATH", None) is not None
        and os.environ.get("PBS_JOBNAME", None) is not None
    )


def get_bucket_mount_root() -> Dict[str, str]:
    if is_running_on_aidi():
        bucket2root = dict()  # noqa C408
        prefix = "bucket_"
        for key_i, val_i in os.environ.items():
            if key_i.startswith(prefix):
                bucket2root[key_i[len(prefix) :]] = val_i
        return bucket2root

    else:
        results = subprocess.run(["mount"], capture_output=True, text=True)
        results = results.stdout
        return _parse_mount_root(results)


def is_dmp_url(url: str) -> bool:
    return url.startswith(DMP_PREFIX)


def is_https_url(url: str) -> bool:
    return url.startswith(HTTPS_PREFIX)


def _request_stop_cond(resp):
    return resp.status_code == requests.codes.ok


def check_is_dmp_url(url: str):
    if not is_dmp_url(url):
        raise Exception(f"{url} is not a dmp url")


def split_dmp_url(url: str) -> Tuple[str, str]:
    check_is_dmp_url(url)
    dmp_path = url[len(DMP_PREFIX) :]
    bucket_name_pos = dmp_path.find("/")
    bucket_name = dmp_path[:bucket_name_pos]
    bucket_file_path = dmp_path[bucket_name_pos:]
    return bucket_name, bucket_file_path


@require_packages("aidisdk")
def get_user_token():
    client = aidisdk.AIDIClient()
    return client.session.token


_client = None


def get_bucket_client():
    global _client
    if _client is not None:
        return _client
    _client = BucketClient()
    return _client


class NotValidLocalPathError(FileNotFoundError):
    """This error is raised when the input path is not a valid local path."""


@dataclass
class FileInfo(DataClassJsonMixin):
    bucket_name: str
    name: str
    type: str
    size: int
    path: str
    sha256: str
    description: str
    update_at: str
    update_user: str
    folder: bool
    total_file_size: int
    total_file_count: int
    file_id: str
    create_user: str
    # whether data is stored in tape.. It should be bool value
    tape: Union[bool, int]

    def __post_init__(self):
        # hotfix to make tape into bool
        if isinstance(self.tape, int):
            self.tape = True if self.tape == 1 else False


@dataclass
class BucketFileEntry:
    bucket_name: str
    name: str
    type: str
    size: str  # should be int. AIDI bug.
    path: str
    sha256: str
    depth: int
    description: str
    update_at: str
    update_user: str
    folder: bool
    total_file_size: str  # should be int. AIDI bug.
    total_file_count: str  # should be int. AIDI bug.
    file_id: str
    create_user: str
    tape: int

    @property
    def absolute_path(self) -> str:
        return (
            "dmpv2://" + self.bucket_name + os.path.join(self.path, self.name)
        )


def _download_to_file(response, target_path, with_progress=True):
    chunk_size = 4 * B_PER_MB
    if with_progress:
        pbar = tqdm(desc=target_path, unit="MB")
    else:
        pbar = None
    with open(target_path, "wb") as fwrite:
        for chunk in response.iter_content(chunk_size=chunk_size):
            fwrite.write(chunk)
            chunk_mb = len(chunk) / float(B_PER_MB)
            if pbar is not None:
                pbar.update(chunk_mb)
    if pbar is not None:
        pbar.close()


class FolderScan:
    def __init__(self, clt: "BucketClient"):
        self._clt: BucketClient = clt

    def scan(
        self,
        url: str,
        file_callback: Callable[[BucketFileEntry], Any],
        recursive: bool = False,
    ):
        assert self._clt.isfile(url) is False
        for p in self._clt.ls_iterator(url, detail=True, penetrate=True):
            p = BucketFileEntry(**p)
            file_callback(p)
            if p.folder and recursive:
                self.scan(p.absolute_path, file_callback)


class FileCountDict:
    def __init__(self, clt: "BucketClient"):
        self._clt: BucketClient = clt
        self._dict: OrderedDict[str, int] = OrderedDict()

    @property
    def data(self) -> OrderedDict[str, int]:
        return self._dict

    def delete_cnt(self, url: str):
        url = self._strip_folder_path(url)
        if url not in self.data:
            raise KeyError("url is not in dict.")
        cnt = self.data[url]
        to_delete_keys = []
        for k in self.data:
            # if k is subfolder or k is url
            if k.startswith(url):
                to_delete_keys.append(k)
            # if k is parent folder
            elif url.startswith(k):
                self.data[k] -= cnt
        for k in to_delete_keys:
            self.data.pop(k)

    def _strip_folder_path(self, url: str) -> str:
        if url[-1] == "/":
            url = url[:-1]
        return url

    def scan(self, url: str, recursive: bool = False) -> "FileCountDict":
        if self._clt.exists(url) is False:
            raise FileNotFoundError(f"{url} not found.")
        # make sure url exist and is a folder
        self._dict = self._get_file_cnt(url, recursive=recursive)
        return self

    def _get_file_cnt(
        self, url: str, recursive: bool = False
    ) -> OrderedDict[str, int]:
        url = self._strip_folder_path(url)
        if not recursive:
            # include self
            return OrderedDict({url: self._clt.ls_count(url) + 1})

        def recursive_ls_iter(url, res: Dict[str, int]) -> int:
            # including self
            logger.info(f"scan {url}")
            cnt = 1
            for p in self._clt.ls_iterator(url, detail=True, penetrate=True):
                p = BucketFileEntry(**p)
                if p.folder:
                    cnt += recursive_ls_iter(p.absolute_path, res)
                else:
                    cnt += 1
            if url not in res:
                res[url] = 0
            res[url] += cnt
            return cnt

        ret = {}
        ret[url] = recursive_ls_iter(url, ret)
        return OrderedDict(sorted(ret.items(), key=lambda x: x[1]))


def parse_url(url: str) -> Dict:
    """Get protocol and path from url-like string.

    Args:
        url (str): url in format

    Returns:
        Dict: dictionary with keys 'protocol' and 'path' and other options.
    """
    if not url:
        return None

    # Add more protocols here...
    if url.startswith("dmpv2://"):
        client = get_bucket_client()
        path = client.url_to_local(url)
        options = infer_storage_options(path)
    else:
        options = infer_storage_options(url)
    return options


def url_to_local_path(
    path: str,
    bucket_client: "BucketClient" = None,
    *,
    allow_invalid_bucket_url: bool = True,
):
    """Change dmp url path to local path.

    Args:
        path: dmp url path
        bucket_client: bucket client, default None
        allow_invalid_bucket_url: whether allow invalid bucket url, default True # noqa

    Returns:
        path: local path.
    """
    # get bucket client
    bucket_client = get_bucket_client()
    # judge the validity of input path
    if bucket_client.valid_url(path):
        return bucket_client.url_to_local(path)
    elif allow_invalid_bucket_url:
        return path
    else:
        raise ValueError(f"Invalid dmp url: {path}")


def local_path_to_url(
    path: str,
    bucket_client: "BucketClient" = None,
    *,
    allow_invalid: bool = False,
):
    """Change local path to url path.

    Args:
        path: local path
        bucket_client: bucket client, default None
        allow_invalid: whether allow invalid local path, default False # noqa

    Returns:
        path: dmp url path.
    """
    # get bucket client
    bucket_client = get_bucket_client()
    # judge the validity of input path
    if not bucket_client.valid_url(path):
        try:
            path = bucket_client.local_to_url(path)
        except FileNotFoundError:
            if allow_invalid:
                return path
            else:
                raise ValueError(f"Invalid local url: {path}")
    return path


def url2ip(urlfile, outfile):
    with open(urlfile, "r") as fread, open(outfile, "w") as fwrite:  # noqa
        for oneurl in fread.readlines():
            if not oneurl.strip():
                continue
            url = str(oneurl.strip())
            try:
                ip = socket.gethostbyname(url)
                fwrite.write(str(ip) + "\n")
            except Exception as e:
                print(f"This {url} to ip failed, error msg {e}")


def call_and_retry(
    func,
    args=None,
    kwargs=None,
    stop_cond=None,
    catch_exception=(Exception,),
    raise_catch_exception_when_failed=True,
    max_retry=0,
    retry_interval=1,
):
    """
    Call function with retry.

    Args:
        func : callable
            Function
        args : tuple/list, optional
            Function input args, by default None
        kwargs : dict, optional
            Function kwargs, by default None
        stop_cond : callable, optional
            The way to parse function outputs, return True or False, by default None.
            This function is called in the following way:

            .. code-block:: python

                ret = func(*args, **kwargs)
                flag = stop_cond(ret)
                if flag:
                    return (True, ret)
                else:
                    # retry
                    pass


        catch_exception : tuple, optional
            Handled exceptions, by default ('Exception', )
        raise_catch_exception_when_failed : bool, optional
            Whether to raise exception when retry failed, by default True
        max_retry : int, optional
            The maximum number of retry, by default 0
        retry_interval : int or tuple, optional
            Retry interval, by default 1

    Returns:
        The first one indicates the function is executed success or not, the
        second is the outputs.
    """  # noqa
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if stop_cond is not None:
        assert callable(stop_cond)
    assert isinstance(catch_exception, (list, tuple))
    assert max_retry >= 0

    if isinstance(retry_interval, (int, float)):
        retry_interval = [retry_interval, retry_interval]
    assert isinstance(retry_interval, (list, tuple))
    assert len(retry_interval) == 2
    assert retry_interval[1] >= retry_interval[0] and retry_interval[0] >= 0

    retry_idx = 0

    while retry_idx <= max_retry:
        exception = None
        try:
            ret = func(*args, **kwargs)
        except Exception as e:
            if not catch_exception or not isinstance(
                e, catch_exception
            ):  # noqa
                raise
            exception = e

        if exception is not None:
            logger.warning(
                f"Call {func} with args={args}, kwargs={kwargs} failed! With exception = {exception}! Retry {retry_idx} times"  # noqa
            )  # noqa
            # raise
            if raise_catch_exception_when_failed and retry_idx >= max_retry:
                raise exception
            else:
                outputs = (False, exception)
        elif stop_cond is not None and not stop_cond(ret):
            logger.warning(
                f"Call {func} with args={args}, kwargs={kwargs} not success! Outputs = {ret}! Retry {retry_idx} times"  # noqa
            )  # noqa
            outputs = (False, ret)
        else:
            outputs = (True, ret)
            break

        time.sleep(random.uniform(retry_interval[0], retry_interval[1]))
        retry_idx += 1

    return outputs


class BucketClient(object):
    def __init__(
        self,
        token=None,
        host="http://model.aidi.hobot.cc",
        max_retry=5,
        check_mounted_buckets_visiable=True,
    ):
        self._token = token
        self._host = host
        self._max_retry = max_retry
        self.bucket2root = get_bucket_mount_root()

        self.check_mounted_buckets_visiable = check_mounted_buckets_visiable
        self.checked_buckets = set()

        if not self._host.startswith("http"):
            self._host = "http://" + self._host

        self._session = requests.Session()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["bucket2root"] = None
        state["checked_buckets"] = set()
        state["_session"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        # NOTE: reset this value because the local and remote value
        # is different
        self.bucket2root = get_bucket_mount_root()
        self._session = requests.Session()

    def _check_bucket(self, bucket_name):
        """Check whether the bucket can be visited or not.

        Sometimes, bucket may be blocked, for the first time of visiting,
        check whether can be visited or not.
        """
        if bucket_name in self.checked_buckets:
            return
        assert (
            bucket_name in self.bucket2root
        ), f"{bucket_name} does not mount!"  # noqa
        bucket_root = self.bucket2root[bucket_name]
        try:
            timeout_call(120, os.listdir, (bucket_root,))
        except TimeoutError:
            raise OSError(
                f"Cannot visit the mounted bucket {bucket_name}, exceed 120 seconds"  # noqa
            )  # noqa
        self.checked_buckets.add(bucket_name)

    def __del__(self):
        if hasattr(self, "_session") and self._session is not None:
            self._session.close()
            self._session = None

    def valid_url(self, url: str) -> bool:
        return url.startswith(DMP_PREFIX) or url.startswith(HTTPS_PREFIX)

    def mount_buckets(self):
        return list(self.bucket2root.keys())

    def get_mount_root(self, bucket_name) -> str:
        if bucket_name not in self.bucket2root:
            raise FileNotFoundError(f"bucket not mounted: {bucket_name}")
        if self.check_mounted_buckets_visiable:
            self._check_bucket(bucket_name)
        return self.bucket2root[bucket_name]

    def url_to_local(
        self,
        url: str,
        root_only: Optional[bool] = False,
        check_visiable: Optional[bool] = True,
    ) -> str:
        if is_https_url(url):
            local_root = url.replace(HTTPS_PATH, GFPS_PREFIX)
            return local_root
        else:
            bucket_name, bucket_file_path = split_dmp_url(url)
            if check_visiable:
                local_root = self.get_mount_root(bucket_name)
            else:
                local_root = "/horizon-bucket/{}".format(bucket_name)
            return local_root if root_only else local_root + bucket_file_path

    def local_to_url(self, path: str) -> str:
        path = os.path.abspath(path)
        if path.startswith(GFPS_PREFIX):
            return path.replace(GFPS_PREFIX, HTTPS_PATH)
        for bucket_name in self.bucket2root:
            local_root = self.bucket2root[bucket_name]
            if path.startswith(local_root):
                path = f"{DMP_PREFIX}{bucket_name}{path[len(local_root):]}"
                return path
        else:
            raise NotValidLocalPathError(
                "Input path not valid local bucket path: %s" % path
            )

    def mkdir(self, url, recursive=False):
        bucket_name, bucket_file_path = split_dmp_url(url)
        data = {"directories": [bucket_file_path], "recursive": recursive}
        flag, resp = self._post(
            url=self._get_api("mkdir", bucket_name), data=json.dumps(data)
        )
        if flag and not resp.json():
            return True
        else:
            raise OSError(resp.json())

    def exists(self, url: str) -> bool:
        bucket_name, bucket_file_path = split_dmp_url(url)
        flag, resp = self._get(
            url=self._get_api("file_exist", bucket_name),
            params={"path": bucket_file_path},
        )
        if flag:
            return resp.content == b"true"
        else:
            raise OSError(resp.text)

    def file_info(self, url: str):
        if not self.exists(url):
            raise FileNotFoundError(f"{url} does not exists")
        bucket_name, bucket_file_path = split_dmp_url(url)
        flag, resp = self._get(
            url=self._get_api("file_info", bucket_name),
            params={"path": bucket_file_path},
        )
        if flag:
            ret = FileInfo.from_json(resp.content)
            return ret
        else:
            raise OSError(resp)

    def isfile(self, url: str) -> bool:
        if not self.exists(url):
            return False
        file_info = self.file_info(url)
        return file_info is not None and file_info.folder is False

    def ls_iterator(
        self,
        url: str,
        detail=True,
        full_path=False,
        force_sync=False,
        force_sync_timeout=3600,
        penetrate: bool = True,
    ):  # noqa
        if force_sync:
            self.force_sync(url, wait=True, timeout=force_sync_timeout)
        bucket_name, bucket_file_path = split_dmp_url(url)

        if self.isfile(url):
            raise FileNotFoundError("{} is not a folder".format(url))

        if not self.exists(url):
            return iter([])

        def list_file_iter(penetrate: bool = True):
            params = {
                "path": bucket_file_path,
                "penetrate": penetrate,
                "page_token": "page_number=1",
                "page_size": 100,
            }
            page_num = 1
            visit_idx = 0
            while True:
                params["page_token"] = "page_number={}".format(page_num)
                flag, resp = self._get(
                    url=self._get_api("list", bucket_name), params=params
                )
                content = None
                page_num += 1
                if flag:
                    content = resp.json()
                    for f in content["files"]:
                        if not detail:
                            if full_path:
                                f = "{}/{}".format(url, f["name"])
                            else:
                                f = f["name"]
                        visit_idx += 1
                        yield f
                    if visit_idx >= int(content["total_count"]):
                        break
                else:
                    break

        return list_file_iter(penetrate=penetrate)

    def ls_count(self, url: str):
        bucket_name, bucket_file_path = split_dmp_url(url)
        if not self.exists(url):
            return 0
        if self.isfile(url):
            raise NotADirectoryError("{} is not a folder".format(url))
        params = {
            "path": bucket_file_path,
            "penetrate": False,
            "page_token": "page_number=1",
        }
        flag, resp = self._get(
            url=self._get_api("list", bucket_name), params=params
        )
        if flag:
            content = json.loads(resp.content)
            return int(content["total_count"])
        else:
            return 0

    def download(
        self,
        url,
        target,
        offset=None,
        length=None,
        stream=True,
        with_progress=True,
        overwrite=False,
    ):
        if os.path.exists(target):
            if overwrite:
                os.remove(target)
            else:
                raise FileExistsError(f"File {target} already exist!")
        bucket_name, bucket_file_path = split_dmp_url(url)
        flag, resp = self._get(
            url=self._get_api("download_file", bucket_name) + bucket_file_path,
            params={"offset": offset, "length": length},
            stream=stream,
        )
        if not flag:
            raise OSError(resp)
        else:
            _download_to_file(resp, target, with_progress=with_progress)
            return True

    @require_packages("requests_toolbelt")
    def upload(
        self, local_file, url, with_progress=True, timeout=None, overwrite=True
    ):
        if not overwrite:
            raise NotImplementedError("overwrite=False is not supported")

        bucket_name, bucket_file_path = split_dmp_url(url)
        fd, f = os.path.split(bucket_file_path)
        url = self._get_api("upload_file", bucket_name) + fd

        from hat_engine.utils.filestream.io import reader

        def create_callback(encoder: MultipartEncoder, bar_obj: tqdm):
            encoder_len = encoder.len
            bar_obj.total = encoder_len / float(B_PER_MB)

            def callback(monitor):
                read_size = monitor.bytes_read / float(B_PER_MB)
                bar_obj.update(read_size - bar_obj.n)

            return callback

        def _impl():
            with reader(local_file, mode="rb") as fin:
                payload = {
                    "file": (
                        os.path.basename(f),
                        fin,
                        "application/octet-stream",
                    )
                }
                monitor = MultipartEncoder(payload)
                bar_object = None
                if with_progress:
                    bar_format = (
                        "{desc}:{percentage:.2f}%|{bar}|"
                        + "{n:.3f}/{total:.3f}"
                        + "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                    )
                    bar_object = tqdm(
                        desc=local_file,
                        unit="MB",
                        total=0,
                        bar_format=bar_format,
                    )
                    callback = create_callback(monitor, bar_object)
                    monitor = MultipartEncoderMonitor(monitor, callback)
                flag, resp = self._post(
                    url=url,
                    data=monitor,
                    content_type=monitor.content_type,
                    allow_redirects=True,
                    max_retry=0,
                )
                if bar_object is not None:
                    bar_object.close()

            if flag:
                return True
            else:
                raise OSError(resp)

        flag, msg = call_and_retry(
            timeout_call,
            args=(timeout, _impl),
            kwargs={
                "message": "Upload {} to {} exceed {}s".format(
                    local_file, url, timeout
                ),
            },
            max_retry=self._max_retry,
        )
        if flag:
            return True
        raise OSError(msg)

    def delete(self, url, recursive=False):
        if not self.exists(url):
            return True
        if not self.isfile(url) and not recursive:
            msg = "To delete a folder, you need to set recursive flag"
            raise IsADirectoryError(msg)
        bucket_name, bucket_file_path = split_dmp_url(url)
        flag, resp = self._post(
            url=self._get_api("delete_file", bucket_name),
            data=json.dumps(
                {"files": [bucket_file_path], "recursive": recursive}
            ),
        )
        if flag and not resp.json():
            return True
        else:
            raise OSError(resp.json())

    def delete_files(self, url_list: List[str], split_num=1000):
        """Delete files in batch.

        Args:
            url_list (List[str]): The list of urls to be deleted.
            split_num (int, optional): The number of files to be deleted in
                one request. Defaults to 1000.

        Raises:
            OSError: If the deletion failed.
        """
        bucket_files = {}
        for url in url_list:
            bucket_name, bucket_file_path = split_dmp_url(url)
            if bucket_name not in bucket_files:
                bucket_files[bucket_name] = []
            bucket_files[bucket_name].append(bucket_file_path)

        for bucket_name, bucket_file_path_list in bucket_files.items():
            file_n = len(bucket_file_path_list)
            for i in range(0, file_n, split_num):
                flag, resp = self._post(
                    url=self._get_api("delete_file", bucket_name),
                    data=json.dumps(
                        {
                            "files": bucket_file_path_list[
                                i : min(i + split_num, file_n)
                            ],
                            "recursive": False,
                        }
                    ),
                )
                if flag and not resp.json():
                    continue
                else:
                    raise OSError(resp.json())

    def force_sync(self, url, wait=True, timeout=3600):
        bucket_name, bucket_file_path = split_dmp_url(url)

        flag, resp = self._post(
            url=self._get_api("force_sync", bucket_name),
            data=json.dumps({"path": bucket_file_path}),
        )

        if flag:
            sync_task_id = resp.json()["id"]
        else:
            msg = resp.json()

            # compatible with new API
            if "already exist" in msg["error"]:
                sync_task_id = msg["error"].split(" ")[2]
            else:
                raise RuntimeError(f"Sync failed, msg = {msg}")

        if wait:

            def query_func():
                status = self.get_sync_task_status(
                    bucket_name, sync_task_id
                )  # noqa
                return status in ["FAILED", "COMPLETED"]

            wait_until_finish(
                query_func,
                timeout=timeout,
                wait_what=f"GPFS force_sync {url}, task_id {sync_task_id}",
            )  # noqa
            return (
                self.get_sync_task_status(bucket_name, sync_task_id)
                == "COMPLETED"
            )  # noqa
        else:
            return sync_task_id

    def get_sync_task_status(self, bucket_name, task_id):
        flag, resp = self._get(
            url=self._get_api("get_sync_task", bucket_name),
            params={"task_id": task_id},
        )
        if flag:
            return resp.json()["status"]
        else:
            raise RuntimeError(
                f"Get sync task status failed, msg = {resp.json()}"
            )  # noqa

    def recall_from_tape(
        self,
        paths: str,
        priority: int = 3,
        wait: bool = False,
        timeout: float = 3600 * 24,
    ):
        """Recall files from tape.

        Args:
        paths: All paths should be in the same bucket.
        priority: Task priority, possible values are {1, 2, 3, 4, 5},
            by default 3
        wait: Whether wait until recall task finished, by default False
        timeout: Maximuum waiting time, by default 3600 * 24 (24 hours).
        """
        paths = _as_list(paths)
        assert all(
            [self.valid_url(path_i) for path_i in paths]
        ), f"All paths should starts with {DMP_PREFIX}, but get {paths}"

        split_bucket_name = None
        file_paths = []
        for path_i in _as_list(paths):
            bucket_name, bucket_file_path = split_dmp_url(path_i)
            if split_bucket_name is None:
                split_bucket_name = bucket_name
            else:
                assert (
                    bucket_name == split_bucket_name
                ), f"all paths should in the same bucket, but get {bucket_name} vs. {split_bucket_name}"  # noqa
            file_paths.append(bucket_file_path)

        assert file_paths, f"paths = {paths} is not allowed to be empty"

        flag, resp = self._post(
            url=self._get_api("recall_from_tape", split_bucket_name),
            data=json.dumps({"file_list": file_paths, "priority": priority}),
        )
        ret = resp.json()

        if flag:
            task_id = ret["content"]["job_id"]

            if task_id == 0:
                raise RuntimeError(f"Recall from tape failed, msg = {ret}")

            if wait:

                def _query_func():
                    status = self.get_recall_from_tape_task_status(
                        task_id=task_id
                    )  # noqa
                    return status in ["completed", "failed", "cancaled"]

                wait_until_finish(_query_func, timeout=timeout)
                return (
                    self.get_recall_from_tape_task_status(task_id=task_id)
                    == "completed"
                )  # noqa

            else:
                return task_id
        else:
            raise RuntimeError(f"Recall from tape failed, msg = {ret}")

    def get_recall_from_tape_task_status(self, task_id: int):
        """
        Get recall from tape task status.

        Args:
            task_id: Task id
        """
        flag, resp = self._get(
            url=self._get_api("get_recall_from_tape_task", task_id),
            params=None,
        )

        if flag:
            ret = resp.json()["content"]
            assert len(ret) == 1, ret
            return ret[0]["status"]
        else:
            raise RuntimeError(
                f"Geet recall from tape task status failed, msg = {resp.json()}"  # noqa
            )  # noqa

    # ----- Internal functions -----

    def _get(
        self,
        url,
        params,
        allow_redirects=False,
        content_type="application/json",
        stop_cond=_request_stop_cond,
        max_retry=None,
        **kwargs,
    ):
        if self._token is None:
            _token = get_user_token()
        headers = {
            "Content-Type": content_type,
            "X-Forwarded-User": _token,
        }
        max_retry = self._max_retry if max_retry is None else max_retry
        flag, resp = call_and_retry(
            self._session.get,
            kwargs=dict(
                url=url,
                params=params,
                headers=headers,
                allow_redirects=allow_redirects,
                **kwargs,
            ),
            max_retry=max_retry,
            stop_cond=stop_cond,
        )
        return flag, resp

    def _request_stop_cond(self, resp):
        return resp.status_code == requests.codes.ok or resp.json().get(
            "code"
        ) in [
            # 8,  # 8: "out of file num limit[7000] or time out[500s]."
        ]

    def _post(
        self,
        url,
        data,
        allow_redirects=False,
        content_type="application/json",
        stop_cond=None,
        max_retry=None,
        **kwargs,
    ):
        if self._token is None:
            _token = get_user_token()
        headers = {
            "Content-Type": content_type,
            "X-Forwarded-User": _token,
        }
        if stop_cond is None:
            stop_cond = self._request_stop_cond

        max_retry = self._max_retry if max_retry is None else max_retry
        flag, resp = call_and_retry(
            self._session.post,
            kwargs=dict(
                url=url,
                data=data,
                headers=headers,
                allow_redirects=allow_redirects,
                **kwargs,
            ),
            max_retry=max_retry,
            stop_cond=stop_cond,
        )
        return flag, resp

    def _get_api(self, api_name, bucket_name):
        if api_name == "mkdir":
            return GPFS_MKDIR_API % (self._host, bucket_name)
        elif api_name == "move_file":
            return GPFS_MV_API % (self._host)
        elif api_name == "upload_file":
            return GPFS_UPLOAD_FILE_API % (self._host, bucket_name)
        elif api_name == "download_file":
            return GPFS_DOWNLOAD_FILE_API % (self._host, bucket_name)
        elif api_name == "delete_file":
            return GPFS_DELETE_FILE_API % (self._host, bucket_name)
        elif api_name == "file_info":
            return GPFS_FILE_INFO_API % (self._host, bucket_name)
        elif api_name == "file_exist":
            return GPFS_EXIST_API % (self._host, bucket_name)
        elif api_name == "list":
            return GPFS_LIST_API % (self._host, bucket_name)
        elif api_name == "force_sync":
            return GPFS_FORCE_SYNC_API % (self._host, bucket_name)
        elif api_name == "get_sync_task":
            return GPFS_GET_SYNC_TASK_API % (self._host, bucket_name)
        elif api_name == "recall_from_tape":
            return GPFS_RECALL_FROM_TAPE_API % (self._host, bucket_name)
        elif api_name == "get_recall_from_tape_task":
            return GPFS_GET_RECALL_FROM_TAPE_TASK_API % (
                self._host,
                bucket_name,
            )  # noqa
        else:
            raise ValueError(f"Unsupported api {api_name}")
