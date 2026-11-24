# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import hashlib
import logging
import os
import re

from fsspec.implementations.local import LocalFileSystem

from hat_engine.utils.logger import MSGColor, format_msg
from .filesystem import get_filesystem, split_file_path

logger = logging.getLogger(__name__)

__all__ = [
    "regex_search",
    "calculate_sha256",
    "check_sha256",
    "generate_sha256_file",
    "get_hash_file_if_hashed_and_local",
    "remove_file_by_regex",
]


def regex_search(in_str: str, pattern: str):
    """Find the first match object in the string that can \
        match the regular expression pattern.

    Args:
        in_str: Input str.
        pattern: Regular expression pattern.

    Returns:
        Regular expression pattern match result.
    """

    assert pattern is not None, "`pattern` can not be None."
    prog = re.compile(pattern)
    r = prog.search(in_str)
    ret = r.group(1) if r else None
    return ret


def calculate_sha256(file_path: str, chunk_size: int = 1024 * 1024):
    """Generate SHA256 checksum of input file.

    Args:
        file_path: Input file.
        chunk_size: Size of per block for hashlib to calculate hash.
                    Defaults to 1024*1024.

    Returns:
        SHA256 checksum hash value.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 1M
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    sha256 = sha256_hash.hexdigest()

    return sha256


def check_sha256(in_file: str, sha256: str):
    """Check SHA256 checksum between input file and input sha256 value.

    Args:
        in_file: Input file.
        sha256 (str): Input sha256 hash value.

    Returns:
        True or False.
    """
    sha = calculate_sha256(in_file)

    assert len(sha) >= len(sha256), (
        f"The len of `hash_value` {sha256} should not "
        f"be greater than {len(sha)}"
    )

    return sha[: len(sha256)] == sha256


def remove_file_by_regex(dir_path, regex):
    for name in os.listdir(dir_path):
        reg_file = regex_search(name, pattern=regex)
        if reg_file:
            try:
                os.remove(os.path.join(dir_path, name))
            except Exception as e:
                logger.warning(
                    "1. Make sure there are not two or more jobs, which "
                    "have the same path for saving models; 2. Make sure "
                    "you have the permission to manipulate the folder "
                    "where the models are saved."
                )
                logger.warning(str(e))


def generate_sha256_file(
    in_file: str,
    remove_old: bool = False,
):
    """Generate a file with sha256 hash value.

    Examples:
        Suppose the sha256 hash value of in_file is 1d3765fc...:
        >>> in_file = "/tmp_dir/float-checkpoint-0026.pth.tar"
        >>> ret_file = generate_sha256_file(in_file)
        >>> assert str(
        ...    ret_file) == "/tmp_dir/float-checkpoint-0026-1d3765fc.pth.tar"

        Suppose the sha256 hash value of new_in_file is 234567c...:
        >>> new_in_file = "/tmp_dir/float-checkpoint-0026.pth.tar"
        >>> new_ret_file = generate_sha256_file(new_in_file, remove=True)
        >>> assert not os.path.join(ret_file)
        >>> assert str(
        ...    new_ret_file) == "/tmp_dir/float-checkpoint-0026-234567c.pth.tar"  # noqa E501

    Args:
        in_file: File path.
        remove_old: Whether to delete files with the same prefix.
                    Defaults to False.

    Returns:
        Path of hashed file.
    """

    assert os.path.exists(in_file), f"File {in_file} do not exist"

    ext_list = (".pth.tar", ".pt", ".hbm")
    dir_path, file_name, file_ext = split_file_path(in_file)

    assert (
        file_ext in ext_list
    ), f"The file extention should be in {ext_list}, but get {file_ext}"

    prefix = file_name[: -len(file_ext)]
    regex = "%s-([a-f0-9]*)\\." % prefix

    # remove hashed files with the same prefix as in_file
    if remove_old:
        remove_file_by_regex(dir_path, regex)

    # check in_file:
    # if the name of in_file contains hash value,
    # and the hash value is correct, return in_file directly.
    ret = regex_search(file_name, pattern=regex)
    if ret and check_sha256(in_file, ret):
        return in_file

    sha = calculate_sha256(in_file)

    if in_file.endswith(file_ext):
        out_file_name = in_file[: -len(file_ext)]
    else:
        out_file_name = in_file
    final_file = out_file_name + f"-{sha[:8]}" + file_ext

    if "-last" in prefix or "-best" in prefix:
        # shutil copyfile is not recommended on ali cluster
        os.system(f"cp {in_file} {final_file}")
    else:
        try:
            os.replace(in_file, final_file)
        except Exception as e:
            logger.warning(
                "1. Make sure there are not two or more jobs, which have "
                "the same path for saving models; 2. Make sure you have the "
                "permission to manipulate the folder where the models "
                "are saved."
            )
            logger.warning(str(e))

    return final_file


# TODO (mengyang.duan): remove this function next version.
def get_hash_file_if_hashed_and_local(in_file: str, check_hash: bool = True):
    """Get file with hash value according to the prefix of the input file. \
    If input file is in `hdfs` or `http`, do nothing.

    Args:
        in_file (str): Input file path.
        check_hash (bool): Whether to check the file hash.

    Returns:
        Final file path.
    """

    in_file = str(in_file)
    fs = get_filesystem(in_file)
    if not isinstance(fs, LocalFileSystem):
        # hdfs or http
        return in_file

    dir_path, file_name, file_ext = split_file_path(in_file)

    prefix = file_name[: -len(file_ext)]
    regex = "%s-([a-f0-9]*)\\." % prefix

    regex_files = []
    if os.path.exists(in_file):
        regex_files.append(file_name)
    else:
        for name in os.listdir(dir_path):
            if regex_search(name, pattern=regex):
                regex_files.append(name)

    if len(regex_files) > 1:
        newest_file = max(
            [os.path.join(dir_path, name) for name in regex_files],
            key=os.path.getmtime,
        )
        newest_file = os.path.basename(newest_file)
        logger.warning(
            format_msg(
                f"Found more than one hashed file with prefix `{prefix}` in "
                f"`{dir_path}`:\n {regex_files}.\n Default will use the newest"
                f" one `{newest_file}`. Please make sure it's the correct file"
                f", or set the correct checkpoint path with hash value.",
                MSGColor.RED,
            )
        )
        final_file = newest_file
    elif len(regex_files) == 1:
        final_file = regex_files[0]
    else:
        final_file = None

    if final_file:
        final_file = os.path.join(dir_path, final_file)
        assert os.path.exists(final_file), f"File {final_file} do not exist."

        if check_hash:
            sha = regex_search(os.path.basename(final_file), pattern=regex)
            if sha:
                if not check_sha256(final_file, sha):
                    raise ValueError(
                        f"The SHA256 checksum of {in_file} didn't match th expected {sha}."  # noqa E501
                    )
            else:
                logger.warning(
                    format_msg(
                        f"Don not found hash value in name of {final_file}, "
                        f"will skip check hash...",
                        MSGColor.RED,
                    )
                )

        return final_file
    else:
        raise ValueError(f"File {in_file} and its hashed file do not exist.")
