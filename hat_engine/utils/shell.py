# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import subprocess

__all__ = ["get_bin_path", "get_lib_path"]


def get_bin_path(name: str):
    """Get binary file path.

    Args:
        name: Binarary file name

    Returns:
        flag: Whether the binary exists or not
        message: If exists, return the binary file path. Otherwise, return the
            error message.
    """
    assert isinstance(name, str)
    p = subprocess.Popen(
        ["which", name],
        close_fds=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p.wait()
    if p.returncode == 0:
        flag = True
        message = p.stdout.read().decode().strip()
    else:
        flag = False
        message = p.stderr.read().decode()
    p.stdin.close()
    p.stdout.close()
    p.stderr.close()

    return flag, message


def get_lib_path(name: str):
    """Get library file path.

    Args:
        name: Library file name

    Returns:
        flag: Whether the library exists or not
        message: If exists, return the library file path. Otherwise, return the
            error message.
    """
    assert isinstance(name, str)
    p = subprocess.Popen(
        ["whereis", name],
        close_fds=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    p.wait()
    if p.returncode == 0:
        message = p.stdout.read().decode().strip()
        possible_paths = message.split(":")[1]
        if possible_paths == "":
            flag = False
        else:
            flag = True
            message = possible_paths.strip().split(" ")[0]
    else:
        flag = False
        message = p.stderr.read().decode()
    p.stdin.close()
    p.stdout.close()
    p.stderr.close()

    return flag, message
