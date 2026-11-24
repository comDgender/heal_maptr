# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import time

__all__ = ["wait_until_finish"]


def wait_until_finish(
    query_func, timeout=None, query_interval=5, log_interval=30, wait_what=None
):
    """
    Wait until finish.

    Args:
        query_func : callable
            The way to query whether the task is finished or not, return True
            or False, this function is called is the following way

            .. code-block:: python

                flag = query_func()
        timeout : float, optional
            Maximum waiting time, by default None
        query_interval : float, optional
            Wait until `query_interval` and then query again, by default 5
        log_interval : float, optional
            Log every `log_interval` seconds, by default 30
        wait_what : str, optional
            Waiting what, used for logging, by default None
    """
    tic = time.time()
    pre_log_interval = log_interval
    while True:
        if query_func():
            break
        time.sleep(query_interval)
        time_cost = time.time() - tic
        if timeout is not None and time_cost > timeout:
            raise TimeoutError
        if time_cost > pre_log_interval:
            if wait_what:
                print(f"{wait_what}: cost {time_cost} seconds")
            else:
                print(f"Waiting cost {time_cost} seconds")
            pre_log_interval += log_interval
