# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Copyright The Lightning AI team.
#
# This file has been modified by Beijing Horizon Information Technology Co., Ltd. on 2026/6/9.
# Modifications: adapted PythonProfiler to OBJECT_REGISTRY, decoupling
# them from the PyTorch Lightning framework.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cProfile
import functools
import io
import logging
import pstats
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from hat_engine.profiler.profilers import BaseProfiler, ProfilerAction
from hat_engine.registry import OBJECT_REGISTRY

__all__ = [
    "PythonProfiler",
]

logger = logging.getLogger(__name__)


def get_profile_stats(
    prof: cProfile.Profile,
    sort_by: str = "cumulative",
    print_fraction: float = 1.0,
):
    """Extract and returns the statistics from cProfile.

    Args:
        prof_file: cProfile.Profile instance.
        sort_by: Sorting criterion, default is "cumulative"
        print_fraction: Fraction of statistics to print, default is 1.0.

    """
    buf = io.StringIO()
    stat_obj = pstats.Stats(prof, stream=buf)
    stat_obj.sort_stats(sort_by)
    stat_obj.print_stats(print_fraction)
    return buf.getvalue()


@OBJECT_REGISTRY.register
class PythonProfiler(BaseProfiler):  # noqa: D205,D400
    """
    This profiler uses Python's cProfiler to record more detailed
    information about time spent in each function call recorded
    during a given action. The output is quite verbose and you should
    only use this if you want very detailed reports.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        line_count_restriction: float = 1.0,
        output_filename: Optional[str] = None,
        auto_describe: bool = True,
        summary_interval: int = -1,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
    ) -> None:  # noqa: D205,D400
        """
        Args:
            dirpath: Directory path for the ``filename``.
            filename: If present, filename where the profiler results will be
            saved instead of printing to stdout. The ``.txt`` extension will
            be used automatically.
            line_count_restriction: this can be used to limit the number of
            functions reported for each action. either an integer
            (to select a count of lines), or a decimal fraction between 0.0
            and 1.0 inclusive (to select a percentage of lines)
            auto_describe: Whether to automatically output summary
                information at regular intervals.
            schedule: A callable func that takes step (int) as a single
                parameter and returns ``ProfilerAction`` value that specifies
                the profiler action to perform at each step. For example:
                `hat.profiler.profilers.schedule(wait=0, warmup=0, active=10)`
            summary_interval: Step interval for print summary.


        Raises:
            ValueError:
                If you attempt to stop recording an action which was
                never started.
        """
        super(PythonProfiler, self).__init__(
            dirpath=dirpath,
            filename=filename,
            auto_describe=auto_describe,
            summary_interval=summary_interval,
            schedule=schedule,
        )
        self.profiled_actions: Dict[str, cProfile.Profile] = {}
        self.line_count_restriction = line_count_restriction
        logger.warning(
            "Sometimes you want to get a specific line's information,"
            "we recommand you not to use multiple processes in your code."
        )
        logger.warning(
            "PythonProfiler is suitable for perf of cpu operation,"
            "not recommended for GPU training operation."
        )

    def start(self, action_name: str) -> None:
        if action_name not in self.profiled_actions:
            self.profiled_actions[action_name] = cProfile.Profile()
        self.profiled_actions[action_name].enable()

    def stop(self, action_name: str) -> None:
        active_prof = self.profiled_actions.get(action_name)
        if active_prof is None:
            raise ValueError(
                f"Attempting to stop recording an action "
                f"({action_name}) which was never started."
            )
        active_prof.disable()

    def reset(self):
        self.profiled_actions.clear()

    def summary(self) -> str:
        action_stats = {}
        for action_name, action_prof in self.profiled_actions.items():
            out_buf = io.StringIO()
            stats = pstats.Stats(action_prof, stream=out_buf)
            stats.sort_stats("cumulative")
            stats.print_stats(self.line_count_restriction)
            action_stats[action_name] = out_buf.getvalue()
        return self._stats_to_str(action_stats)

    def teardown(self) -> None:
        super(PythonProfiler, self).teardown()
        self.profiled_actions = {}

    def __reduce__(self):
        # avoids `TypeError: cannot pickle 'cProfile.Profile' object`
        return (
            self.__class__,
            (),
            {
                "dirpath": self.dirpath,
                "filename": self.filename,
                "line_count_restriction": self.line_count_restriction,
            },
        )


def py_profile(
    seconds: Optional[float] = None,
    call_freq: int = 1,
    output_file: Optional[str] = None,
):
    """Python profile wrapper.

    Args:
        seconds: Timeout thresh. If `None`, the profile result will be output
            every time func is called. Otherwise, the profile result will only
            be output when the execution time of the function exceeds threshold
        call_freq: The frequency of function calls to print the profile result.
            Default `1`, the profile result will be printed every time the
            function is called.
        output_file: The file to save the profile result. If `None`, the
            profile result will be printed to the console.

    Examples:
        with timeout: only print profile result when the execution time of
            `test_func` is overthan `seconds` second.
        >>> @py_profile(seconds=0.5)
        ... def test_func1():
        ...    time.sleep(1)

        no timeout: always print profile result when `test_func` is excuted.
        >>> @py_profile()
        ... def test_func2():
        ...    time.sleep(1)

        >>> @py_profile(call_freq=1, output_file="profile.log")
        ... def test_func3():
        ...    time.sleep(1)
    """

    if seconds is not None and call_freq > 1:
        raise ValueError(
            "If `seconds` is not None, `call_freq` must be 1. "
            "Because the timeout threshold (`seconds`) is designed to trigger"
            "profiling output based on cumulative execution time, which "
            "conflicts with the periodic output controlled by `call_freq`."
        )

    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            def _init_state():
                wrapped_func._profiler = cProfile.Profile()
                wrapped_func._call_count = 0
                wrapped_func._cost_time = 0.0

            if not hasattr(wrapped_func, "_profiler"):
                _init_state()

            wrapped_func._profiler.enable()
            t_begin = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                t_end = time.time()

                wrapped_func._profiler.disable()
                wrapped_func._call_count += 1
                elapsed = round((t_end - t_begin), 4)
                wrapped_func._cost_time += elapsed

                emit_output = False

                summary_msg = (
                    f"{func.__name__} called {wrapped_func._call_count} "
                    f"times, cost time: "
                    f"{round(wrapped_func._cost_time, 3)}s"
                )

                timeout_triggered = (
                    seconds is not None and wrapped_func._cost_time >= seconds
                )
                if timeout_triggered:
                    emit_output = True
                    summary_msg = (
                        "[Timeout Warning] " + summary_msg + f"(>{seconds}s)"
                    )

                freq_triggered = (
                    call_freq > 0 and wrapped_func._call_count >= call_freq
                )
                if freq_triggered:
                    emit_output = True
                    summary_msg = "[LogFreq] " + summary_msg

                if emit_output:
                    prof_text = get_profile_stats(
                        wrapped_func._profiler, sort_by="cumulative"
                    )
                    summary_msg = "[Profile] " + summary_msg + f"\n{prof_text}"

                    if output_file:
                        with open(output_file, "a") as fh:
                            fh.write(f"{summary_msg}\n")
                    else:
                        logger.warning(summary_msg)

                    wrapped_func._profiler.clear()
                    _init_state()

            return result

        return wrapped_func

    return decorator
