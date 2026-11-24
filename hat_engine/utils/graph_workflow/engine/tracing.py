# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, List

__all__ = ["Tracing", "TracerInterface"]


logger = logging.getLogger(__name__)


class TracerInterface(metaclass=ABCMeta):
    _current = None
    _inputs = None

    def __init__(self, input):
        self._inputs = input
        self.__class__._current = self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_trace):
        pass

    @abstractmethod
    def trace(self, trace_data: dict):
        pass

    @abstractmethod
    def node_before(self, node, args, kwargs):
        pass

    @abstractmethod
    def node_after(self, node, args, kwargs, ret, exception=None):
        pass


class Tracing:
    """
    Tracing class.

    BI Trace, tracing node when graph executing, recording infos
    after node execut, also recording custom trace data.
    """

    def __init__(self, tracers: List[TracerInterface] = None, inputs=None):
        self._tracers = {}
        self._tracers_status = {}
        self._last_error = None

        self._inputs = inputs
        if not tracers:
            return
        for tracer in tracers:
            try:
                self._tracers_status[tracer] = -1
                self._tracers[tracer] = tracer(inputs)
                self._tracers_status[tracer] = 0
            except Exception as e:
                logging.error(
                    f"tracer:{tracer} throw exception when enter, err:{str(e)}"
                )
                self._tracers_status[tracer] = -2

    def trace(self, trace_data: Dict):
        return self._traversal("trace", trace_data)

    def last_error(self):
        return self._last_error

    def __enter__(self):
        self._traversal("__enter__")

    def __exit__(self, exc_type, exc_val, exc_trace):
        self._traversal("__exit__", exc_type, exc_val, exc_trace)

    def node_before(self, node, args, kwargs):
        self._traversal("node_before", node, args, kwargs)

    def node_after(self, node, args, kwargs, ret=None, exception=None):
        self._traversal("node_after", node, args, kwargs, ret, exception)

    def _traversal(self, method, *args, **kwargs):
        for key, tracer in self._tracers.items():
            if self._tracers_status[key] < 0:
                continue
            try:
                getattr(tracer, method)(*args, **kwargs)
            except Exception as e:
                logging.error(
                    f"tracer:{tracer} throw exception when {method}, "
                    f"err:{str(e)}"
                )
                import traceback

                traceback.print_exc()
                self._tracers_status[key] = -1
                self._last_error = e
