# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from abc import abstractmethod
from typing import Optional

from hat_engine.utils.graph_workflow.cache import CacheContext
from hat_engine.utils.graph_workflow.cache.context import SkipFlagCache
from hat_engine.utils.graph_workflow.trace import (
    CACHE_ARGS_TYPE,
    _exec_op_with_cache,
    execute_and_trace,
    get_current_graph_tracer,
)

__all__ = ["Operator"]


class Operator(object):
    def __init__(
        self,
        *,
        allow_input_skip=False,
        cache_args: Optional[CACHE_ARGS_TYPE] = "auto",
    ):
        assert isinstance(
            allow_input_skip, bool
        ), f"Expected type `bool`, get but {type(allow_input_skip)}"  # noqa
        self.__workflow_allow_input_skip__ = allow_input_skip
        self.cache_args = cache_args

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        graph_tracer = get_current_graph_tracer()
        if graph_tracer is None:
            ret = _exec_op_with_cache(
                self.forward,
                args,
                kwargs,
                obj_self=self,
                cache_context=CacheContext.current(),
                cache_args=self.cache_args,
            )
            if isinstance(ret, SkipFlagCache):
                return ret.data
            else:
                return ret
            # return self.forward(*args, **kwargs)
        else:
            return execute_and_trace(
                self,
                args,
                kwargs,
                workflow_allow_input_skip=self.__workflow_allow_input_skip__,  # noqa
                cache_args=self.cache_args,
            )  # noqa
