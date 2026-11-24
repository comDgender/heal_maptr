# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


__all__ = ["SkipDownStream", "OpSkipFlag", "ReturnOpSkipFlagIsNotAllowedError"]


class SkipDownStream(Exception):
    """For operators that want to skip down stream task, raise this exception.

    .. note::

        This only works when :py:class:`hat_engine.utils.graph_workflow.trace.GraphTracer` is active.
    """  # noqa


class OpSkipFlag:
    """The output value for skipped operators."""  # noqa


class ReturnOpSkipFlagIsNotAllowedError(Exception):
    ...
