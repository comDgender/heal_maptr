# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


import logging
import os
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

from hat_engine.utils.graph_workflow.cache import CacheContext
from hat_engine.utils.graph_workflow.cache.context import SkipFlagCache
from hat_engine.utils.graph_workflow.engine.tracing import Tracing
from hat_engine.utils.graph_workflow.exception import OpSkipFlag
from hat_engine.utils.graph_workflow.symbol import Symbol
from hat_engine.utils.graph_workflow.trace import _exec_op_with_cache
from hat_engine.utils.logger import MSGColor, format_msg

__all__ = ["BaseSymbolExecutor", "SymbolExecutor"]


logger = logging.getLogger(__name__)


@contextmanager
def fake_context():
    yield


HAT_ENABLE_LOGGING_SYMBOL_EXECUTOR = (
    os.environ.get("HAT_ENABLE_LOGGING_SYMBOL_EXECUTOR", "0") == "1"
)  # noqa


class BaseSymbolExecutor(object):
    """Base symbolic graph executor.

    Args:
        symbol: Symbolic graph.
    """

    def __init__(self, symbol: Symbol):
        assert isinstance(
            symbol, Symbol
        ), f"Expected {Symbol}, but get {type(symbol)}"
        self.symbol = symbol

    @abstractmethod
    def __call__(self, inputs: Dict[str, Any]):
        pass


class SymbolExecutor(BaseSymbolExecutor):
    """Simple executor to execute an symbolic graph, using dfs visit."""

    def __init__(
        self, symbol: Symbol, verbose: Optional[Union[bool, None]] = None
    ):
        super().__init__(symbol)
        self._head2child_map = {}
        for symbol_i in self.symbol:
            self._head2child_map[symbol_i.name] = set(
                symbol_i.get_children_name(recursive=True)
            )
        if verbose is None:
            verbose = HAT_ENABLE_LOGGING_SYMBOL_EXECUTOR
        self.verbose = verbose

    def _fvisit(self, node, tracing, inputs: Dict, node_outputs: Dict):
        def _has_skip_input(node):
            for input_i in node.inputs:
                if node_outputs[input_i.name] is OpSkipFlag:
                    return True
            return False

        if node.name in node_outputs:
            return

        if node.is_optional_placeholder:
            if node.name not in inputs:
                node_outputs[node.name] = node.attr["default"]
            else:
                node_outputs[node.name] = inputs[node.name]
            if self.verbose:
                logger.info(
                    f"{node.name} is OptionalVariable, value={node_outputs[node.name]}"  # noqa
                )  # noqa
            return

        if node.is_placeholder:
            assert (
                node.name in inputs
            ), f"Missing input for placeholder node {node.name}"
            node_outputs[node.name] = inputs[node.name]
            if self.verbose:
                logger.info(
                    f"{node.name} is Variable, value={node_outputs[node.name]}"  # noqa
                )  # noqa
            return

        if node.is_constant:
            node_outputs[node.name] = node.attr["data"]
            if self.verbose:
                logger.info(
                    f"{node.name} is Constant, value={node_outputs[node.name]}"  # noqa
                )  # noqa
            return

        if (
            _has_skip_input(node)
            and not node.attr["__workflow_allow_input_skip__"]
        ):  # noqa
            node_outputs[node.name] = OpSkipFlag
            if self.verbose:
                logger.info(
                    f"Some input of node={node.name} is skip, skip this node"  # noqa
                )  # noqa
            return

        args = node._map_args(
            node.args, lambda x: node_outputs[x.name]
        )  # noqa
        kwargs = node._map_args(
            node.kwargs, lambda x: node_outputs[x.name]
        )  # noqa
        if self.verbose:
            logger.info(
                f"Begin executing {node.name}, op={node.op}, args={args}, kwargs={kwargs}"  # noqa
            )  # noqa
        tracing.node_before(node, args, kwargs)

        try:
            ret = _exec_op_with_cache(
                node.op,
                args,
                kwargs,
                cache_context=CacheContext.current(),
                cache_args=node.attr.get("__cache_args__", "auto"),
                node_name=node.name,
            )
            if isinstance(ret, SkipFlagCache):
                msg = f"Skip down stream of node={node.name}, op={node.op}, args={args}, kwargs={kwargs}, reason={ret.reason}"  # noqa
                msg = format_msg(
                    msg,
                    MSGColor.RED,
                )
                logger.info(f"\n\n{msg}\n\n")
                ret = ret.data
            node_outputs[node.name] = ret
            tracing.node_after(node, args, kwargs, ret)
        except Exception as e:
            logger.error(
                f"Executing {node.name}, op={node.op}, args={args}, kwargs={kwargs} Failed! Exception: {e}"  # noqa
            )  # noqa
            tracing.node_after(node, args, kwargs, exception=e)
            raise e
        if self.verbose:
            logger.info(
                f"End executing {node.name}, op={node.op}, "
                f"args={args}, kwargs={kwargs}, outputs={node_outputs[node.name]}\n"  # noqa
            )  # noqa

    def __call__(
        self,
        inputs: Dict[str, Any],
        caches: Optional[Union[Dict, None]] = None,
        tracers=None,
        cache_ctx: Optional[Union[CacheContext, None]] = None,
    ):
        """Execute graph using dfs visit.

        When a :py:class:`hat_engine.graph_workflow.exception.SkipDownStream` exception occurs, will output a `OpSkipFlag` object.

        Args:
            inputs: Inputs, should contains all placeholder values.
        """  # noqa
        assert isinstance(
            inputs, dict
        ), f"Expected dict, but get {type(inputs)}"

        if caches is None:
            caches = {}
        assert isinstance(
            caches, dict
        ), f"Expected dict, but get {type(caches)}"

        node_outputs = {}

        tracing = Tracing(tracers, inputs)

        if cache_ctx is None:
            cache_ctx = fake_context()

        with tracing, cache_ctx:

            def inner(node):
                self._fvisit(node, tracing, inputs, node_outputs)
                caches[node.name] = node_outputs[node.name]

            self.symbol.post_order_dfs_visit(inner)

        # get head output
        outputs = []
        for node in self.symbol._outputs:
            if node.name not in node_outputs:
                outputs.append(OpSkipFlag)
            else:
                outputs.append(node_outputs[node.name])
        return outputs[0] if len(outputs) == 1 else outputs
