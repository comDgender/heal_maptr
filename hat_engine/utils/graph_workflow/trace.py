# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import dis
import functools
import inspect
import logging
import operator
import threading
import types
from collections import ChainMap, OrderedDict
from collections.abc import Mapping
from enum import Enum
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from hatbc.workflow.proxy import WorkflowVariable as hatbc_WorkflowVariable
    from hatbc.workflow.proxy import (
        _ProxyDataUnsetFlag as hatbc_ProxyDataUnsetFlag,
    )
except ImportError:
    hatbc_WorkflowVariable = None
    hatbc_ProxyDataUnsetFlag = None

from hat_engine.utils.apply_func import _as_list, map_aggregate
from hat_engine.utils.graph_workflow.cache import (
    CacheArgs,
    CacheContext,
    CacheNotExistsError,
    DisableCacheContext,
    OpCacheContext,
)
from hat_engine.utils.graph_workflow.cache.context import SkipFlagCache
from hat_engine.utils.graph_workflow.exception import (
    OpSkipFlag,
    ReturnOpSkipFlagIsNotAllowedError,
    SkipDownStream,
)
from hat_engine.utils.graph_workflow.patch import Patcher
from hat_engine.utils.graph_workflow.symbol import (
    Node,
    Symbol,
    get_operator_name,
    group,
)
from hat_engine.utils.logger import MSGColor, format_msg

__all__ = [
    "combine_dict",
    "get_item_from_dict",
    "get_part_dict",
    "get_item_from_list",
    "WorkflowVariable",
    "Variable",
    "OptionalVariable",
    "Constant",
    "get_traced_graph",
    "get_traced_outputs",
    "ControlFlow",
    "GraphTracer",
    "make_traceable",
]


logger = logging.getLogger(__name__)


CACHE_ARGS_TYPE = Union[CacheArgs, str]


class _ProxyDataUnsetFlag:
    pass


def _not_implemented(func):
    def _inner(*args, **kwargs):
        name = func.__name__
        raise NotImplementedError(
            f"{WorkflowVariable} does not support function {name}"
        )  # noqa

    return _inner


class WorkflowVariable(object):
    """
    Bind data with :py:class:`Symbol`.

    This data structure is used to trace the computational graph.

    .. note::

        We need this data strucure to trace the computational graph because
        the input and output of our workflow operators are arbitrary.

    Args:
        symbol: Symbol
        data: Data
    """

    def __init__(self, symbol, data=_ProxyDataUnsetFlag):
        self._isymbol = symbol
        self._idata = data

    @property
    def _iop(self):
        return self._isymbol._outputs[0].op

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise RuntimeError("Cannot modify type(self)")
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self.__dict__:
            return super().__getattr__(name)
        else:
            variable = execute_and_trace(getattr, args=(self, name))
            if isinstance(variable, WorkflowVariable):
                return Attribute(
                    pre_variable=self, variable=variable, attr_name=name
                )
            else:
                return variable

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()

    __deepcopy__ = None

    @_not_implemented
    def __hash__(self):
        pass

    @_not_implemented
    def __str__(self):
        pass

    @_not_implemented
    def __bool__(self):
        pass

    def __call__(self, *args, **kwargs):
        return call_obj(self, *args, **kwargs)

    def __iter__(self):
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        instructions_dict = {
            instr.offset: instr
            for instr in dis.get_instructions(calling_frame.f_code)
        }
        inst = instructions_dict[calling_frame.f_lasti]
        if inst.opname == "UNPACK_SEQUENCE":
            return (self[idx] for idx in range(inst.argval))
        elif inst.opname == "GET_ITER":
            if isinstance(self._idata, (list, tuple)):
                return iter([self[idx] for idx in range(len(self._idata))])
            elif isinstance(self._idata, dict):
                return iter(self._idata)
            elif self._idata is _ProxyDataUnsetFlag:
                raise NotImplementedError(
                    "Cannot return iter when data is unset"
                )  # noqa
            else:
                raise NotImplementedError(
                    f"Unsupported iter for {type(self._idata)}"
                )  # noqa
        else:
            raise NotImplementedError

    def keys(self):
        assert self._idata is not _ProxyDataUnsetFlag
        return self._idata.keys()

    def __len__(self):
        assert self._idata is not _ProxyDataUnsetFlag
        return len(self._idata)

    def wait(self, variables):
        """Wait for computation to be finished."""
        for variable_i in _as_list(variables):
            assert isinstance(
                variable_i, WorkflowVariable
            ), f"Expected type {WorkflowVariable}, but get {type(variable_i)}"
            self._isymbol.wait(variable_i._isymbol)

    def after(self, *args, **kwargs):
        return self.wait(*args, **kwargs)


class Variable(WorkflowVariable):
    """
    Bind data with name.

    Args:
        data: Data
        name: Name, should be uniquely
    """

    def __init__(self, name, data=_ProxyDataUnsetFlag):
        assert isinstance(name, str), f"Expected str, but get {type(name)}"
        super().__init__(Symbol.create_placeholder(name), data)
        self._iname = name


class OptionalVariable(WorkflowVariable):
    def __init__(self, name, default):
        assert isinstance(name, str), f"Expected str, but get {type(name)}"
        assert (
            default is not _ProxyDataUnsetFlag
        ), f"data cannot be {_ProxyDataUnsetFlag} for {type(self)}"  # noqa
        super().__init__(
            Symbol.create_optional_placeholder(name, default), default
        )  # noqa
        self._iname = name


class Constant(WorkflowVariable):
    def __init__(self, data, name=None):
        super().__init__(Symbol.create_constant(data, name), data)
        self._iname = self._isymbol.name


class Attribute(WorkflowVariable):
    """
    Proxy get object attribute or get object attribute and call.

    Args:
        pre_variable: Previous variable
        variable: Variable for get attribute operator
        attr_name: Get attribute name
    """

    def __init__(self, pre_variable, variable, attr_name):
        assert isinstance(pre_variable, WorkflowVariable)
        self._ipre_variable = pre_variable
        assert isinstance(variable, WorkflowVariable)
        self._ivariable = variable
        self._iattr_name = attr_name

    @property
    def _isymbol(self):
        return self._ivariable._isymbol

    @property
    def _idata(self):
        return self._ivariable._idata

    def __call__(self, *args, **kwargs):
        return get_attr_and_call(
            self._ipre_variable, self._iattr_name, *args, **kwargs
        )  # noqa


reflectable_magic_methods = (
    "add",
    "sub",
    "mul",
    "floordiv",
    "truediv",
    "div",
    "mod",
    "pow",
    "lshift",
    "rshift",
    "and",
    "or",
    "xor",
)
magic_methods = (
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
    "pos",
    "neg",
    "invert",
    "getitem",
) + reflectable_magic_methods


# patch magic methods
for method_i in magic_methods:

    def wrap(method):
        def _impl(self, *args, **kwargs):
            return execute_and_trace(
                getattr(operator, method), args=(self,) + args, kwargs=kwargs
            )

        _impl.__name__ = method
        setattr(WorkflowVariable, f"__{method}__", _impl)

    wrap(method_i)

# patch reflectable magic methods
for method_i in reflectable_magic_methods:

    def wrap_reflectable(method):
        def _impl(self, rhs):
            return execute_and_trace(
                getattr(operator, method), args=(rhs, self), kwargs=None
            )

        _impl.__name__ = method
        setattr(WorkflowVariable, f"__r{method}__", _impl)

    wrap_reflectable(method_i)


def cache_by_node_name(ctx: OpCacheContext) -> Union[str, None]:
    """Return node name as the hash value.

    .. note::
        Use this feature carefully because it does not take care of the operator implementation and arguments.
    """  # noqa
    return ctx.node_name


def _check_cache_args(cache_args):
    if isinstance(cache_args, CacheArgs):
        return
    if isinstance(cache_args, str) and cache_args in ["auto", "disable"]:
        return
    raise ValueError(
        "possible values are {instance of CacheArgs, auto, disable}"
    )


def _exec_op_with_cache(
    op: Callable,
    args: Tuple,
    kwargs: Dict,
    obj_self: Optional[Union[Any, None]] = None,
    allow_return_skip_flag: bool = False,
    cache_context: Optional[Union[CacheContext, None]] = None,
    cache_args: Optional[CACHE_ARGS_TYPE] = "auto",
    **op_cache_ctx,
) -> Any:
    """Execute operator with cache mechanism."""

    def _run_op_and_check():
        # disable cache when running operator
        with DisableCacheContext():
            try:
                ret = op(*args, **kwargs)
                if ret is OpSkipFlag and not allow_return_skip_flag:
                    hint_name = op_cache_ctx.get(
                        "node_name",
                        get_operator_name(op if obj_self is None else op),
                    )
                    raise ReturnOpSkipFlagIsNotAllowedError(
                        f"Op {hint_name} has return `OpSkipFlag`, which is not allowed, raise SkipDownStream instead."  # noqa
                    )
            except SkipDownStream as e:
                ret = SkipFlagCache(data=OpSkipFlag, reason=e.args)
            return ret

    # check enable cache or not
    if cache_args == "disable" or cache_context is None:
        disable_cache_flags = True
        current_cache_args = None
    else:
        if cache_args == "auto":
            current_cache_args = cache_context.cache_args
        else:
            current_cache_args = cache_args
        disable_cache_flags = current_cache_args is None

    if disable_cache_flags:
        return _run_op_and_check()

    op_ctx = OpCacheContext(
        fn=op if obj_self is None else obj_self,
        args=args,
        kwargs=kwargs,
        **op_cache_ctx,
    )

    # compute cache key
    try:
        cache_key = current_cache_args.cache_key_fn(op_ctx)
    except Exception as e:
        if not current_cache_args.ignore_cache_failure:
            raise e
        logger.debug(f"Ignore cache for `{op}` due to `{e.args}")
        cache_key = None

    # if not cache_key
    if cache_key is None:
        return _run_op_and_check()
    try:
        ret = cache_context.get(cache_key, current_cache_args.serializer_impl)
        logger.debug(
            "\n\n[Cache] Using cache result for node {}\n\n".format(
                op_ctx.node_name
            )
        )
        return ret
    except CacheNotExistsError:
        ret = _run_op_and_check()
        cache_context.set(cache_key, ret, current_cache_args.serializer_impl)
        return ret


class TraceFn:
    def _map_func(self, x, take_node=False, check_type=True):
        if isinstance(x, WorkflowVariable):
            if take_node:
                assert len(x._isymbol) == 1
                return x._isymbol._outputs[0]
            else:
                assert (
                    x._idata is not _ProxyDataUnsetFlag
                ), f"You should bind data with WorkflowVariable {x._isymbol.name} when tracing with imperative mode or in dynamic mode"  # noqa
                return x._idata

        if hatbc_WorkflowVariable is not None and isinstance(
            x, hatbc_WorkflowVariable
        ):
            if take_node:
                assert len(x._isymbol) == 1
                return x._isymbol._outputs[0]
            else:
                assert (
                    x._idata is not hatbc_ProxyDataUnsetFlag
                ), f"You should bind data with WorkflowVariable {x._isymbol.name} when tracing with imperative mode or in dynamic mode"  # noqa

        if check_type:
            assert isinstance(
                x, GraphTracer._BASIC_TYPES
            ), f"Invalid type {type(x)}, only allow {GraphTracer._BASIC_TYPES}"  # noqa
        return x

    def build_graph(
        self,
        graph_tracer: "GraphTracer",
        op,
        args,
        kwargs,
        output,
        node_attr: Dict,
        wait_nodes: Optional[Tuple] = (),
        node_name: Optional[Union[str, None]] = None,
    ) -> "WorkflowVariable":
        if node_name is None:
            graph_tracer.trace_op(op)
            node_name = graph_tracer.get_op_name(op, args, kwargs)

        node = Node(
            op=op,
            name=node_name,
            args=map_aggregate(
                args, functools.partial(self._map_func, take_node=True)
            ),
            kwargs=map_aggregate(
                kwargs, functools.partial(self._map_func, take_node=True)
            ),
            attr=node_attr,
        )
        symbol = Symbol(node)
        ret = WorkflowVariable(symbol=symbol, data=output)

        if wait_nodes:
            ret.wait(wait_nodes)

        return ret

    def __call__(
        self,
        op,
        args=None,
        kwargs=None,
        *,
        workflow_allow_input_skip: Optional[bool] = False,
        cache_args: Optional[CACHE_ARGS_TYPE] = "auto",
    ) -> "WorkflowVariable":
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        graph_tracer = get_current_graph_tracer()
        graph_tracer_active = graph_tracer is not None

        if graph_tracer_active:
            graph_tracer.trace_op(op)
            node_name = graph_tracer.get_op_name(op, args, kwargs)
        else:
            node_name = None

        has_op_skip_flag = False

        wait_nodes = []

        if graph_tracer is not None and ControlFlow.is_active():
            for node in ControlFlow.current().wait_nodes:
                if isinstance(node, WorkflowVariable):
                    wait_nodes.append(node)
                    if node._idata is OpSkipFlag:
                        has_op_skip_flag = True
                elif node is OpSkipFlag:
                    has_op_skip_flag = True

        def _check_has_skip(x):
            if isinstance(x, WorkflowVariable) and x._idata is OpSkipFlag:
                nonlocal has_op_skip_flag
                has_op_skip_flag = True

        map_aggregate(args, _check_has_skip)
        map_aggregate(kwargs, _check_has_skip)

        if graph_tracer is None or graph_tracer.imperative:
            with GraphTracerPause():
                new_args = map_aggregate(
                    args,
                    functools.partial(
                        self._map_func, check_type=graph_tracer_active
                    ),  # noqa
                )
                new_kwargs = map_aggregate(
                    kwargs,
                    functools.partial(
                        self._map_func, check_type=graph_tracer_active
                    ),  # noqa
                )

                if has_op_skip_flag and not workflow_allow_input_skip:
                    logger.debug(
                        f"[Tracing]: Some child node has raise the {SkipDownStream} exception, ignore executing {op}"  # noqa
                    )
                    output = OpSkipFlag
                else:
                    output = _exec_op_with_cache(
                        op,
                        new_args,
                        new_kwargs,
                        cache_context=CacheContext.current(),
                        cache_args=cache_args,
                        node_name=node_name,
                    )
                    if isinstance(output, SkipFlagCache):
                        msg = f"[Tracing]: Skip down stream of op={op}, args={new_args}, kwargs={new_kwargs}, reason={output.reason}"  # noqa
                        msg = format_msg(
                            msg,
                            MSGColor.RED,
                        )
                        logger.debug(f"\n\n{msg}\n\n")  # noqa
                        output = output.data

        else:
            output = _ProxyDataUnsetFlag

        if graph_tracer is None:
            return output

        return self.build_graph(
            graph_tracer,
            op,
            args,
            kwargs,
            output,
            {
                "__workflow_allow_input_skip__": workflow_allow_input_skip,
                "__cache_args__": cache_args,
            },
            wait_nodes,
            node_name=node_name,
        )


execute_and_trace = TraceFn()


def _make_function_traceable(
    fn,
    *,
    allow_input_skip=False,
    trace_fn=None,
    cache_args: Optional[CACHE_ARGS_TYPE] = "auto",
):
    """
    Wrap function `fn` to make it traceable underGraphTracer scope.

    Hits:
    No-op when outside it.
    """

    if trace_fn is None:
        trace_fn = TraceFn()

    @functools.wraps(fn)
    def _wrap(*args, **kwargs):
        """
        Wrap function `fn` to make it traceable under GraphTracer scope.

        This function is wrapped by `hat_engine.utils.graph_workflow` to
        make it traceable under GraphTracer scope (no-op when outside it).
        """
        graph_tracer = get_current_graph_tracer()
        if graph_tracer is None:
            ret = _exec_op_with_cache(
                fn,
                args,
                kwargs,
                cache_context=CacheContext.current(),
                cache_args=cache_args,
            )
            if isinstance(ret, SkipFlagCache):
                return ret.data
            return ret
        else:
            return trace_fn(
                fn,
                args,
                kwargs,
                workflow_allow_input_skip=allow_input_skip,
                cache_args=cache_args,
            )  # noqa

    _wrap.__workflow_traceable__ = True
    _wrap.__workflow_allow_input_skip__ = allow_input_skip
    return _wrap


def _make_class_traceable(
    obj,
    *,
    return_patch_func_only: Optional[bool] = False,
    trace_fn: Optional[Union[TraceFn, None]] = None,
    allow_input_skip: Optional[bool] = False,
    cache_args: Optional[CACHE_ARGS_TYPE] = "auto",
):
    """Wrap __call__ function of class `obj` to make it traceable under GraphTracer scope (no-op when outside it)."""  # noqa

    if trace_fn is None:
        trace_fn = TraceFn()

    assert callable(obj), "object should have the __call__ function"
    assert isinstance(
        obj.__call__, types.FunctionType
    ), "the __call__ function of object should not be static"

    fn = obj.__call__

    @functools.wraps(fn)
    def _wrap(self, *args, **kwargs):
        """
        Wrap function `fn` to make it traceable under GraphTracer scope.

        This function is wrapped by `hat_engine.utils.graph_workflow` to
        make it traceable under GraphTracer scope (no-op when outside it).
        """
        graph_tracer = get_current_graph_tracer()
        if graph_tracer is None:
            ret = _exec_op_with_cache(
                fn,
                (self,) + args,
                kwargs,
                obj_self=self,
                cache_context=CacheContext.current(),
                cache_args=cache_args,
            )
            if isinstance(ret, SkipFlagCache):
                return ret.data
            return ret
        else:
            return trace_fn(
                self,
                args,
                kwargs,
                workflow_allow_input_skip=allow_input_skip,
                cache_args=cache_args,
            )  # noqa

    _wrap.__workflow_traceable__ = True
    _wrap.__workflow_allow_input_skip__ = allow_input_skip

    if return_patch_func_only:
        return _wrap
    else:
        obj.__call__ = _wrap
        return obj


def make_traceable(
    obj: Optional[Union[Callable, None]] = None,
    *,
    trace_fn: Optional[Union[TraceFn, None]] = None,
    allow_input_skip: Optional[bool] = False,
    cache_args: Optional[CACHE_ARGS_TYPE] = "auto",
):
    """
    Decorate to make a callable object to be an workflow operator.

    .. note::
        Any operator decorated by this decorator cannot return :py:class:`hat_engine.utils.graph_workflow.exception.OpSkipFlag`

    Args:
        obj: Wrapped callable object, by default None.
        trace_fn: The trace function to build the graph, by default None.
        allow_input_skip: Whether allow input is skip, for more information,
            please see the workflow tutorial, by default False.
        cache_args: Cache argument, possible values are
            {instance of CacheArgs, auto, disable}, `disable` means disable cache, `auto` means
            using cache arguments from CacheContext, by default auto.
    """  # noqa

    if trace_fn is None:
        trace_fn = TraceFn()

    _check_cache_args(cache_args)

    def _impl(obj):
        assert callable(obj), f"{obj} should be callable"

        if is_workflow_traceable(obj):
            return obj

        if inspect.isfunction(obj) or isinstance(obj, MethodType):
            return _make_function_traceable(
                obj,
                trace_fn=trace_fn,
                allow_input_skip=allow_input_skip,
                cache_args=cache_args,
            )
        else:
            return _make_class_traceable(
                obj,
                trace_fn=trace_fn,
                allow_input_skip=allow_input_skip,
                cache_args=cache_args,
            )

    if obj is None:
        return _impl
    else:
        return _impl(obj)


def is_workflow_traceable(obj: Any) -> bool:
    def _trace_attr_is_true(fn):
        return (
            hasattr(fn, "__workflow_traceable__") and fn.__workflow_traceable__
        )

    if inspect.isfunction(obj):
        return _trace_attr_is_true(obj)
    elif callable(obj):
        return _trace_attr_is_true(obj.__call__)
    else:
        return False


class GraphTracer(object):
    """
    Tracing workflows when running under this scope.

    Args:
        imperative: Whether tracing graph using imperative
            mode, by default False.

    Imperative means executing the operator and then tracing, otherwise
    just tracing without executing.

    .. code-block:: python

        @make_traceable
        def add(a, b):
            ret = a + b
            print(ret)  # when executing, will print the result
            return ret

        a = Variable('a', 1)
        b = Variable('b', 2)

        with GraphTracer(imperative=True):
            c = add(a, b)  # the terminal will output 3

        with GraphTracer(imperative=False):
            c = add(a, b)  # the terminal will output nothing
    """

    # thread id 2 current GraphTracer instance or None
    _tid2current = {}

    _TRACEABLE_CLASSES = ()
    _BASIC_TYPES = (int, float, str, type(None), bool, Enum)
    _OP_NAME_FUNC = {}

    def __init__(self, imperative=False):
        self._old_scope = None
        self._op2count = {}
        self.imperative = imperative
        self._patcher = Patcher()

    def __enter__(self):
        self._old_scope = GraphTracer.current()

        if self._old_scope is not None:
            self._op2count = copy.copy(self._old_scope._op2count)

        GraphTracer.set_current(self)

        self._patcher.__enter__()
        for obj in GraphTracer._TRACEABLE_CLASSES:
            if not is_workflow_traceable(obj):
                self._patcher.patch(
                    obj,
                    "__call__",
                    _make_class_traceable(obj, return_patch_func_only=True),
                )

        return self

    def __exit__(self, ptype, value, trace):
        self._patcher.__exit__(ptype, value, trace)

        if self._old_scope is not None:
            self._old_scope._op2count.update(self._op2count)

        # recovery current
        if self._old_scope is None:
            tid = threading.get_ident()
            # pop to avoid dict `_tid2current` explosion.
            GraphTracer._tid2current.pop(tid)
        else:
            GraphTracer.set_current(self._old_scope)

    def trace_op(self, op):
        op_type_name = get_operator_name(op)
        if op_type_name not in self._op2count:
            self._op2count[op_type_name] = 0
        self._op2count[op_type_name] += 1

    def get_op_name(self, op, args, kwargs):
        op_type_name = get_operator_name(op)
        count = self._op2count[op_type_name]
        if op_type_name in GraphTracer._OP_NAME_FUNC:
            return GraphTracer._OP_NAME_FUNC[op_type_name](
                op, args, kwargs, count
            )
        else:
            return f"{op_type_name}{count}"

    @classmethod
    def register_class_traceable_under_scope(cls, fn):
        assert type(fn).__name__ == "type", f"{fn} is not class"
        assert callable(fn), f"{fn} should be callable"
        if fn not in cls._TRACEABLE_CLASSES:
            cls._TRACEABLE_CLASSES += (fn,)

    @classmethod
    def register_basic_types(cls, obj):
        if obj not in cls._BASIC_TYPES:
            cls._BASIC_TYPES += (obj,)

    @classmethod
    def register_op_name_function(cls, name: str, func: Callable):
        """Register name function for operator named `name`.

        Args:
            name: Operator name.
            func: This function is called in the following way

        .. code-block:: python

            name = func(op, op_args, op_kwargs, conut)
        """
        if name in cls._OP_NAME_FUNC:
            raise KeyError(f"{name} already registered")
        cls._OP_NAME_FUNC[name] = func

    @classmethod
    def current(cls) -> Union["GraphTracer", None]:
        """Get current GraphTracer instance of current thread.

        .. note::

            Different thread owns different `current`.

        Returns:
            current: Current GraphTracer instance.
        """
        tid = threading.get_ident()
        cls._tid2current.setdefault(tid, None)
        return cls._tid2current[tid]

    @classmethod
    def set_current(cls, current: Union["GraphTracer", None]):
        """Set current GraphTracer instance of current thread.

        .. note::

            Different thread owns different `current`.

        Args:
            current: Current GraphTracer instance.
        """
        assert current is None or isinstance(current, GraphTracer), type(
            current
        )
        tid = threading.get_ident()
        cls._tid2current[tid] = current

    @classmethod
    def is_active(cls):
        """Whether is under the GraphTracer scope."""
        return cls.current() is not None


def get_current_graph_tracer():
    return GraphTracer.current()


class GraphTracerPause(object):
    def __init__(self):
        self._old_scope = None

    def __enter__(self):
        self._old_scope = GraphTracer.current()
        GraphTracer.set_current(None)

    def __exit__(self, ptype, value, trace):
        GraphTracer.set_current(self._old_scope)


class ControlFlow:
    _current = None

    def __init__(self, *, wait=()):
        self._old = None
        self.wait_nodes = list(_as_list(wait))

    def __enter__(self):
        self._old = ControlFlow._current
        if self._old is None:
            ControlFlow._current = self
            return self
        else:
            ControlFlow._current = ControlFlow._current.copy()
            ControlFlow._current.wait_nodes.extend(self.wait_nodes)
            return ControlFlow._current

    def __exit__(self, ptype, value, trace):
        ControlFlow._current = self._old
        self._old = None

    def copy(self):
        return ControlFlow(wait=copy.copy(self.wait_nodes))

    @classmethod
    def current(cls):
        return cls._current

    @classmethod
    def is_active(cls):
        return cls._current is not None


class ControlFlowPause:
    def __init__(self):
        self._old = None

    def __enter__(self):
        self._old = ControlFlow._current
        ControlFlow._current = None
        return self

    def __exit__(self, ptype, value, trace):
        ControlFlow._current = self._old
        self._old = None


@make_traceable
def call_obj(fn, *args, **kwargs):
    return fn(*args, **kwargs)


@make_traceable
def get_attr_and_call(obj, attr_name, *args, **kwargs):
    attr = getattr(obj, attr_name)
    return attr(*args, **kwargs)


def get_traced_graph(*outputs):
    """Get graph by traced outputs."""

    # for backward compatible
    if len(outputs) == 1:
        outputs = _as_list(outputs[0])

    assert all(
        [isinstance(out_i, WorkflowVariable) for out_i in _as_list(outputs)]
    )  # noqa

    return group(*(out_i._isymbol for out_i in _as_list(outputs)))


def get_traced_outputs(*outputs):
    # for backward compatible
    if len(outputs) == 1:
        outputs = _as_list(outputs[0])

    assert len(outputs) >= 1, "At least one output"

    assert all(
        [isinstance(out_i, WorkflowVariable) for out_i in _as_list(outputs)]
    )  # noqa

    outputs = [out_i._idata for out_i in _as_list(outputs)]
    return outputs[0] if len(outputs) == 1 else outputs


@make_traceable
def combine_dict(*dict_list):
    """
    Traceable function for combining multi dict to a single dict.

    usually used for building a graph model.
    """
    return OrderedDict(ChainMap(*dict_list))


@make_traceable
def get_item_from_list(data_list, index):
    """
    Traceable function return an item from a list.

    usually used for building a graph model.
    """
    return data_list[index]


@make_traceable
def get_item_from_dict(data_dict, name):
    """
    Traceable function to return an item from a dict.

    usually used for building a graph model.
    """
    return data_dict.get(name, None)


@make_traceable
def get_part_dict(data_dict: Mapping, names: List):
    """
    Traceable function to return a subsetion of of original dict.

    usually used for building a graph model.
    """
    result = OrderedDict()
    for name in names:
        if name in data_dict:
            result[name] = data_dict[name]
    return result
