# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
from typing import Callable, List, Optional

from horizon_plugin_profiler.utils.model_helper import register_hook_on_leaf
from torch.utils.hooks import RemovableHandle

from hat_engine.callbacks.callbacks import CallbackMixin
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import (
    match_children_modules_by_name,
    match_children_modules_by_regex,
)

logger = logging.getLogger(__name__)

__all__ = ["HookModel"]


@OBJECT_REGISTRY.register
class HookModel(CallbackMixin):
    """Register hooks on all leaf modules in a model.

    Args:
        modules: modules to add hook.
        regex: Whether to match by regex. if not, match by module name.
        strict: Whether regular expression is required to be all matched.
        forward_pre_hook: The forward_pre_hook to register.
        forward_hook: The forward_hook to register.
        backward_hook: The backward_hook to register.
    """

    def __init__(
        self,
        modules: Optional[List[str]] = None,
        regex: bool = True,
        strict: bool = False,
        forward_pre_hook: Optional[Callable] = None,
        forward_hook: Optional[Callable] = None,
        backward_hook: Optional[Callable] = None,
    ) -> None:

        self.modules = modules
        self.regex = regex
        self.strict = strict
        self.forward_pre_hook = forward_pre_hook
        self.forward_hook = forward_hook
        self.backward_hook = backward_hook
        self.hooks_handler = {}

    def on_loop_begin(self, model, **kwargs):
        if self.forward_hook or self.forward_pre_hook or self.backward_hook:
            if self.modules:
                if self.regex:
                    gen = match_children_modules_by_regex
                else:
                    gen = match_children_modules_by_name

                for n, m in gen(model, self.modules, strict=self.strict):
                    hook_handler = register_hook_on_leaf(
                        model=m,
                        prefix=n,
                        forward_pre_hook=self.forward_pre_hook,
                        forward_hook=self.forward_hook,
                        backward_hook=self.backward_hook,
                    )
                    self.hooks_handler.update({n: hook_handler})
            else:
                self.hooks_handler = register_hook_on_leaf(
                    model=model,
                    forward_pre_hook=self.forward_pre_hook,
                    forward_hook=self.forward_hook,
                    backward_hook=self.backward_hook,
                )

    def on_loop_end(self, **kwargs):
        self.remove_handler(self.hooks_handler)

    def remove_handler(self, handlers):

        if isinstance(handlers, dict):
            for _, v in handlers.items():
                self.remove_handler(v)
        elif isinstance(handlers, (list, tuple)):
            for handler in handlers:
                self.remove_handler(handler)
        elif isinstance(handlers, RemovableHandle):
            handlers.remove()
