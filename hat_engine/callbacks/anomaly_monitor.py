# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import contextlib
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from hat_engine.utils.package_helper import require_packages

try:
    import horizon_plugin_pytorch as horizon
    from horizon_plugin_profiler.utils.model_helper import (
        attach_qualified_name,
        remove_qualified_name,
    )
except ImportError:
    horizon = None
    attach_qualified_name = None
    remove_qualified_name = None
import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from prettytable import PrettyTable
from torch.overrides import TorchFunctionMode

from hat_engine.callbacks.hook_model import HookModel
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import is_inf, is_nan
from hat_engine.utils.deprecate import deprecated_warning
from hat_engine.utils.distributed import get_dist_info, get_local_host
from hat_engine.utils.logger import LOG_DIR, MSGColor, format_msg
from .callbacks import CallbackMixin

logger = logging.getLogger(__name__)

__all__ = ["AnomalyMonitor"]


class AnomalyMonitorBase(CallbackMixin):
    """The callback is used to monitor and log details
        if nan or inf is detected in training.

    Args:
        log_if_anomaly_grad_loss: Whether log info if nan or inf occur in
            loss and grad.
        record_only_anomaly_data: Whether to record anomaly data details,
            include module's weight/bias, input/output and grad input/output.
        record_all_data_anyway: Whether to record all data details all times.
        log_freq: Log every `log_freq` steps.
    """

    def __init__(
        self,
        log_if_anomaly_grad_loss: bool = False,
        record_anomaly_data_info: bool = True,
        record_all_data_info: bool = False,
        log_freq: int = 1,
        **kwargs,
    ) -> None:
        self.record_anomaly_data_info = record_anomaly_data_info
        self.record_all_data_info = record_all_data_info
        self.log_freq = log_freq
        self.record = record_anomaly_data_info or record_all_data_info

        if "monitor_loss" in kwargs or "monitor_grad" in kwargs:
            deprecated_warning(
                author="mengyang.duan",
                deprecation_version="2.3.1",
                removal_version="2.4.0",
                old_name="AnomalyMonitor",
                old_key="`monitor_loss` and `monitor_grad`",
                new_key="log_if_anomaly_grad_loss",
                rank_zero_only=True,
            )
        self.log_if_anomaly_grad_loss = log_if_anomaly_grad_loss
        if kwargs.get("monitor_loss", False) or kwargs.get(
            "monitor_grad", False
        ):
            self.log_if_anomaly_grad_loss = True

        self._task = None
        self._rank = None
        self._hostname = None

        self.monitor_infos = []
        self.features_infos = []

        self.all_records_infos = defaultdict(list)

    def on_backward_end(
        self,
        batch,
        step_id,
        non_finite_loss,
        model,
        **kwargs,
    ):

        if self.log_if_anomaly_grad_loss and non_finite_loss:
            msg = "loss is nan or inf "

            if self._task is not None:
                msg += f"of {self._task}"
            msg = msg + f" in Step[{step_id}]"
            self.monitor_infos.append(msg)

    def on_batch_end(self, batch, **kwargs):

        file_name = f"Step{self._global_step_id}-Host[{self.hostname}]-Rank{self.rank}-Task({self._task})"  # noqa: E501

        if self.record and len(self.all_records_infos) > 0:
            self._record_to_file(
                records=self.all_records_infos,
                prefix=file_name,
            )
            self._plotly_records(self.all_records_infos, prefix=file_name)

            self.all_records_infos.clear()

    def on_optimizer_step_begin(self, model, **kwargs):
        if self.record:
            self._record_grad(model)

    def on_optimizer_step_end(
        self,
        optimizer,
        step_id,
        non_finite_grad,
        **kwargs,
    ):

        # TODO: maybe print more info of grad in optimizer

        if self.log_if_anomaly_grad_loss and non_finite_grad:

            msg = "find nan or inf in grad"
            if self._task is not None:
                msg += f" of {self._task}"

            msg += f" in Step[{step_id}]"
            self.monitor_infos.append(msg)

    def on_step_end(self, epoch_id, step_id, global_step_id, **kwargs):

        if self.log_freq > 0 and (step_id + 1) % self.log_freq == 0:
            prefix = "Epoch[%d] Step[%d] GlobalStep[%d]: " % (
                epoch_id,
                step_id,
                global_step_id,
            )
            if self.log_if_anomaly_grad_loss and len(self.monitor_infos) > 0:
                self._log_monitor_message(
                    monitor_infos=self.monitor_infos,
                    prefix=prefix,
                )
                self.monitor_infos.clear()

    def on_step_begin(self, epoch_id, step_id, global_step_id, **kwargs):
        self._epoch_id = epoch_id
        self._step_id = step_id
        self._global_step_id = global_step_id

    def on_batch_begin(self, batch, **kwargs):
        self._task = "batch"

        if isinstance(batch, Tuple) and len(batch) == 2:
            task_name = batch[1]
        else:
            task_name = None

        batch_idx = kwargs.get("batch_idx", None)
        if batch_idx is not None:
            self._task += f"_{batch_idx}"

        if task_name is not None:
            self._task = f"_{task_name}"

    def _log_monitor_message(self, monitor_infos, prefix=""):
        if len(monitor_infos) == 0:
            return
        if len(self.monitor_infos) > 0:
            msg = "\n".join(self.monitor_infos)
            msg = prefix + "\n" + msg
            logger.error(format_msg(msg, MSGColor.RED))
            self.monitor_infos.clear()

    def _record_to_file(
        self, records: List[Dict], prefix: Optional[str] = None
    ):

        output_dir = os.path.join(LOG_DIR, "anomaly_monitor")
        os.makedirs(output_dir, exist_ok=True)

        f_string = ""
        for tag, values in records.items():
            f_string += f"\n{tag}: \n"

            table = PrettyTable()

            if len(values) == 0:
                return
            table.field_names = list(values[0].keys())
            for data in values:
                table.add_row(list(data.values()))
            f_string += table.get_string()
            f_string += "\n"

        with open(os.path.join(output_dir, f"{prefix}.txt"), "w") as f:
            f.write(f_string)

    def _plotly_records(self, records, prefix):

        output_dir = os.path.join(LOG_DIR, "plot")
        with contextlib.suppress(FileExistsError):
            os.makedirs(output_dir, exist_ok=True)

        all_tags = list(self.all_records_infos.keys())

        fig = make_subplots(
            rows=len(all_tags),
            cols=1,
            subplot_titles=[f"{prefix}: {tag}" for tag in all_tags],
            vertical_spacing=0.1,
        )

        for idx, values in enumerate(records.values(), start=1):
            module_names, min_values, max_values = [], [], []
            for data in values:
                if "num_batches_tracked" in data["data_category"]:
                    continue
                module_names.append(
                    data["module_name"] + data["data_category"]
                )
                min_values.append(data["min"])
                max_values.append(data["max"])

            # per step cost
            fig.add_trace(
                go.Scatter(
                    x=module_names,
                    y=min_values,
                    showlegend=True,
                ),
                row=idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=module_names,
                    y=max_values,
                    showlegend=True,
                ),
                row=idx,
                col=1,
            )
            fig.update_xaxes(
                title_text="module name", tickangle=-45, row=idx, col=1
            )
            fig.update_yaxes(title_text="value", row=1, col=idx)

        fig.update_layout(height=2000 * len(all_tags))

        file_name = f"{prefix}.html"

        pic_path = os.path.join(output_dir, file_name)
        try:
            fig.write_html(pic_path)
        except Exception as e:
            logger.error(f"Error saving {pic_path}: {e}")

    def record_data(
        self,
        data: Any,
        module: Union[nn.Module, str],
        data_category: str,
        record_tag: str,
    ):
        if data is None:
            return

        if isinstance(module, nn.Module):
            if horizon is not None:
                ff_method = (
                    f"[{module._last_called_method_name}]"
                    if type(module)
                    in (
                        horizon.nn.qat.FloatFunctional,
                        horizon.nn.quantized.FloatFunctional,
                    )
                    else ""
                )
            else:
                ff_method = ""

            module_name = module._qualified_name + ff_method
        else:
            module_name = module

        tensor_data_types = (torch.Tensor,)
        if horizon is not None:
            tensor_data_types += (horizon.QTensor,)
        if isinstance(data, tensor_data_types):
            if not data.numel() > 0:
                return

            if horizon is not None and isinstance(data, horizon.QTensor):
                q_scale = data.q_scale()
                data = data.dequantize()
            else:
                q_scale = None

            data = data.float()
            has_nan = is_nan(data)
            has_inf = is_inf(data)
            record = {
                "module_name": module_name,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "data_category": data_category,
                "min": round(data.min().item(), 5),
                "max": round(data.max().item(), 5),
                "mean": round(data.mean().item(), 5),
                "var": round(data.var().item(), 5),
                "scale": q_scale,
                "dtype": data.dtype,
            }

            if isinstance(module, nn.Module):
                record["module_type"] = type(module)

            is_nan_or_inf = has_nan or has_inf
            if self.record_all_data_info or (
                is_nan_or_inf and self.record_anomaly_data_info
            ):
                self.all_records_infos[record_tag].append(record)

            if is_nan_or_inf:
                logger.error(
                    "===> NonFinite on global step"
                    f" [{self._global_step_id}]:\n {record}\n"
                )

        elif isinstance(data, (tuple, list)):
            if len(data) == 1:
                self.record_data(
                    data[0],
                    module,
                    f"{data_category}",
                    record_tag=record_tag,
                )
            else:
                for i, d in enumerate(data):
                    self.record_data(
                        d,
                        module,
                        f"{data_category}-{i}",
                        record_tag=record_tag,
                    )
        elif isinstance(data, (int, float)):
            record = {
                "module_name": module,
                "has_nan": False,
                "has_inf": False,
                "data_category": data_category,
                "min": data,
                "max": data,
                "mean": data,
                "var": data,
                "scale": None,
                "dtype": type(data),
            }
            self.all_records_infos[record_tag].append(record)

    def _record_grad(self, model):
        for _name, _param in model.named_parameters():
            if _param.requires_grad and _param.grad is not None:
                self.record_data(_param.grad, _name, "grad", record_tag="grad")

    @property
    def rank(self):
        if self._rank is None:
            self._rank, _ = get_dist_info()
        return self._rank

    @property
    def hostname(self):
        if self._hostname is None:
            self._hostname = get_local_host()
        return self._hostname


@OBJECT_REGISTRY.register
class AnomalyMonitor(AnomalyMonitorBase, HookModel):  # noqa: D205,D400
    """The callback is used to monitor and log details
        if nan or inf is detected in training.

    Args:
        log_if_anomaly_grad_loss: Whether log info if nan or inf occur in
            loss and grad.
        record_only_anomaly_data: Whether to record anomaly data details,
            include module's weight/bias, input/output and grad input/output.
        record_all_data_anyway: Whether to record all data details all times.
        log_freq: Log every `log_freq` steps.
        modules: modules to add hook.
        regex: Whether to match by regex. if not, match by module name.
        strict: Whether regular expression is required to be all matched.
    """

    def __init__(
        self,
        log_if_anomaly_grad_loss: bool = False,
        record_anomaly_data_info: bool = True,
        record_all_data_info: bool = False,
        log_freq: int = 1,
        modules: Optional[List[str]] = None,
        regex: bool = True,
        strict: bool = False,
        **kwargs,
    ) -> None:
        AnomalyMonitorBase.__init__(
            self,
            log_if_anomaly_grad_loss,
            record_anomaly_data_info,
            record_all_data_info,
            log_freq,
            **kwargs,
        )

        if self.record:
            forward_hook_func = self._forward_hook
            backward_hook_func = None
        else:
            forward_hook_func = None
            backward_hook_func = None

        HookModel.__init__(
            self,
            modules=modules,
            regex=regex,
            strict=strict,
            forward_hook=forward_hook_func,
            backward_hook=backward_hook_func,
        )

    def on_loop_begin(self, model, **kwargs):
        if self.record:
            swap_ff_with_horizonff(model)
            if attach_qualified_name is not None:
                attach_qualified_name(model, True)
        super().on_loop_begin(model, **kwargs)

    def on_loop_end(self, model, **kwargs):
        if self.record and remove_qualified_name is not None:
            remove_qualified_name(model)
        super().on_loop_end(**kwargs)

    def _forward_hook(self, module, input, output):
        # record input
        self.record_data(input, module, "input", record_tag="module_input")

        # record parameters and buffers, include weight/bias...
        for name, param in module.named_parameters():
            if "." not in name:
                self.record_data(param, module, name, record_tag="weights")
        for name, buf in module.named_buffers():
            # skip scale and zero_point
            if all([x not in name for x in (".", "scale", "zero_point")]):
                self.record_data(buf, module, name, record_tag="weights")

        # record output
        self.record_data(output, module, "output", record_tag="module_output")

    def _backward_hook(self, module, grad_in, grad_out):

        # record grad input and output
        self.record_data(grad_in, module, "grad_in")
        self.record_data(grad_out, module, "grad_out")


@require_packages("horizon_plugin_pytorch")
def swap_ff_with_horizonff(model: torch.nn.Module) -> None:
    r"""Refine this docstring in the future.

    Swap torch.nn.quantized.FloatFunctional with
    horizon_plugin_pytorch.nn.quantized.FloatFunctional,
    which is wrapped with horizon fx wrapper
    """
    modules_to_swap = []
    for name, module in model.named_children():
        if isinstance(
            module,
            torch.nn.quantized.FloatFunctional,
        ):
            modules_to_swap.append(name)
        else:
            swap_ff_with_horizonff(module)

    for name in modules_to_swap:
        setattr(
            model,
            name,
            horizon.nn.quantized.FloatFunctional(),
        )


class FunctionTraceMode(TorchFunctionMode):
    def __init__(self, anomaly_monitor):
        super().__init__()
        self.anomaly_monitor = anomaly_monitor
        self.skip_record_ops = {
            "__get__",
            "dim",
            "clone",
            "size",
            "view",
            "detach",
            "shape",
        }
        self.basic_ops = ["mul_", "add_", "sub_", "div_"]

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        output = func(*args, **kwargs)
        if func.__name__ in self.skip_record_ops:
            return output
        find_weight = False
        for idx, data in enumerate(args):
            if type(data) == torch.Tensor:
                self.anomaly_monitor.record_data(
                    data,
                    ".".join(
                        [self.anomaly_monitor.current_module, func.__name__]
                    ),
                    f"input-{idx}",
                    record_tag="module_input",
                )
            elif type(data) == torch.nn.Parameter:
                find_weight = True
                self.anomaly_monitor.record_data(
                    data,
                    self.anomaly_monitor.current_module,
                    "weight" if not find_weight else "bias",
                    record_tag="params",
                )
            elif func.__name__ in self.basic_ops:
                self.anomaly_monitor.record_data(
                    data,
                    ".".join(
                        [self.anomaly_monitor.current_module, func.__name__]
                    ),
                    "scalar-input",
                    record_tag="module_input",
                )

        self.anomaly_monitor.record_data(
            output,
            ".".join([self.anomaly_monitor.current_module, func.__name__]),
            "output",
            record_tag="module_output",
        )
        return output


@OBJECT_REGISTRY.register
class FloatAnomalyMonitor(AnomalyMonitorBase):  # noqa: D205,D400
    """The callback is used to monitor and log details
        if nan or inf is detected in training.

    Args:
        log_if_anomaly_grad_loss: Whether log info if nan or inf occur in
            loss and grad.
        record_only_anomaly_data: Whether to record anomaly data details,
            include module's weight/bias, input/output and grad input/output.
        record_all_data_anyway: Whether to record all data details all times.
        log_freq: Log every `log_freq` steps.
    """

    def __init__(
        self,
        log_if_anomaly_grad_loss: bool = False,
        record_anomaly_data_info: bool = True,
        record_all_data_info: bool = False,
        log_freq: int = 1,
    ) -> None:
        super().__init__(
            log_if_anomaly_grad_loss,
            record_anomaly_data_info,
            record_all_data_info,
            log_freq,
        )
        self.func_trace_mode = FunctionTraceMode(self)
        self.prev_module = None
        self.current_module = None

    def register_module_context_hooks(self, model):

        for module_name, module in model.named_modules():

            def pre_hook(mod, input, name=module_name):
                self.prev_module = self.current_module
                self.current_module = name

            def post_hook(mod, input, output):
                self.current_module = self.prev_module
                self.prev_module, _, _ = self.prev_module.rpartition(".")

            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)

    def on_loop_begin(self, model, **kwargs):
        self.register_module_context_hooks(model)

        def func_trace_decorator(func):
            def wrapper(*args, **kwargs):
                with self.func_trace_mode:
                    output = func(*args, **kwargs)
                return output

            return wrapper

        model.forward = func_trace_decorator(model.forward)
