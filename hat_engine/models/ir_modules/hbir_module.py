# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import copy
import logging
import multiprocessing as mp
from collections.abc import Mapping, Sequence
from multiprocessing.pool import Pool
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import device

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import (
    _as_list,
    convert_numpy,
    convert_tensor,
    to_cuda,
)
from hat_engine.utils.package_helper import require_packages
from .ir_module import IrModule
from .utils import np2torch_dtype_dict

try:
    from hbdk4.compiler import load
except ImportError:
    load = None

try:
    from horizon_plugin_pytorch.quantization.hbdk4 import (
        get_hbir_input_flattener,
        get_hbir_output_unflattener,
    )
except ImportError:
    get_hbir_input_flattener = None
    get_hbir_output_unflattener = None


__all__ = ["HbirModule", "BatchedHbirModule"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class HbirModule(IrModule):
    """Inference module of hbir.

    Args:
         model_path: Path of ir model file.
         return_tensor: Whether to return torch tensor.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
    """

    @require_packages("horizon_plugin_pytorch>=1.10.3", "hbdk4")
    def __init__(
        self,
        model_path: str,
        return_tensor: bool = True,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
    ):
        super().__init__(
            model_path=model_path,
            reformat_input_func=reformat_input_func,
            reformat_output_func=reformat_output_func,
        )
        self.model = load(model_path)
        self.return_tensor = return_tensor
        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

        self.input_flattener = get_hbir_input_flattener(self.model)
        self.output_unflattener = get_hbir_output_unflattener(self.model)

    def check_input_impl(self, data):
        assert isinstance(data, Dict)
        format_data = {}
        for inp in self.model[0].inputs:
            dtype = np2torch_dtype_dict[inp.type.np_dtype]
            self.check_type_shape(inp.name, inp.type.shape, dtype, data)
            format_data[inp.name] = data[inp.name]
            self.device = data[inp.name].device
        return convert_numpy(format_data)

    def check_output_impl(self, data):
        return_data = data

        if self.return_tensor:
            return_data = convert_tensor(return_data)

        if "cuda" in self.device.type:
            return_data = to_cuda(return_data, self.device)
        return return_data

    def forward_impl(self, data):
        output = self.model.functions[0](*self.input_flattener(data))
        output = self.output_unflattener(output)
        return output

    def cpu(self):
        return self

    def cuda(self, device: Optional[Union[int, device]] = None):
        logger.warning("Hbir does not support GPU now. Use CPU instead.")
        return self


hbir_module = None


def hbir_module_init(model_path, return_tensor, input_func, output_func):
    global hbir_module
    hbir_module = HbirModule(
        model_path=model_path,
        return_tensor=return_tensor,
        reformat_input_func=input_func,
        reformat_output_func=output_func,
    )


def hbir_module_task(data):
    return hbir_module(data)


@OBJECT_REGISTRY.register
class BatchedHbirModule(HbirModule):
    """Batch inference module of hbir.

    Args:
         batched_keys: keys of batched in data.
         model_path: Path of ir model file.
         return_tensor: Whether to return torch tensor.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
         num_processes: Number of processes for batched hbir predict.
    """

    @require_packages("horizon_plugin_pytorch>=1.10.3", "hbdk4")
    def __init__(
        self,
        batched_keys: List[str],
        model_path: str,
        return_tensor: bool = True,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
        num_processes: int = -1,
    ):
        self.batched_keys = _as_list(batched_keys)
        self.num_processes = num_processes

        super().__init__(
            model_path=model_path,
            return_tensor=return_tensor,
            reformat_input_func=reformat_input_func,
            reformat_output_func=reformat_output_func,
        )

        if num_processes > 0:
            self.pool = Pool(
                processes=num_processes,
                initializer=hbir_module_init,
                initargs=(
                    model_path,
                    return_tensor,
                    reformat_input_func,
                    reformat_output_func,
                ),
                context=mp.get_context("spawn"),
            )

    def _split_batch_data(self, data):
        if isinstance(data, Sequence) and not isinstance(data, str):
            return list(data)
        elif isinstance(data, torch.Tensor):
            sub_data = torch.split(data, 1, dim=0)
            return list(sub_data)
        elif isinstance(data, np.ndarray):
            sub_data = np.split(data, data.shape[0], axis=0)
            return list(sub_data)
        else:
            raise TypeError(f"Batched data cannot support {type(data)}")

    def _merge_batch_data(self, data):
        assert isinstance(data, Sequence)

        if isinstance(data[0], torch.Tensor):
            return torch.cat(data, 0)
        elif isinstance(data[0], np.ndarray):
            return np.concatenate(data, 0)
        elif isinstance(data[0], Sequence):
            tmp_data = list(map(list, zip(*data)))
            return [self._merge_batch_data(d) for d in tmp_data]
        elif isinstance(data[0], Mapping):
            return {
                k: self._merge_batch_data([d[k] for d in data])
                for k in data[0].keys()
            }
        else:
            return data

    @torch.no_grad()
    def forward(self, data):
        ori_data = copy.deepcopy(data)
        batched_data = {}
        batch_size = -1
        for key in self.batched_keys:
            value = ori_data.pop(key)
            batched_data[key] = self._split_batch_data(value)
            batch_size = len(batched_data[key])

        assert batch_size != -1

        sub_data_list = []
        for i in range(batch_size):
            tmp_data = copy.deepcopy(ori_data)
            tmp_data.update(
                {key: batched_data[key][i] for key in self.batched_keys}
            )
            sub_data_list.append(tmp_data)

        if self.num_processes > 0:
            batch_results = list(
                self.pool.map(hbir_module_task, sub_data_list)
            )
        else:
            batch_results = list(map(super().forward, sub_data_list))

        if isinstance(batch_results[0], Sequence):
            batch_results = map(self._merge_batch_data, zip(*batch_results))
            return tuple(batch_results)
        return self._merge_batch_data(batch_results)
