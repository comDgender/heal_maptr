# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import copy
import logging
import math
import multiprocessing as mp
from collections.abc import Mapping, Sequence
from multiprocessing.pool import Pool
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch import device
from torch.utils._pytree import tree_flatten

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list, convert_numpy, to_cuda
from hat_engine.utils.distributed import get_dist_info
from hat_engine.utils.package_helper import require_packages
from .ir_module import IrModule

try:
    from hbm_infer.hbm_rpc_session import HbmRpcSession
    from hbm_infer.hbm_rpc_session_flexible import HbmHandle, HbmRpcServer
    from hbm_infer.hbm_rpc_session_flexible import (
        HbmRpcSession as HbmRpcSessionFlexible,
    )
    from hbm_infer.hbm_rpc_session_flexible import (
        deinit_hbm,
        deinit_server,
        init_hbm,
        init_server,
    )
except ImportError:
    HbmRpcSession = None
    HbmRpcSessionFlexible = None
    HbmHandle = None
    HbmRpcServer = None
    deinit_hbm = None
    deinit_server = None
    init_hbm = None
    init_server = None


__all__ = [
    "HbmModule",
    "BatchedHbmModule",
    "PipelineHbmServer",
    "PipelineHbmModule",
    "close_hbm_servers",
]

logger = logging.getLogger(__name__)

_BOARD_CORE_NUM = 4
_BOARD_L2M_SIZE = 24


def close_hbm_servers(model: torch.nn.Module) -> None:
    """Explicitly close HBM RPC servers under ``model``."""
    closed = []
    for mod in model.modules():
        if isinstance(mod, (HbmModule, BatchedHbmModule)):
            mod.close_server()
        elif isinstance(mod, PipelineHbmModule):
            server = mod.hbm_server
            if not any(server is done for done in closed):
                closed.append(server)
                server.close_server()
            # A closed server must not be reused by a later run.
            PipelineHbmModule.hbm_server = None


@OBJECT_REGISTRY.register
class HbmModule(IrModule):
    """Inference module of hbir.

    Args:
         host: The network address of the host
         model_path: Path of ir model file.
         hbm_rpc_server: Instance of the HbmRpcServer class for flexible mode.
         hbm_handle: An instance of the HbmHandle class for flexible mode.
         return_tensor: Whether to return torch tensor.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
         remote_root: The root directory path for remote operations.
         with_profile: Whether hbm profile is used.
         quant_input: Whether the input data should be quantized.
         dequant_output: Whether the output result should be dequantized.
         compress_option: The type of compress, Can be "IN" "INOUT" "NONE".
    """

    @require_packages("hbm_infer>=3.2.5")
    def __init__(
        self,
        host: str = "",
        model_path: str = "",
        hbm_rpc_server: HbmRpcServer = None,
        hbm_handle: HbmHandle = None,
        return_tensor: bool = True,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
        remote_root: str = "/map/hbm_infer/",
        with_profile: bool = False,
        quant_input: bool = True,
        dequant_output: bool = True,
        compress_option: str = "NONE",
    ):
        super().__init__(
            model_path=model_path,
            reformat_input_func=reformat_input_func,
            reformat_output_func=reformat_output_func,
        )
        self.host = host
        self.bpu_burn = False
        self.with_profile = with_profile
        self.debug = False
        # self.debug = True

        if hbm_rpc_server is not None:
            assert hbm_handle is not None
            assert remote_root in ["", hbm_rpc_server.remote_root]
            self.remote_root = hbm_rpc_server.remote_root
            self.model = HbmRpcSessionFlexible(
                hbm_handle=hbm_handle,
                hbm_rpc_server=hbm_rpc_server,
                with_profile=with_profile,
                debug=self.debug,
                compress_option=compress_option,
            )
        else:
            self.remote_root = remote_root
            self.model = HbmRpcSession(
                host=self.host,
                local_hbm_path=model_path,
                remote_root=remote_root,
                with_profile=with_profile,
                debug=self.debug,
                compress_option=compress_option,
            )
        self.model.show_input_output_info()
        self.return_tensor = return_tensor

        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

        self.quant_input = quant_input
        self.dequant_output = dequant_output
        self.allow_ambiguous_input = False
        self.input_info = self.model.get_input_info()
        self.output_info = self.model.get_output_info()
        # import hat_engine.utils.forkedpdb as pdb;pdb.set_trace()
        if not all(["quantizeAxis" in v for k, v in self.input_info.items()]):
            logger.warning(
                f"Not all inputs has QuantiScaleInfo"
                f"(quantizeAxis, scale, zero_point), "
                f"while dequant_input is True. "
                f"{['quantizeAxis' in v for k, v in self.input_info.items()]}"
            )

        if not all(["quantizeAxis" in v for k, v in self.output_info.items()]):
            logger.warning(
                f"Not all outputs has QuantiScaleInfo"
                f"(quantizeAxis, scale, zero_point), "
                f"while dequant_output is True. "
                f"{['quantizeAxis' in v for k, v in self.output_info.items()]}"
            )

    def show_input_output_info(self):
        return self.model.show_input_output_info()

    def filter_input_data(self, data):
        def _is_not_container(obj):
            return not any(
                [
                    isinstance(obj, c_type)
                    for c_type in [list, dict, set, tuple]
                ]
            )

        def _dfs_replace_nv12(data):
            if not isinstance(data, dict):
                return data
            maybe_nv12 = [
                k
                for k in data
                if (k.endswith("_y") or k.endswith("_uv"))
                and _is_not_container(data[k])
            ]
            ret_data = {}
            for k in data:
                if (
                    f"{k}_y" in maybe_nv12
                    and f"{k}_uv" in maybe_nv12
                    and _is_not_container(data[k])
                ):
                    ret_data[f"{k}_y"], ret_data[f"{k}_uv"] = (
                        data[f"{k}_y"],
                        data[f"{k}_uv"],
                    )
                else:
                    ret_data[k] = _dfs_replace_nv12(data[k])
            return ret_data

        model_inputs = self.model.get_input_info()
        # 1. All inputs is ready.
        if all([inp in data for inp in model_inputs.keys()]):
            # logger.info("All need keys are found in input data.")
            return {inp: data[inp] for inp in model_inputs.keys()}

        if self.allow_ambiguous_input:
            # 2. Length matched after flatten.
            flat_inputs, _ = tree_flatten(data)
            if len(flat_inputs) == len(model_inputs):
                return {
                    inp: flat_inputs[i]
                    for i, inp in enumerate(model_inputs.keys())
                }

            # 3. Length matched after replacement + flatten.
            flat_inputs, _ = tree_flatten(_dfs_replace_nv12(data))
            if len(flat_inputs) == len(model_inputs):
                return {
                    inp: flat_inputs[i]
                    for i, inp in enumerate(model_inputs.keys())
                }

        raise TypeError(
            "Input data not matched the hbm inputs, check "
            "the input & output info with show_input_output_info()"
        )

    def get_top_device(self, data):
        top_device = torch.device("cpu")
        flat_inputs, _ = tree_flatten(data)
        for d in flat_inputs:
            if isinstance(d, torch.Tensor) and "cuda" in d.device.type:
                top_device = d.device
        return top_device

    def up_to_top_device(self, data, top_device):
        # 0. peel off list & tuple with length == 1
        for k in data:
            if isinstance(data[k], (list, tuple)) and len(data[k]) == 1:
                data[k] = data[k][0]

        for k in data:
            # 1. to torch.Tensor
            if isinstance(data[k], np.ndarray):
                data[k] = torch.from_numpy(data[k])

            # 2. to top_device
            if isinstance(data[k], torch.Tensor):
                if "cuda" in top_device.type:
                    data[k] = to_cuda(data[k], device=top_device)
            else:
                raise TypeError(
                    "Input data not matched the hbm inputs, check the "
                    "input & output info with show_input_output_info()"
                )

        return data

    def quant_input_data(self, data):
        for k in self.input_info.keys():
            if self.input_info[k]["quanti_type"] == "QUANTI_TYPE_NONE":
                continue

            axis = self.input_info[k]["quantizeAxis"]
            new_shape = [1] * len(data[k].shape)
            new_shape[axis] = len(self.input_info[k]["scale_data"])

            tensor_type = self.input_info[k]["tensor_type"]
            _min, _max = {
                "DATA_TYPE_S8": [-128, 127],
                "DATA_TYPE_U8": [0, 255],
                "DATA_TYPE_S16": [-32768, 32767],
            }[tensor_type]

            if isinstance(data[k], torch.Tensor):
                _scale = torch.tensor(
                    self.input_info[k]["scale_data"], device=data[k].device
                ).reshape(new_shape)
                _zero_point = torch.tensor(
                    self.input_info[k]["zero_point_data"],
                    device=data[k].device,
                ).reshape(new_shape)
                data[k] = torch.clamp(
                    torch.round(data[k] / _scale) + _zero_point, _min, _max
                )
                data[k] = data[k].to(self.model.hbm_map_torch[tensor_type][0])
            elif isinstance(data[k], np.ndarray):
                _scale = np.array(self.input_info[k]["scale_data"]).reshape(
                    new_shape
                )
                _zero_point = np.array(
                    self.input_info[k]["zero_point_data"]
                ).reshape(new_shape)
                data[k] = np.clip(
                    np.round(data[k] / _scale) + _zero_point, _min, _max
                )
                data[k] = data[k].astype(
                    self.model.hbm_map_numpy[tensor_type][0]
                )
            else:
                raise TypeError(f"data cannot support {type(data[k])}")
        return data

    def dequant_output_data(self, data):
        for k in self.output_info.keys():
            if "quantizeAxis" not in self.output_info[k]:
                continue

            axis = self.output_info[k]["quantizeAxis"]
            new_shape = [1] * len(data[k].shape)
            new_shape[axis] = len(self.output_info[k]["scale_data"])

            if isinstance(data[k], torch.Tensor):
                _scale = torch.tensor(
                    self.output_info[k]["scale_data"], device=data[k].device
                ).reshape(new_shape)
                _zero_point = torch.tensor(
                    self.output_info[k]["zero_point_data"],
                    device=data[k].device,
                ).reshape(new_shape)
                data[k] = (data[k] - _zero_point) * _scale
            elif isinstance(data[k], np.ndarray):
                _scale = np.array(self.output_info[k]["scale_data"]).reshape(
                    new_shape
                )
                _zero_point = np.array(
                    self.output_info[k]["zero_point_data"]
                ).reshape(new_shape)
                data[k] = (data[k] - _zero_point) * _scale
            else:
                raise TypeError(f"data cannot support {type(data[k])}")
        return data

    def check_input_impl(self, data):
        self.device = self.get_top_device(data)
        data = self.filter_input_data(data)
        data = self.up_to_top_device(data, self.device)

        if self.quant_input:
            data = self.quant_input_data(data)

        return data

    def check_output_impl(self, data):
        return_data = data
        if self.dequant_output:
            return_data = self.dequant_output_data(return_data)

        if not self.return_tensor:
            return_data = convert_numpy(return_data)

        if len(return_data) == 1:
            return_data = return_data[list(return_data.keys())[0]]

        if (
            self.return_tensor
            and self.device is not None
            and "cuda" in self.device.type
        ):
            return_data = to_cuda(return_data, device=self.device)

        return return_data

    def forward_impl(self, data):
        while True:
            if self.with_profile is True:
                (
                    output,
                    server_duration,
                    infer_duration,
                    s2r_duration,
                    pre_duration,
                    post_duration,
                ) = self.model(data)
            else:
                output = self.model(data)

            if not self.bpu_burn:
                break
        return output

    def cpu(self):
        return self

    def cuda(self, device: Optional[Union[int, device]] = None):
        logger.warnings("Hbm does not support GPU now. Use CPU instead.")
        return self

    def __del__(self) -> None:
        try:
            self.close_server()
        except Exception:
            pass

    def close_server(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            model.close_server()


hbm_module = None


def infer_task(data):
    global hbm_module
    return hbm_module(data)


def close_server_task(process_id):
    logger.info(f"Process: {process_id} closing server ...")
    global hbm_module
    hbm_module.close_server()


def hbm_module_init_flexible(
    model_path,
    hbm_rpc_server,
    hbm_handle,
    return_tensor,
    reformat_input_func,
    reformat_output_func,
    with_profile,
    quant_input,
    dequant_output,
    compress_option,
):
    params = {
        "model_path": model_path,
        "hbm_rpc_server": hbm_rpc_server,
        "hbm_handle": hbm_handle,
        "remote_root": "",
        "return_tensor": return_tensor,
        "reformat_input_func": reformat_input_func,
        "reformat_output_func": reformat_output_func,
        "with_profile": with_profile,
        "quant_input": quant_input,
        "dequant_output": dequant_output,
        "compress_option": compress_option,
    }
    global hbm_module
    hbm_module = HbmModule(**params)


def hbm_module_init(
    model_path,
    host,
    remote_root,
    return_tensor,
    reformat_input_func,
    reformat_output_func,
    with_profile,
    quant_input,
    dequant_output,
    compress_option,
):
    params = {
        "model_path": model_path,
        "host": host,
        "remote_root": remote_root,
        "return_tensor": return_tensor,
        "reformat_input_func": reformat_input_func,
        "reformat_output_func": reformat_output_func,
        "with_profile": with_profile,
        "quant_input": quant_input,
        "dequant_output": dequant_output,
        "compress_option": compress_option,
    }
    global hbm_module
    hbm_module = HbmModule(**params)


@OBJECT_REGISTRY.register
class BatchedHbmModule(IrModule):
    """Batch inference module of hbm.

    Args:
         batched_keys: keys of batched in data.
         host: The network address of the host
         model_path: Path of ir model file.
         return_tensor: Whether to return torch tensor.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
         num_processes: Number of processes for batched hbir predict.
         remote_root: The root directory path for remote operations.
         with_profile: Whether hbm profile is used.
         quant_input: Whether the input data should be quantized.
         dequant_output: Whether the output result should be dequantized.
         compress_option: The type of compress, Can be "IN" "INOUT" "NONE".
         is_flexible: Whther use flexible rpcsession.
    """

    @require_packages("hbm_infer>=3.2.5")
    def __init__(
        self,
        batched_keys: List[str],
        host: str,
        model_path: str,
        return_tensor: bool = True,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
        num_processes: int = -1,
        remote_root: str = "/map/hbm_infer/",
        with_profile: bool = False,
        quant_input: bool = True,
        dequant_output: bool = True,
        compress_option: str = "NONE",
        is_flexible: bool = False,
    ):
        self.batched_keys = _as_list(batched_keys)
        self.num_processes = num_processes
        if self.num_processes > 8:
            logger.warning(
                "Be aware of the resource cost when num_processes > 8."
            )
        self.is_flexible = is_flexible

        super().__init__(
            model_path=model_path,
            reformat_input_func=reformat_input_func,
            reformat_output_func=reformat_output_func,
        )

        if self.num_processes > 0:
            if self.is_flexible:
                self.hbm_rpc_server = init_server(
                    host=host, remote_root=remote_root
                )
                self.hbm_handle = init_hbm(
                    local_hbm_path=model_path,
                    hbm_rpc_server=self.hbm_rpc_server,
                )
                self.pool = Pool(
                    processes=num_processes,
                    initializer=hbm_module_init_flexible,
                    initargs=(
                        model_path,
                        self.hbm_rpc_server,
                        self.hbm_handle,
                        return_tensor,
                        reformat_input_func,
                        reformat_output_func,
                        with_profile,
                        quant_input,
                        dequant_output,
                        compress_option,
                    ),
                    context=mp.get_context("spawn"),
                )
            else:
                self.pool = Pool(
                    processes=num_processes,
                    initializer=hbm_module_init,
                    initargs=(
                        model_path,
                        host,
                        remote_root,
                        return_tensor,
                        reformat_input_func,
                        reformat_output_func,
                        with_profile,
                        quant_input,
                        dequant_output,
                        compress_option,
                    ),
                    context=mp.get_context("spawn"),
                )
        else:
            self.model = HbmModule(
                model_path=model_path,
                host=host,
                remote_root=remote_root,
                return_tensor=return_tensor,
                reformat_input_func=reformat_input_func,
                reformat_output_func=reformat_output_func,
                with_profile=with_profile,
                quant_input=quant_input,
                dequant_output=dequant_output,
                compress_option=compress_option,
            )

    def show_input_output_info(self):
        return self.model.show_input_output_info()

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

        assert len({len(batched_data[key]) for key in self.batched_keys}) == 1

        sub_data_list = []
        for i in range(batch_size):
            tmp_data = copy.deepcopy(ori_data)
            tmp_data.update(
                {key: batched_data[key][i] for key in self.batched_keys}
            )
            sub_data_list.append(tmp_data)

        if self.num_processes > 0:
            batch_results = list(self.pool.map(infer_task, sub_data_list))
        else:
            batch_results = [
                self.model(sub_data) for sub_data in sub_data_list
            ]

        if isinstance(batch_results[0], Sequence):
            batch_results = map(self._merge_batch_data, zip(*batch_results))
            return tuple(batch_results)
        return self._merge_batch_data(batch_results)

    def cpu(self):
        return self

    def cuda(self, device: Optional[Union[int, device]] = None):
        logger.warning("Hbm does not support GPU now. Use CPU instead.")
        return self

    def __del__(self) -> None:
        try:
            self.close_server()
        except Exception:
            pass

    def close_server(self) -> None:
        if getattr(self, "num_processes", 0) > 0:
            pool = getattr(self, "pool", None)
            if pool is not None:
                _ = list(
                    pool.map(close_server_task, range(self.num_processes))
                )
            if getattr(self, "is_flexible", False):
                deinit_server(self.hbm_rpc_server)
                deinit_hbm(self.hbm_handle)
        else:
            model = getattr(self, "model", None)
            if model is not None:
                model.close_server()


class PipelineHbmServer(object):
    """One RPC server, hbm handle and session shared by several graphs.

    Args:
         host: The network address of the board.
         hbm_path: Path(s) of the hbm file holding every graph.
         core_id: BPU core id(s), -1 means any core.
         remote_environment: Environment variables exported on the board.
    """

    def __init__(
        self,
        host: str,
        hbm_path: Union[str, List[str]],
        core_id: Union[int, List[int]] = -1,
        remote_environment: Optional[dict] = None,
    ):
        self.hbm_server = HbmRpcServer(host=host)
        self.hbm_handle = HbmHandle(
            local_hbm_path=hbm_path,
            hbm_rpc_server=self.hbm_server,
        )
        self.session = HbmRpcSessionFlexible(
            hbm_handle=self.hbm_handle,
            hbm_rpc_server=self.hbm_server,
            core_id=core_id,
            remote_environment=remote_environment or {},
        )

    def get_session(self):
        return self.session

    def close_server(self) -> None:
        self.hbm_server.deinit()
        self.hbm_handle.deinit()
        self.session.close_server()


@OBJECT_REGISTRY.register
class PipelineHbmModule(IrModule):
    """Inference module of one graph inside a multi-graph hbm.

    Args:
         host: The network address of the board, or a list to pick from.
         hbm_path: Path(s) of the hbm file holding every graph.
         hbm_name: Name of the graph this module infers against.
         use_l2_cache: Whether to assign board cores and L2 to this process.
         core_num: Number of BPU cores the hbm was compiled for.
         reformat_input_func: Callable function to reformat model inputs.
         reformat_output_func: Callable function to reformat model output.
         quant_input: Whether the input data should be quantized.
         dequant_output: Whether the output result should be dequantized.
    """

    hbm_server = None

    @require_packages("hbm_infer>=3.2.7")
    def __init__(
        self,
        host: Union[str, List[str]],
        hbm_path: Union[str, List[str]],
        hbm_name: str,
        use_l2_cache: bool = False,
        core_num: int = 1,
        reformat_input_func: Optional[Callable] = None,
        reformat_output_func: Optional[Callable] = None,
        quant_input: bool = True,
        dequant_output: bool = True,
    ):
        model_path = hbm_path[0] if isinstance(hbm_path, list) else hbm_path
        super().__init__(
            model_path=model_path,
            reformat_input_func=reformat_input_func,
            reformat_output_func=reformat_output_func,
        )
        self.host = host
        self.use_l2_cache = use_l2_cache
        self.core_num = core_num
        self.hbm_name = hbm_name
        self.quant_input = quant_input
        self.dequant_output = dequant_output
        self.return_tensor = True

        if PipelineHbmModule.hbm_server is None:
            host, core_id, remote_environment = self.get_host()
            PipelineHbmModule.hbm_server = PipelineHbmServer(
                host=host,
                hbm_path=hbm_path,
                core_id=core_id,
                remote_environment=remote_environment,
            )
        self.hbm_server = PipelineHbmModule.hbm_server
        self.session = self.hbm_server.get_session()

        self.input_info = self.session.get_input_info(self.hbm_name)
        self.output_info = self.session.get_output_info(self.hbm_name)

    def get_host(self):
        rank, world_size = get_dist_info()
        hosts = _as_list(self.host)
        num_proc_per_host = max(1, math.ceil(world_size / len(hosts)))
        host = hosts[(rank // num_proc_per_host) % len(hosts)]
        local_proc_id = rank % num_proc_per_host

        if self.use_l2_cache:
            first = local_proc_id * self.core_num
            core_id = list(range(first, first + self.core_num))
            # L2M is per core: a model only sees the share of the cores it
            # runs on, so spreading wider than core_num starves it.
            share = _BOARD_L2M_SIZE // num_proc_per_host // self.core_num
            sizes = [0] * _BOARD_CORE_NUM
            for core in core_id:
                sizes[core] = share
            return (
                host,
                core_id,
                {
                    "HB_DNN_USER_DEFINED_L2M_SIZES": ":".join(
                        str(s) for s in sizes
                    )
                },
            )
        return host, -1, {}

    def show_input_output_info(self):
        return self.session.show_input_output_info(self.hbm_name)

    def get_session(self):
        return self.session

    def get_top_device(self, data):
        top_device = torch.device("cpu")
        flat_inputs, _ = tree_flatten(data)
        for d in flat_inputs:
            if isinstance(d, torch.Tensor) and "cuda" in d.device.type:
                top_device = d.device
        return top_device

    def up_to_top_device(self, data, top_device):
        for k in data:
            if isinstance(data[k], (list, tuple)) and len(data[k]) == 1:
                data[k] = data[k][0]

        for k in data:
            if isinstance(data[k], np.ndarray):
                data[k] = torch.from_numpy(data[k])

            if isinstance(data[k], torch.Tensor):
                if "cuda" in top_device.type:
                    data[k] = to_cuda(data[k], device=top_device)
            else:
                raise TypeError(
                    "Input data not matched the hbm inputs, check the "
                    "input & output info with show_input_output_info()"
                )
        return data

    def filter_input_data(self, data):
        if all(name in data for name in self.input_info):
            return {name: data[name] for name in self.input_info}
        raise TypeError(
            "Input data not matched the hbm inputs, check "
            "the input & output info with show_input_output_info()"
        )

    def quant_input_data(self, data):
        for k in self.input_info:
            tensor_type = self.input_info[k]["tensor_type"]
            if tensor_type == "DATA_TYPE_S32":
                data[k] = data[k].to(dtype=torch.int32)
            if tensor_type == "DATA_TYPE_S64":
                data[k] = data[k].to(dtype=torch.int64)
            if tensor_type == "DATA_TYPE_F16" and data[k].dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                data[k] = data[k].to(dtype=torch.float16)
            if "quantizeAxis" not in self.input_info[k]:
                continue
            if tensor_type == "DATA_TYPE_F16":
                continue

            axis = self.input_info[k]["quantizeAxis"]
            new_shape = [1] * len(data[k].shape)
            new_shape[axis] = len(self.input_info[k]["scale_data"])

            _min, _max = {
                "DATA_TYPE_S8": [-128, 127],
                "DATA_TYPE_S16": [-32768, 32767],
            }[tensor_type]

            if isinstance(data[k], torch.Tensor):
                _scale = torch.tensor(
                    self.input_info[k]["scale_data"], device=data[k].device
                ).reshape(new_shape)
                _zero_point = torch.tensor(
                    self.input_info[k]["zero_point_data"],
                    device=data[k].device,
                ).reshape(new_shape)
                data[k] = torch.clamp(
                    torch.round(data[k] / _scale) + _zero_point, _min, _max
                )
                data[k] = data[k].to(
                    self.session.hbm_map_torch[tensor_type][0]
                )
            elif isinstance(data[k], np.ndarray):
                _scale = np.array(self.input_info[k]["scale_data"]).reshape(
                    new_shape
                )
                _zero_point = np.array(
                    self.input_info[k]["zero_point_data"]
                ).reshape(new_shape)
                data[k] = np.clip(
                    np.round(data[k] / _scale) + _zero_point, _min, _max
                )
                data[k] = data[k].astype(
                    self.session.hbm_map_numpy[tensor_type][0]
                )
            else:
                raise TypeError(f"data cannot support {type(data[k])}")
        return data

    def dequant_output_data(self, data):
        for k in self.output_info:
            if k not in data:
                continue
            if "quantizeAxis" not in self.output_info[k]:
                continue

            axis = self.output_info[k]["quantizeAxis"]
            new_shape = [1] * len(data[k].shape)
            new_shape[axis] = len(self.output_info[k]["scale_data"])

            if isinstance(data[k], torch.Tensor):
                _scale = torch.tensor(
                    self.output_info[k]["scale_data"], device=data[k].device
                ).reshape(new_shape)
                _zero_point = torch.tensor(
                    self.output_info[k]["zero_point_data"],
                    device=data[k].device,
                ).reshape(new_shape)
                data[k] = (data[k] - _zero_point) * _scale
            elif isinstance(data[k], np.ndarray):
                _scale = np.array(self.output_info[k]["scale_data"]).reshape(
                    new_shape
                )
                _zero_point = np.array(
                    self.output_info[k]["zero_point_data"]
                ).reshape(new_shape)
                data[k] = (data[k] - _zero_point) * _scale
            else:
                raise TypeError(f"data cannot support {type(data[k])}")
        return data

    def check_input_impl(self, data):
        self.device = self.get_top_device(data)
        data = self.filter_input_data(data)
        data = self.up_to_top_device(data, self.device)
        if self.quant_input:
            data = self.quant_input_data(data)
        return data

    def check_output_impl(self, data):
        return_data = data
        if self.dequant_output:
            return_data = self.dequant_output_data(return_data)

        if (
            self.return_tensor
            and self.device is not None
            and "cuda" in self.device.type
        ):
            return_data = to_cuda(return_data, device=self.device)
        return return_data

    def forward_impl(self, data):
        return self.session(data=data, model_name=self.hbm_name)

    def cpu(self):
        return self

    def cuda(self, device: Optional[Union[int, device]] = None):
        logger.warning("Hbm does not support GPU now. Use CPU instead.")
        return self
