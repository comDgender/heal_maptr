import os

from common import ckpt_dir
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)


def get_custom_qconfig(is_calib):
    my_qconfig_setter = QconfigSetter(
        reference_qconfig=get_qconfig(
            observer=(
                observer_v2.MSEObserver
                if is_calib
                else observer_v2.MinMaxObserver
            ),
            fix_scale=False,  # not is_calib,
        ),
        templates=[
            ModuleNameTemplate({"": qint8}),
            MatmulDtypeTemplate(
                input_dtypes=[qint8, qint16],
            ),
            ConvDtypeTemplate(
                input_dtype=qint8,
                weight_dtype=qint8,
            ),
        ],
        enable_optimize=True,
        save_dir=os.path.join(ckpt_dir, "qconfig_setting"),
        custom_qconfig_mapping=None,
    )
    return my_qconfig_setter


qat_qconfig_setter = get_custom_qconfig(False)
calib_qconfig_setter = get_custom_qconfig(True)
