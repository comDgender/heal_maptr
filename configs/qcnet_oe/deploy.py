import os

from common import ckpt_dir, march, qat_mode, task_name
from model import deploy_inputs, deploy_model
from qconfig import qat_qconfig_setter

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            example_inputs=deploy_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)

real_deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            example_inputs=deploy_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-last.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)
enable_vpu = True

hbir_save_dir = ckpt_dir
compile_dir = os.path.join(hbir_save_dir, "compile")


hbir_exporter = dict(
    type="HbirExporter",
    model=deploy_model,
    model_convert_pipeline=real_deploy_model_convert_pipeline,
    example_inputs=deploy_inputs,
    save_path=hbir_save_dir,
    model_name=task_name,
    march=march,
    enable_vpu=True,
)

hbir_compiler = dict(
    type="HbirCompiler",
    model_path=os.path.join(ckpt_dir, "qat.bc"),
    out_path=os.path.join(compile_dir, "model.hbm"),
    march=march,
    opt=2,
    jobs=32,
    max_time_per_fc=0.0,
    preserve_quant_dequant=False,
    input_source=["ddr"],
    cache_path="",
    enable_vpu=enable_vpu,
)

calops_cfg = dict(method="hook")
