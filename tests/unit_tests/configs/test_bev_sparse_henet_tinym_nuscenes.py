"""Config structure tests for bev_sparse_henet_tinym."""

from pathlib import Path

from heal.utils.config import load_config_with_optional_base

CONFIG_DIR = (
    Path(__file__).resolve().parents[3] / "configs/bev_sparse_henet_tinym"
)


def _load_config(filename: str):
    return load_config_with_optional_base(str(CONFIG_DIR / filename))


def _input_source_count(value) -> int:
    """Count input-source entries whether given as a list or a comma string."""
    if isinstance(value, str):
        return len([s for s in value.split(",") if s.strip()])
    return len(value)


def test_bev_sparse_common_task_name():
    cfg = _load_config("common.py")
    assert cfg.task_name == "bev_sparse_henet_tinym"
    assert cfg.num_classes == 10
    assert len(cfg.CLASSES) == cfg.num_classes


def test_bev_sparse_train_stages():
    cfg = _load_config("train.py")
    assert hasattr(cfg, "float_trainer")
    assert hasattr(cfg, "calibration_trainer")
    # bev_sparse has no qat training stage.
    assert not hasattr(cfg, "qat_trainer")


def test_bev_sparse_predict_stages():
    cfg = _load_config("predict.py")
    for name in (
        "float_predictor",
        "calibration_predictor",
        "hbir_predictor",
        "hbm_predictor",
    ):
        assert hasattr(cfg, name), f"missing {name}"
    assert not hasattr(cfg, "qat_predictor")


def test_bev_sparse_infer_stages():
    cfg = _load_config("infer.py")
    for name in (
        "gen_demo_cfg",
        "float_infer",
        "calibration_infer",
        "hbir_infer",
        "hbm_infer",
    ):
        assert hasattr(cfg, name), f"missing {name}"
    assert not hasattr(cfg, "qat_infer")
    assert cfg.gen_demo_cfg["data_saver"] is not None


def test_bev_sparse_deploy_artifacts():
    cfg = _load_config("deploy.py")
    assert hasattr(cfg, "hbir_exporter")
    assert hasattr(cfg, "hbir_compiler")
    assert hasattr(cfg, "calops_cfg")
    assert cfg.hbir_exporter["march"] == cfg.march
    assert cfg.hbir_compiler["march"] == cfg.march
    assert _input_source_count(cfg.hbir_compiler["input_source"]) == 4
    assert cfg.hbir_compiler["model_path"].endswith("qat.bc")


def test_bev_sparse_qconfig_calibration_only_pipeline():
    cfg = _load_config("qconfig.py")
    for name in (
        "calibration_train_model_convert_pipeline",
        "calibration_predict_model_convert_pipeline",
        "deploy_model_convert_pipeline",
        "calibration_qconfig_setter",
        "compile_cfg",
    ):
        assert hasattr(cfg, name), f"missing {name}"
    # bev_sparse has no qat training pipeline.
    assert not hasattr(cfg, "qat_train_model_convert_pipeline")
    # Export builds a QAT (fake-quant) graph from the calibration product.
    converters = cfg.deploy_model_convert_pipeline["converters"]
    assert any(c["type"] == "Float2QAT" for c in converters)
    assert _input_source_count(cfg.compile_cfg["input_source"]) == 6
    assert "split_dim" in cfg.compile_cfg
