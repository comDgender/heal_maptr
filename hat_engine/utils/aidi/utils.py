# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import abc
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import yaml

from hat_engine.utils.config import crop_configs
from hat_engine.utils.distributed import rank_zero_only
from hat_engine.utils.logger import MSGColor, format_msg
from hat_engine.utils.package_helper import require_packages

logger = logging.getLogger(__name__)

try:
    import aidisdk
    from aidisdk import AIDIClient
    from aidisdk.experiment import Image, Table
    from aidisdk.experiment.artifact import Artifact
    from aidisdk.experiment.model_eval import ModelEvalDataset
    from aidisdk.utils import running_in_cluster
    from dataclasses_json import DataClassJsonMixin

    IS_LOCAL = not running_in_cluster()
except ImportError:
    AIDIClient, Image, Table, Artifact, ModelEvalDataset = (
        None,
        None,
        None,
        None,
        None,
    )
    DataClassJsonMixin = object

    IS_LOCAL = True


__all__ = [
    "AIDIClient",
    "get_aidi_client",
    "get_aidi_token",
    "AIDIExperimentLogger",
    "Artifact",
    "ModelEvalDataset",
    "EvalResult",
    "tracking_model_as_input",
    "is_running_on_aidi",
    "is_running_on_aidi_pipeline",
    "is_running_on_local_docker",
    "get_user_token",
    "IS_LOCAL",
]

HOSTFILE = "/job_data/hosts"

# default configuration of root logging
DEFAULT_LOG_CONFIG = """
version: 1
disable_existing_loggers: False
formatters:
    default:
        format: '{format}'

handlers:
    info_console:
        class: logging.StreamHandler
        level: INFO
        formatter: default
        stream: ext://sys.stderr  # stdout may cause problem in cluster

    warning_console:
        class: logging.StreamHandler
        level: WARNING
        formatter: default
        stream: ext://sys.stderr

    debug_file_handler:
        class: logging.handlers.RotatingFileHandler
        level: DEBUG
        formatter: default
        filename: {log_dir}/debug.log
        maxBytes: 104857600       # 100MB
        backupCount: 10000        # about 1T logs files
        encoding: utf8

    warning_file_handler:
        class: logging.handlers.RotatingFileHandler
        level: WARNING
        formatter: default
        filename: {log_dir}/warn.log
        maxBytes: 104857600       # 100MB
        backupCount: 10000        # about 1T logs files
        encoding: utf8

    error_file_handler:
        class: logging.handlers.RotatingFileHandler
        level: ERROR
        formatter: default
        filename: {log_dir}/error.log
        maxBytes: 104857600       # 100MB
        backupCount: 10000        # about 1T logs files
        encoding: utf8

loggers:
    auto_dp:
        level: WARNING
        handlers: [warning_console]
        propagate: no

root:
    level: DEBUG
    handlers: [info_console, debug_file_handler,
        warning_file_handler, error_file_handler]
"""

DEFAULT_LOG_FORMAT = (
    "%(asctime)-15s %(levelname)s "
    "| %(process)d | %(threadName)s | "
    "%(module)s:%(name)s:L%(lineno)d %(message)s"
)


def get_aidi_client():
    """Get user client.

    Returns:
      client: User client.
    """
    if AIDIClient is None:
        return None
    return AIDIClient()


def get_aidi_token():
    """Get user token.

    Returns:
      token: User token.
    """
    client = AIDIClient()
    token = client.session.token
    return token


def is_running_on_aidi():
    """Whether is running on aidi."""
    return (
        os.environ.get("WORKING_PATH", None) is not None
        and os.environ.get("PBS_JOBNAME", None) is not None
    )


def is_running_on_aidi_pipeline():
    """Whether is running on aidi pipeline."""
    return is_running_on_aidi() and os.environ.get("DAG_ID") is not None


def is_running_on_local_docker():
    """Whether is running on local docker environment."""
    return os.environ.get("HAT_AIDI_DAG_DOCKER_WORKSPACE") is not None


@require_packages("aidisdk")
def get_user_token():
    client = aidisdk.AIDIClient()
    return client.session.token


@lru_cache()
def check_tracking_status(aidi_client=None):
    """Check whether to enabled aidi tracking."""
    if aidi_client is None:
        aidi_client = get_aidi_client()

    enable_tracking = bool(
        int(os.environ.get("HAT_ENABLE_MODEL_TRACKING", "0"))
    )

    try:
        enabled = aidi_client.experiment.enabled
    except Exception as e:
        logger.info(
            format_msg(
                f"AIDI Experiment: Get aidi experiment.enabled failed: {str(e)}, will set enable_tracking=False",  # noqa E501
                MSGColor.RED,
            )
        )
        enabled = False
    return enabled and enable_tracking


@rank_zero_only
def tracking_model_as_input(
    model_name: str,
    model_version: str,
    stage: str,
):
    """Tracking aidi model.

    Args:
        model_name: model name.
        model_version: model version.
        stage: model training stage.
    """
    client = get_aidi_client()

    if check_tracking_status(client):
        try:
            client.experiment.log_input_artifact(
                Artifact.from_model_stage(
                    client.model.finditem(
                        model_name=model_name,
                        model_version=model_version,
                        stage=stage,
                    )
                )
            )
            logger.info(
                format_msg(
                    f"AIDI Tracking:  tracking input checkpoint for model[{model_name}], "  # noqa E501
                    f"version[{model_version}], stage[{stage}] successfully.",  # noqa E501
                    MSGColor.GREEN,
                )
            )
        except Exception as e:
            logger.info(
                format_msg(
                    f"AIDI Tracking:  tracking input checkpoint for model[{model_name}], "  # noqa E501
                    f"version[{model_version}], stage[{stage}] failed: {e}.",  # noqa E501
                    MSGColor.RED,
                )
            )


@dataclass
class EvalResult(DataClassJsonMixin):
    """Dataclass for result of evaluation."""

    summary: dict = None
    tables: List[Table] = None
    plots: List[dict] = None
    images: List[Image] = None
    confusion_matrixes: List[dict] = None


class AIDIExperimentLogger:
    def __init__(self) -> None:
        self._client = None

    def init(
        self,
        experiment_name: str,
        project_id: str,
        experiment_path: str,
        enable_tracking: str,
        run_name: str,
        runtime: str = "local",
        config_file: str = None,
    ):
        if self.client.experiment.get_experiment(experiment_name) is None:
            self.client.experiment.create_experiment(
                name=experiment_name,
                project_id=project_id,
                experiment_path=experiment_path,
            )
        with self.client.experiment.init(
            experiment_name=experiment_name,
            run_name=run_name,
            enabled=enable_tracking,
        ) as run:
            run.log_runtime(runtime=runtime, config_file=config_file)

    def init_group(self, group):
        try:
            if self.enabled_tracking():
                self.client.experiment.init_group(group)
        except Exception as e:
            logger.warning(f"failed to init_group: {str(e)}")

    def log_config(self, config):
        try:
            if self.enabled_tracking():
                self.client.experiment.log_config(crop_configs(config))
        except Exception as e:
            logger.warning(f"failed to log_config: {str(e)}")

    def log_artifact(
        self,
        artifact_name: str,
        artifact_type: str,
        artifact_aliases: List[str],
        artifact_tags: List[str],
        files: Optional[List[str]] = None,
        overwrite: bool = False,
    ):
        if self.enabled_tracking():
            output_artifact = Artifact(
                artifact_name,
                type=artifact_type,
                aliases=artifact_aliases,
                tags=artifact_tags,
            )

            if files:
                for file in files:
                    if os.path.exists(file):
                        output_artifact.add_file(file, overwrite=overwrite)

            self.client.experiment.log_artifact(
                artifact=output_artifact,
                overwrite=overwrite,
            )

    @classmethod
    def enabled_tracking(cls, client=None):
        """Check whether to enabled aidi tracking."""
        return check_tracking_status(client)

    @classmethod
    def download_checkpoint_from_artifact(
        cls,
        artifact_path: str,
        enable_tracking: bool = False,
    ):
        (
            _,
            _,
            model_name,
            training_stage,
            entry_alias,
            ckpt_name,
        ) = artifact_path.split(os.sep)
        artifact_name = f"{model_name}-{training_stage}:{entry_alias}"

        input_artifact = cls().client.experiment.use_artifact(
            artifact_name, trace=enable_tracking
        )
        return input_artifact.get_file(ckpt_name)

    def log_exception(self, exception):
        if not IS_LOCAL:
            self.client.single_job.log_exception(exception)

    def log_metrics(
        self,
        metrics_list: Dict[str, Any],
        epoch_id: int,
        step_id: int,
        **kwargs,
    ):
        for k, v in metrics_list:
            if isinstance(v, EvalResult):
                if v.summary is not None:
                    self.client.experiment.log_summary(v.summary)
                if v.tables is not None:
                    for table in v.tables:
                        self.client.experiment.log_table(table)
                if v.plots is not None:
                    for plot in v.plots:
                        self.client.experiment.log_plot(
                            plot["Table"].name,
                            plot["Table"],
                            plot["Line"],
                        )
                if v.images is not None:
                    for image in v.images:
                        self.client.experiment.log_image(image)
                    self.client.experiment.flush()
                if v.confusion_matrixes is not None:
                    for confusion_matrix in v.confusion_matrixes:
                        self.client.experiment.log_confusion_matrix(
                            confusion_matrix["labels"],
                            confusion_matrix["matrix"],
                            confusion_matrix["name"],
                        )
            else:
                self.client.experiment.log_metrics(
                    key=k,
                    value=v,
                    step=step_id,
                    epoch_id=epoch_id,
                )

    @property
    def client(self):
        if self._client is None:
            self._client = get_aidi_client()
        return self._client


class BaseLoggingConfig(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set_logging_config(self):
        pass


class FileLoggingConfig(BaseLoggingConfig):
    """A file logging config which read config from file or dict."""

    def __init__(
        self,
        level: str = None,
        fpath: str = None,
        dict_config: dict = None,
        log_dir: str = None,
    ):
        """Initialize the logging configuration.

        Args:
            level: root logging level, by default None
            fpath: config file path, by default None
            cfg_dict: config dict, by default None. If
                there are both `fpath` and `cfg_dict`,
                it will cause error
            log_dir: The output log dir path
        """
        assert (
            fpath is None or dict_config is None
        ), "DO NOT both file configuration and dict configuration"

        self.level = level
        self.fpath = fpath
        self.dict_config = dict_config
        self.log_dir = log_dir

    def set_logging_config(self):
        assert (
            self.fpath is None or self.dict_config is None
        ), "DO NOT both file configuration and dict configuration"

        if self.dict_config is not None:
            config = self.dict_config
        else:
            config = self._get_yaml_config(self.fpath, self.log_dir)

        self._create_log_file()
        logging.config.dictConfig(config)

        if self.level is not None:
            self.set_logger_level("root", self.level)

    def _get_yaml_config(self, fpath: str = None, log_dir: str = None):
        """Read configurations from yaml file.

        Args:
            fpath: path of the configuration file, by default None
            log_dir: The output log dir path

        Returns:
            dictionary of logging configurations
        """
        if fpath is None:
            if is_running_on_aidi():
                log_dir = "/job_log"
            elif log_dir is None:
                log_dir = "log"
            config = yaml.safe_load(
                DEFAULT_LOG_CONFIG.format(
                    format=DEFAULT_LOG_FORMAT, log_dir=log_dir
                )
            )
        else:
            assert os.path.exists(fpath), "No configuration file was found!"
            if log_dir:
                raise ValueError("log_dir should be None when fpath exists")
            with open(fpath) as f:
                config = yaml.safe_load(f.read())

        return config

    def _create_log_file(self):
        """Create logger directories.

        When we configure the handler filename as 'folder1/xxx.log', if the
        folder doesn't exist, the logging module will raise error.
        """
        handlers = self.config.get("handlers")
        if handlers is None:
            return

        for _, handler in handlers.items():
            filename = handler.get("filename")
            if filename is None:
                continue
            dir = os.path.dirname(filename)
            if dir != "":
                os.makedirs(dir, exist_ok=True)

    def set_logger_level(self, logger_name: str, level: Union[str, int]):
        """Change level for certain logger.

        Args:
            logger_name: name of the logger
            level: level to be set. Both string and int are acceptable
        """
        if isinstance(level, str):
            # make sure the level string is upper case
            level = str.upper(level)
        logger = self.get_logger(logger_name)
        logger.setLevel(level)


default_logging_config = FileLoggingConfig()
