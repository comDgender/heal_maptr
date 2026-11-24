# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from hat_engine.utils.checkpoint import load_checkpoint, load_state_dict
from hat_engine.utils.global_var import set_value
from hat_engine.utils.logger import MSGColor, format_msg
from hat_engine.utils.msic import singleton

logger = logging.Logger(__name__)

IS_LOCAL = not os.path.exists("/running_package")


class StateType(Enum):
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


def use_elastic():
    return bool(int(os.getenv("USE_ELASTIC", "0")))


def elastic_need_resume():
    is_restart_training = int(os.getenv("ELASTIC_NEED_RESUME", "0")) > 0

    return is_restart_training and use_elastic()


class JsonStorage:
    def __init__(self, json_file) -> None:
        self.json_file = json_file

    def _load(self):
        try:
            with open(self.json_file, "r") as f:
                data = json.load(f)
                if not data:
                    return {}
                return data
        except FileNotFoundError:
            return {}

    def _save(self, data):
        with open(self.json_file, "w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_data(cls, json_file):
        return cls(json_file)._load()

    @classmethod
    def save_data(cls, json_file, data):
        return cls(json_file)._save(data)


@dataclass
class JobStateMeta:
    command: str = ""
    state: str = ""
    checkpoint: Optional[str] = None

    def from_json(self, json_file: str, cmd_key: str):
        data = JsonStorage.load_data(json_file)
        cmd_data = data.get(cmd_key, {})
        return JobStateMeta(
            command=cmd_data.get("command", cmd_key),
            state=cmd_data.get("state", None),
            checkpoint=cmd_data.get("checkpoint", None),
        )

    def to_dict(self):
        return {
            "command": self.command,
            "state": self.state,
            "checkpoint": self.checkpoint,
        }


@singleton
class ElasticState:
    def __init__(self) -> None:
        if not IS_LOCAL:
            self._elastic_record = os.path.join(
                "/job_data/elastic/",
                "elastic_record.json",
            )
        else:
            self._elastic_record = os.path.join(
                ".hat_logs/elastic/",
                "elastic_record.json",
            )
        with contextlib.suppress(FileExistsError):
            os.makedirs(os.path.dirname(self._elastic_record), exist_ok=True)

        self._state = None

    def update_checkpoint(self, checkpoint):
        """Save checkpoint path to json file.

        Args:
            checkpoint: path of checkpoint.
        """

        state_meta = self.get_job_state()
        if state_meta.checkpoint == checkpoint:
            logger.warning(
                format_msg(
                    f"Elastic has same checkpoint, skip in update checkpoint: {checkpoint}",  # noqa E501
                    MSGColor.GREEN,
                )
            )
            return
        state_meta.checkpoint = checkpoint
        self._update(state_meta=state_meta)
        logger.warning(
            format_msg(
                f"Elastic recorded checkpoint: {checkpoint}",
                MSGColor.GREEN,
            )
        )

    def load_checkpoint(self, model):
        """Resume model state from saved checkpoint."""

        state = self.get_job_state()
        checkpoint_path = state.checkpoint
        if checkpoint_path and model:
            # Strangely, log will not be printed by using `logger.info()`
            logger.warning(
                format_msg(
                    f"Elastic will resume from checkpoint: {checkpoint_path}",
                    MSGColor.GREEN,
                )
            )
            model_checkpoint = load_checkpoint(
                path_or_dict=checkpoint_path,
                map_location="cpu",
                check_hash=False,
            )
            set_value("model_checkpoint", model_checkpoint)
            model = load_state_dict(
                model,
                path_or_dict=model_checkpoint,
                allow_miss=False,
                ignore_extra=False,
                verbose=True,
            )
            logger.warning(
                format_msg(
                    f"Elastic has resumed from checkpoint: {checkpoint_path}",
                    MSGColor.GREEN,
                )
            )

        return model

    def update_state(self, state):
        """Update current command running state."""
        old_state = self._state
        if self._state == state:
            logger.warning(
                format_msg(
                    f"Elastic has same state, skip in update state: {state}",
                    MSGColor.GREEN,
                )
            )
            return
        else:
            self._state = state

        state_meta = self.get_job_state()
        state_meta.state = state
        self._update(state_meta=state_meta)
        logger.warning(
            format_msg(
                f"Elastic update state from {old_state} to {state}",
                MSGColor.GREEN,
            )
        )

    def _update(self, state_meta: JobStateMeta):
        try:
            data = JsonStorage.load_data(self._elastic_record)
            data[self.cur_command] = state_meta.to_dict()
            JsonStorage.save_data(self._elastic_record, data)
        except Exception as e:
            logger.error(
                format_msg(
                    f"ElasticState update failed: {e}",
                    MSGColor.RED,
                )
            )

    @property
    def cur_command(self):
        return os.getenv("HAT_PIPELINE_COMMAND", None)

    def get_job_state(self, json_file: str = None, cmd: str = None):

        if json_file is None:
            json_file = self._elastic_record

        if cmd is None:
            if self.cur_command:
                cmd = self.cur_command
            else:
                raise ValueError(
                    "cmd and `HAT_PIPELINE_COMMAND` can not both be None"
                )
        return JobStateMeta().from_json(json_file, cmd)
