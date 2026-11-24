# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import json
import urllib.request
from copy import copy

import torch


def _read_text(path):
    """Read text content from a local path or an HTTP(S) URL.

    Args:
        path: Local file path or URL to read from.

    Returns:
        Decoded text content of the file.
    """
    if path.startswith("http://") or path.startswith("https://"):
        with urllib.request.urlopen(path) as response:
            return response.read().decode("utf-8")
    with open(path, "r") as file_handle:
        return file_handle.read()


def openjson(path):
    """Load a JSON file from a local path or an HTTP(S) URL.

    Args:
        path: Local file path or URL of the JSON file.

    Returns:
        Parsed JSON content.
    """
    value = _read_text(path)
    return json.loads(value)


class StateNormalizer:
    """Normalize agent states with per-agent mean and standard deviation."""

    def __init__(self, mean, std):
        """Initialize the state normalizer.

        Args:
            mean: Per-agent state mean used for normalization.
            std: Per-agent state standard deviation used for normalization.
        """
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    @classmethod
    def from_json(
        cls,
        normalization_file_path,
        predicted_neighbor_num=10,
        use_neighbors_future=True,
    ):
        """Build a state normalizer from a normalization JSON file.

        Args:
            normalization_file_path: Path or URL of the normalization JSON
                file.
            predicted_neighbor_num: Number of predicted neighbor agents.
            use_neighbors_future: Whether to include neighbor agents'
                statistics.

        Returns:
            StateNormalizer instance built from the JSON statistics.
        """
        data = openjson(normalization_file_path)
        if use_neighbors_future:
            mean = [[data["ego"]["mean"]]] + [
                [data["neighbor"]["mean"]]
            ] * predicted_neighbor_num
            std = [[data["ego"]["std"]]] + [
                [data["neighbor"]["std"]]
            ] * predicted_neighbor_num
        else:
            mean = [[data["ego"]["mean"]]]
            std = [[data["ego"]["std"]]]
        return cls(mean, std)

    def __call__(self, data):
        d = data.shape[-1]
        return (data - self.mean[:, :, :d].to(data.device)) / self.std[
            :, :, :d
        ].to(data.device)

    def inverse(self, data):
        d = data.shape[-1]
        return data * self.std[:, :, :d].to(data.device) + self.mean[
            :, :, :d
        ].to(data.device)

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist(),
        }


class ObservationNormalizer:
    def __init__(self, normalization_dict):
        self._normalization_dict = normalization_dict

    @classmethod
    def from_json(cls, args):
        if isinstance(args, str):
            path = args
        else:
            path = args.normalization_file_path

        data = openjson(path)
        ndt = {}
        for key, value in data.items():
            if key not in ["ego", "neighbor"]:
                ndt[key] = {
                    "mean": torch.tensor(value["mean"], dtype=torch.float32),
                    "std": torch.tensor(value["std"], dtype=torch.float32),
                }
        return cls(ndt)

    def __call__(self, data):
        norm_data = copy(data)
        for key, value in self._normalization_dict.items():
            if key not in data:  # Check if key `key` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[key], 0), dim=-1) == 0
            norm_data[key] = (
                data[key] - value["mean"].to(data[key].device)
            ) / value["std"].to(data[key].device)
            norm_data[key][mask] = 0
        return norm_data

    def inverse(self, data):
        norm_data = copy(data)
        for key, value in self._normalization_dict.items():
            if key not in data:  # Check if key `key` exists in `data`
                continue
            mask = torch.sum(torch.ne(data[key], 0), dim=-1) == 0
            norm_data[key] = data[key] * value["std"].to(
                data[key].device
            ) + value["mean"].to(data[key].device)
            norm_data[key][mask] = 0
        return norm_data

    def to_dict(self):
        return {
            key: {
                inner_key: inner_value.detach().cpu().numpy().tolist()
                for inner_key, inner_value in value.items()
            }
            for key, value in self._normalization_dict.items()
        }
