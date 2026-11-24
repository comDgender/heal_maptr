# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


def _copy_state_dict(state_dict):
    new_state_dict = type(state_dict)()
    new_state_dict.update(state_dict)
    if hasattr(state_dict, "_metadata"):
        new_state_dict._metadata = state_dict._metadata
    return new_state_dict


def expand_sampling_wrapper_state_dict(state_dict, diffusion_pred_steps: int):
    """Backward-compatible key expansion for sampler wrapper splitting.

    Supported legacy layouts:
    1. sample.dpm_solver.dit_model.wrapper_model.xxx
    2. sample.dpm_solver.dit_model.wrapper_model.{i}.xxx

    Target layout:
    sample.dpm_solver.dit_model.{i}.wrapper_model.xxx
    """

    legacy_token = "sample.dpm_solver.dit_model.wrapper_model."
    target_token_tmpl = "sample.dpm_solver.dit_model.{}.wrapper_model"

    new_state_dict = type(state_dict)()
    if hasattr(state_dict, "_metadata"):
        new_state_dict._metadata = state_dict._metadata

    for key, value in state_dict.items():
        if legacy_token not in key:
            new_state_dict[key] = value
            continue

        key_prefix, key_suffix = key.split(legacy_token, 1)
        key_head, dot, key_tail = key_suffix.partition(".")

        if key_head.isdigit():
            new_key = key_prefix + target_token_tmpl.format(key_head)
            if dot:
                new_key = new_key + "." + key_tail
            if new_key not in new_state_dict:
                new_state_dict[new_key] = value
            continue

        for ix in range(diffusion_pred_steps + 1):
            new_key = (
                key_prefix + target_token_tmpl.format(ix) + "." + key_suffix
            )
            if new_key not in new_state_dict:
                new_state_dict[new_key] = value

    return new_state_dict


class UpdateStateDict:
    """Callable that updates state_dict for checkpoint loading.

    Splits dit model weights for the sampler (DPM solver with multiple steps).
    Uses a class instead of closure to be picklable for Ray workers.
    """

    def __init__(self, diffusion_pred_steps: int):
        self.diffusion_pred_steps = diffusion_pred_steps

    def __call__(self, state_dict):
        return expand_sampling_wrapper_state_dict(
            state_dict, self.diffusion_pred_steps
        )


def get_update_state_dict(diffusion_pred_steps: int):
    """Backward compatibility: returns UpdateStateDict instance."""
    return UpdateStateDict(diffusion_pred_steps)
