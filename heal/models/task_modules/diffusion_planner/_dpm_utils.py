# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch


def interpolate_fn(x, xp, yp):
    # Piecewise linear function y = f(x), differentiable (autograd-friendly).
    n, k = x.shape[0], xp.shape[1]
    all_x = torch.cat(
        [x.unsqueeze(2), xp.unsqueeze(0).repeat((n, 1, 1))], dim=2
    )
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, k),
            torch.tensor(k - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(
        torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1
    )
    start_x = torch.gather(
        sorted_all_x, dim=2, index=start_idx.unsqueeze(2)
    ).squeeze(2)
    end_x = torch.gather(
        sorted_all_x, dim=2, index=end_idx.unsqueeze(2)
    ).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, k),
            torch.tensor(k - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(n, -1, -1)
    start_y = torch.gather(
        y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)
    ).squeeze(2)
    end_y = torch.gather(
        y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)
    ).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and total dim `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
