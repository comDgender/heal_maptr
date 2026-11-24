# Copyright (c) Horizon Robotics. All rights reserved.

import logging
from typing import Optional, Set

import torch
import torch.nn as nn
from hat_engine.callbacks.callbacks import CallbackMixin
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.logger import MSGColor, format_msg

from heal.models.model_convert.converters import LinearMinMaxStat

__all__ = ["ApplySmoothQuant"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ApplySmoothQuant(CallbackMixin):
    """Apply SmoothQuant to every hooked Linear, fold-only.

    Workflow
    --------
    1. Group Linears by their Norm predecessor (tagged by
       ``LinearMinMaxStat`` via ``stat_predecessor_norm``).
    2. For every Norm group, compute a shared in-channel scale
       ``s_shared = aggregate(s_1, ..., s_n)``, then re-clamp against the
       intersection of every member's ``[s_lower, s_upper]`` so aggregation
       cannot violate a sibling Linear's weight-range bound.
    3. Fold ``s_shared`` into the Norm's affine (``weight /= s``,
       ``bias /= s``) and into each downstream Linear's weight
       (``weight *= s`` along in_feats).
    4. SwiGLU ``down_proj`` has no Norm predecessor (input is
       ``SiLU(gate) * up``). Fold its scale into the sibling ``up_proj``
       along **out_features** instead::

           up.weight[j, :] /= s_j ;  down.weight[:, j] *= s_j

       ``gate_proj`` is left untouched because SiLU is not homogeneous.
    5. Remaining Linears without a fold target are **skipped** — wrapping
       would insert an ``x / s`` division that costs one quant node per
       Linear and pollutes the deploy graph.

    Scale formula (SmoothQuant, Xiao et al 2022)::

        s_j = max(|x_j|)^alpha / max(|w_j|)^(1 - alpha)

    When ``unconstrained_act_hard_topk > 0``, Linears with the highest
    activation hard-quant score (``outlier * (1 + pt_gap)``) skip the
    ``scale_min/max_ratio`` clamp and keep the raw SmoothQuant scale
    (``eps`` only). That supersedes ``down_proj_unconstrained``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        eps: float = 1e-6,
        module_filter=None,
        aggregate: str = "max",
        scale_min_ratio: float = 0.8,
        scale_max_ratio: float = 1.0,
        cast_fp32_after_fold: bool = False,
        down_proj_unconstrained: bool = False,
        unconstrained_act_hard_topk: int = 0,
        hard_act_report_path: Optional[str] = None,
        hard_act_report_topk: int = 20,
    ):
        """Init SmoothQuant applier.

        Args:
            alpha: balance factor; 0.5 = equal input/weight contribution.
            eps: clamp lower bound on |x| and |w| before the pow.
            module_filter: optional list of substrings; only Linear whose
                qualified name contains one of them is folded.
            aggregate: how to merge per-Linear scales sharing a Norm
                predecessor. ``max`` / ``mean`` / ``min``.
            scale_min_ratio: lower bound on post-fold weight magnitude as a
                multiple of the original per-out-channel **min**. Default 0.8
                lets weights shrink up to 20% of the original row-min.
            scale_max_ratio: upper bound on post-fold weight magnitude as a
                multiple of the original per-out-channel **max**. Default 1.0
                keeps the folded max ≤ original row-max (SmoothQuant then
                never grows the per-channel weight range).
            cast_fp32_after_fold: if True, cast the whole model to fp32 after
                folding (useful before a fp32 calibration pass).
            down_proj_unconstrained: if True (and
                ``unconstrained_act_hard_topk`` is 0), SwiGLU
                ``down_proj``→``up_proj`` fold uses the raw SmoothQuant
                scale (only ``eps`` safety), ignoring ``scale_min_ratio`` /
                ``scale_max_ratio``. Norm→Linear groups still use the
                ratio clamps.
            unconstrained_act_hard_topk: if > 0, after sync dump a hard-act
                ranking and leave the top-K Linears unconstrained (raw ``s``,
                eps only). Overrides ``down_proj_unconstrained``.
            hard_act_report_path: optional CSV path for the ranking dump.
            hard_act_report_topk: how many rows to print in the log.
        """
        super(ApplySmoothQuant, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.module_filter = list(module_filter) if module_filter else None
        self.aggregate = aggregate
        self.scale_min_ratio = scale_min_ratio
        self.scale_max_ratio = scale_max_ratio
        self.cast_fp32_after_fold = cast_fp32_after_fold
        self.down_proj_unconstrained = down_proj_unconstrained
        self.unconstrained_act_hard_topk = int(unconstrained_act_hard_topk)
        self.hard_act_report_path = hard_act_report_path
        self.hard_act_report_topk = int(hard_act_report_topk)
        self._hard_unconstrained_names: Set[str] = set()
        if aggregate not in ("max", "mean", "min"):
            raise ValueError(
                f'aggregate must be one of "max","mean","min"; got {aggregate}'
            )

    def _is_unconstrained(
        self, name: str, *, is_down_proj: bool = False
    ) -> bool:
        """Whether ``name`` should skip scale_min/max_ratio clamps."""
        if self.unconstrained_act_hard_topk > 0:
            return name in self._hard_unconstrained_names
        return bool(is_down_proj and self.down_proj_unconstrained)

    def _match(self, name: str) -> bool:
        if not self.module_filter:
            return True
        return any(sub in name for sub in self.module_filter)

    @torch.no_grad()
    def _compute_scale_and_bounds(self, linear: nn.Linear):
        """Return ``(s, s_lower, s_upper)`` for a Linear's in-channel scale.

        ``s`` is the raw SmoothQuant scale; callers must clamp to
        ``[s_lower, s_upper]`` (and, for shared groups, to the intersection
        of every member's bounds after aggregation).
        """
        x_abs = (
            torch.maximum(
                linear.stat_input_max_ch.abs(), linear.stat_input_min_ch.abs()
            )
            .clamp(min=self.eps)
            .float()
        )
        w = linear.weight.detach().float()  # [out_feats, in_feats]
        w_abs_per_j = w.abs().amax(dim=0).clamp(min=self.eps)
        s = x_abs.pow(self.alpha) / w_abs_per_j.pow(1.0 - self.alpha)

        # Per-out-channel min/max targets: fold must keep every weight inside
        # [r_lo * orig_min_i, r_hi * orig_max_i].
        #
        # Upper cap per column j:  |W[i,j] * s_j| ≤ r_hi * M_i  ∀ i
        #         ⟹  s_j ≤ min_i( r_hi * M_i / |W[i,j]| )
        # Lower floor per column j:
        #         |W[i,j] * s_j| ≥ r_lo * m_i  ∀ i
        #         ⟹  s_j ≥ max_i( r_lo * m_i / |W[i,j]| )
        w_abs = w.abs().clamp(min=self.eps)
        per_out_max = w_abs.amax(dim=1)
        per_out_min = w_abs.amin(dim=1)

        r_hi = self.scale_max_ratio
        s_upper = (r_hi * per_out_max).unsqueeze(1) / w_abs
        s_upper = s_upper.amin(dim=0)

        r_lo = self.scale_min_ratio
        s_lower = (r_lo * per_out_min).unsqueeze(1) / w_abs
        s_lower = s_lower.amax(dim=0).clamp(min=0)

        return s, s_lower, s_upper

    def _up_proj_div_bounds(self, up: nn.Linear, device, dtype):
        """Bounds on ``s`` when folding via ``up.weight[j,:] /= s_j``.

        Entire out-channel row ``j`` is scaled by ``1/s_j``, so:
          post_max_j = M_j / s_j ≤ r_hi * M_j  ⟹  s_j ≥ 1 / r_hi
          post_min_j = m_j / s_j ≥ r_lo * m_j  ⟹  s_j ≤ 1 / r_lo
        """
        n = up.out_features
        r_hi = self.scale_max_ratio
        r_lo = self.scale_min_ratio
        s_lower = torch.full(
            (n,),
            (1.0 / r_hi) if r_hi > 0 else 0.0,
            device=device,
            dtype=dtype,
        )
        s_upper = torch.full(
            (n,),
            (1.0 / r_lo) if r_lo > 0 else float("inf"),
            device=device,
            dtype=dtype,
        )
        return s_lower, s_upper

    def _aggregate(self, scales):
        stacked = torch.stack(scales, dim=0)
        if self.aggregate == "max":
            return stacked.amax(dim=0)
        if self.aggregate == "mean":
            return stacked.mean(dim=0)
        return stacked.amin(dim=0)

    @staticmethod
    def _clamp_shared(s, lowers, uppers):
        """Clamp ``s`` to the intersection of every member's bound interval."""
        s_lower = torch.stack(lowers, dim=0).amax(dim=0)
        s_upper = torch.stack(uppers, dim=0).amin(dim=0)
        # If a channel has lower > upper (pathological weights), prefer upper
        # so we never grow the weight range beyond scale_max_ratio.
        s_lower = torch.minimum(s_lower, s_upper)
        return s.clamp(min=s_lower, max=s_upper)

    @torch.no_grad()
    def _fold_norm_group(self, norm, s_shared: torch.Tensor) -> None:
        # Compute in fp32 (independent of weight dtype) to avoid intermediate
        # rounding when weights are stored in fp16/bf16; cast back at the end.
        w_dtype = norm.weight.dtype
        w_fp32 = norm.weight.detach().float() / s_shared.float()
        norm.weight.data = w_fp32.to(w_dtype)
        if getattr(norm, "bias", None) is not None:
            b_fp32 = norm.bias.detach().float() / s_shared.float()
            norm.bias.data = b_fp32.to(norm.bias.dtype)

    @torch.no_grad()
    def _scale_linear_weight(self, linear: nn.Linear, s: torch.Tensor) -> None:
        """Multiply Linear weight along in_features: ``W[:, j] *= s_j``."""
        w_dtype = linear.weight.dtype
        w_fp32 = linear.weight.detach().float() * s.unsqueeze(0).float()
        linear.weight.data = w_fp32.to(w_dtype)

    @torch.no_grad()
    def _div_linear_out_channels(
        self, linear: nn.Linear, s: torch.Tensor
    ) -> None:
        """Divide Linear weight/bias along out_features: ``W[j, :] /= s_j``."""
        w_dtype = linear.weight.dtype
        w_fp32 = linear.weight.detach().float() / s.unsqueeze(1).float()
        linear.weight.data = w_fp32.to(w_dtype)
        if linear.bias is not None:
            b_fp32 = linear.bias.detach().float() / s.float()
            linear.bias.data = b_fp32.to(linear.bias.dtype)

    @torch.no_grad()
    def _apply_norm_groups(self, model: nn.Module):
        """Fold shared-Norm Linear groups.

        Returns:
            ``(folded, orphan, folded_ids)``.
        """
        fold_groups = {}
        num_orphan = 0
        for parent_name, parent in model.named_modules():
            for attr, child in parent.named_children():
                if not isinstance(child, nn.Linear):
                    continue
                if not hasattr(child, "stat_input_max_ch"):
                    continue
                full_name = f"{parent_name}.{attr}" if parent_name else attr
                if not self._match(full_name):
                    continue

                norm_ref = getattr(child, "stat_predecessor_norm", None)
                norm = norm_ref() if norm_ref is not None else None
                if norm is None:
                    num_orphan += 1
                    continue
                key = (id(norm), child.in_features)
                if key not in fold_groups:
                    fold_groups[key] = {
                        "norm": norm,
                        "linears": [],
                        "names": [],
                    }
                fold_groups[key]["linears"].append(child)
                fold_groups[key]["names"].append(full_name)

        num_folded = 0
        folded_ids = set()
        for group in fold_groups.values():
            scales = []
            lowers = []
            uppers = []
            # If every member is hard-unconstrained, skip ratio clamps on the
            # shared scale; otherwise keep per-member bounds so siblings that
            # still need the clamp are protected.
            all_unconstrained = all(
                self._is_unconstrained(n) for n in group["names"]
            )
            for linear, name in zip(group["linears"], group["names"]):
                s, s_lower, s_upper = self._compute_scale_and_bounds(linear)
                if all_unconstrained or self._is_unconstrained(name):
                    s_lower = torch.zeros_like(s)
                    s_upper = torch.full_like(s, float("inf"))
                scales.append(s)
                lowers.append(s_lower)
                uppers.append(s_upper)
            if all_unconstrained:
                s_shared = self._aggregate(scales).clamp(min=self.eps)
            else:
                s_shared = self._clamp_shared(
                    self._aggregate(scales), lowers, uppers
                )
            norm = group["norm"]
            if getattr(norm, "_smoothquant_folded", False):
                num_orphan += len(group["linears"])
                continue
            self._fold_norm_group(norm, s_shared)
            norm._smoothquant_folded = True
            for linear in group["linears"]:
                self._scale_linear_weight(linear, s_shared)
                folded_ids.add(id(linear))
                num_folded += 1

        return num_folded, num_orphan, folded_ids

    @torch.no_grad()
    def _apply_swiglu_down_proj(self, model: nn.Module, folded_ids: set):
        """Fold ``down_proj`` scale into sibling ``up_proj`` out-channels.

        For SwiGLU ``h = SiLU(gate(x)) * up(x)``::

            h / s = SiLU(gate(x)) * (up(x) / s)

        so ``1/s`` absorbs into ``up_proj`` rows and ``s`` into ``down_proj``
        columns. ``gate_proj`` is never touched.
        """
        num_folded = 0
        num_skipped = 0
        for parent_name, parent in model.named_modules():
            down = getattr(parent, "down_proj", None)
            up = getattr(parent, "up_proj", None)
            if not isinstance(down, nn.Linear) or not isinstance(
                up, nn.Linear
            ):
                continue
            if not hasattr(down, "stat_input_max_ch"):
                continue
            down_name = (
                f"{parent_name}.down_proj" if parent_name else "down_proj"
            )
            if not self._match(down_name):
                continue
            if id(down) in folded_ids:
                continue
            if down.in_features != up.out_features:
                num_skipped += 1
                continue
            if getattr(up, "_smoothquant_down_folded", False):
                num_skipped += 1
                continue

            s, s_lo_down, s_hi_down = self._compute_scale_and_bounds(down)
            if self._is_unconstrained(down_name, is_down_proj=True):
                # Only keep s finite / positive; allow free migration of
                # activation range into up_proj / down_proj weights.
                s = s.clamp(min=self.eps)
            else:
                s_lo_up, s_hi_up = self._up_proj_div_bounds(
                    up, device=s.device, dtype=s.dtype
                )
                s = self._clamp_shared(
                    s, [s_lo_down, s_lo_up], [s_hi_down, s_hi_up]
                )

            self._div_linear_out_channels(up, s)
            self._scale_linear_weight(down, s)
            up._smoothquant_down_folded = True
            folded_ids.add(id(down))
            folded_ids.add(id(up))
            num_folded += 1

        return num_folded, num_skipped

    @torch.no_grad()
    def _apply(self, model: nn.Module) -> tuple:
        num_norm_folded, num_orphan, folded_ids = self._apply_norm_groups(
            model
        )
        num_down_folded, num_down_skipped = self._apply_swiglu_down_proj(
            model, folded_ids
        )
        # Orphans that got a SwiGLU fold are no longer skips.
        num_orphan = max(0, num_orphan - num_down_folded) + num_down_skipped
        return num_norm_folded, num_down_folded, num_orphan

    def _resolve_hard_unconstrained(self, model: nn.Module) -> int:
        """Sync-derived ranking → populate ``_hard_unconstrained_names``."""
        self._hard_unconstrained_names = set()
        if (
            self.unconstrained_act_hard_topk <= 0
            and not self.hard_act_report_path
        ):
            return 0
        # Ensure absmax / derived scalars are populated even if hooks only
        # wrote min/max (e.g. older ckpt stats).
        for _name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not hasattr(mod, "stat_act_absmax_ch"):
                continue
            if float(mod.stat_act_absmax_ch.max()) <= 0 and hasattr(
                mod, "stat_input_max_ch"
            ):
                absmax = torch.maximum(
                    mod.stat_input_max_ch.abs(),
                    mod.stat_input_min_ch.abs(),
                )
                mod.stat_act_absmax_ch.copy_(absmax)
            LinearMinMaxStat._refresh_act_hard_buffers(mod)

        report_topk = max(
            self.hard_act_report_topk, self.unconstrained_act_hard_topk, 0
        )
        # Only rank-0 writes the CSV; every rank still needs the name set.
        write_path = None
        if self.hard_act_report_path:
            rank = 0
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                rank = torch.distributed.get_rank()
            if rank == 0:
                write_path = self.hard_act_report_path
        LinearMinMaxStat.dump_hard_act_report(
            model, topk=report_topk, path=write_path
        )
        if self.unconstrained_act_hard_topk <= 0:
            return 0
        names, _rows = LinearMinMaxStat.topk_hard_act_names(
            model, self.unconstrained_act_hard_topk
        )
        self._hard_unconstrained_names = set(names)
        logger.info(
            format_msg(
                "ApplySmoothQuant: unconstrained %d layers by hard_score "
                "(topk=%d)."
                % (
                    len(self._hard_unconstrained_names),
                    self.unconstrained_act_hard_topk,
                ),
                MSGColor.GREEN,
            )
        )
        return len(self._hard_unconstrained_names)

    def on_loop_end(self, model, **kwargs):
        num_reduced = self._maybe_sync_stats(model)
        num_hard = self._resolve_hard_unconstrained(model)
        num_norm_folded, num_down_folded, num_orphan = self._apply(model)
        if self.cast_fp32_after_fold:
            model.to(torch.float32)
        logger.info(
            format_msg(
                "ApplySmoothQuant: synced %d, norm-folded %d, "
                "down_proj-folded %d, skipped %d "
                "(alpha=%.3f, aggregate=%s, cast_fp32=%s, "
                "down_proj_unconstrained=%s, hard_unconstrained=%d)."
                % (
                    num_reduced,
                    num_norm_folded,
                    num_down_folded,
                    num_orphan,
                    self.alpha,
                    self.aggregate,
                    self.cast_fp32_after_fold,
                    self.down_proj_unconstrained,
                    num_hard,
                ),
                MSGColor.GREEN,
            )
        )

    @staticmethod
    def _maybe_sync_stats(model) -> int:
        """All-reduce Linear stat buffers across DDP ranks.

        Ensures every rank's fold/wrap decision uses the same global
        min/max. No-op on a single rank. After sync, each rank runs the
        deterministic fold/wrap logic on the same inputs so weights match
        without a later weight broadcast.
        """
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return 0
        num_reduced = 0
        for _name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            for buf_name, op in (
                ("stat_input_max_ch", torch.distributed.ReduceOp.MAX),
                ("stat_input_min_ch", torch.distributed.ReduceOp.MIN),
                ("stat_input_global_max", torch.distributed.ReduceOp.MAX),
                ("stat_input_global_min", torch.distributed.ReduceOp.MIN),
                ("stat_act_absmax_ch", torch.distributed.ReduceOp.MAX),
                ("stat_act_outlier", torch.distributed.ReduceOp.MAX),
                ("stat_act_pt_gap", torch.distributed.ReduceOp.MAX),
            ):
                buf = getattr(mod, buf_name, None)
                if buf is None:
                    continue
                torch.distributed.all_reduce(buf, op=op)
                num_reduced += 1
            if hasattr(mod, "stat_act_absmax_ch"):
                LinearMinMaxStat._refresh_act_hard_buffers(mod)
        return num_reduced
