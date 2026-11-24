# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.quantization import QuantStub

try:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
    from nuplan.planning.training.modeling.torch_module_wrapper import (
        TorchModuleWrapper,
    )
    from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (  # noqa
        EgoTrajectoryTargetBuilder,
    )
except ImportError:
    TrajectorySampling = None
    TorchModuleWrapper = None
    EgoTrajectoryTargetBuilder = None

# PlutoFeatureBuilder is a HAT visualize helper, optional.
try:
    from heal.visualize.planner.feature_builders.pluto_feature_builder import (
        PlutoFeatureBuilder,
    )
except ImportError:
    PlutoFeatureBuilder = None

__all__ = [
    "Diffusion_Planner",
    "Diffusion_Planner_Encoder",
    "Diffusion_Planner_Decoder",
]

trajectory_sampling = (
    TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)
    if TrajectorySampling
    else None
)


def _get_diffusion_planner_base():
    if TorchModuleWrapper is not None:
        return TorchModuleWrapper
    return nn.Module


@OBJECT_REGISTRY.register
class Diffusion_Planner(_get_diffusion_planner_base()):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        feature_builder=None,
        use_neighbors_future=True,
    ):
        if (
            TorchModuleWrapper is not None
            and EgoTrajectoryTargetBuilder is not None
            and trajectory_sampling is not None
        ):
            fb = feature_builder or (
                PlutoFeatureBuilder() if PlutoFeatureBuilder else None
            )
            super().__init__(
                feature_builders=[fb] if fb else [],
                target_builders=[
                    EgoTrajectoryTargetBuilder(trajectory_sampling)
                ],
                future_trajectory_sampling=trajectory_sampling,
            )
        else:
            super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_neighbors_future = use_neighbors_future
        if self.use_neighbors_future:
            self.neighbor_past_quant = QuantStub()

    def forward(self, inputs):
        if self.use_neighbors_future:
            quantized_neighbor_agents_past = self.neighbor_past_quant(
                inputs["neighbor_agents_past"]
            )

            encoder_outputs = self.encoder(
                inputs, quantized_neighbor_agents_past
            )
            decoder_outputs = self.decoder(
                encoder_outputs, inputs, quantized_neighbor_agents_past
            )
        else:
            encoder_outputs = self.encoder(inputs)
            decoder_outputs = self.decoder(encoder_outputs, inputs)
        return decoder_outputs


@OBJECT_REGISTRY.register
class Diffusion_Planner_Encoder(nn.Module):
    """Diffusion planner encoder wrapper with weight initialization."""

    def __init__(
        self,
        encoder_layer: nn.Module,
    ):
        """Initialize Diffusion_Planner_Encoder.

        Args:
            encoder_layer: Underlying encoder module.
        """
        super().__init__()

        self.encoder = encoder_layer
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize encoder weights."""

        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Initialize embedding MLP:
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        if self.encoder.neighbor_encoder is not None:
            nn.init.normal_(
                self.encoder.neighbor_encoder.type_emb.weight,
                std=0.02,
            )
        if self.encoder.lane_encoder is not None and getattr(
            self.encoder.lane_encoder, "with_traffic_emb", False
        ):
            nn.init.normal_(
                self.encoder.lane_encoder.traffic_emb.weight,
                std=0.02,
            )

    def forward(self, inputs, quantized_neighbor_agents_past=None):
        """Run encoder forward.

        Args:
            inputs: Input feature dict.
            quantized_neighbor_agents_past: Quantized neighbor past tensor.

        Returns:
            Encoder outputs consumed by the decoder.
        """
        encoder_outputs = self.encoder(inputs, quantized_neighbor_agents_past)

        return encoder_outputs


@OBJECT_REGISTRY.register
class Diffusion_Planner_Decoder(nn.Module):
    """Diffusion planner decoder wrapper with DiT initialization."""

    def __init__(
        self,
        decoder_layer: nn.Module,
        criterion: nn.Module = None,
    ):
        """Initialize Diffusion_Planner_Decoder.

        Args:
            decoder_layer: Underlying decoder module.
            criterion: Optional loss module used during training.
        """
        super().__init__()

        self.decoder = decoder_layer
        self.criterion = criterion
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize decoder and DiT weights."""

        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(
            self.decoder.dit.route_encoder.speed_limit_emb.weight, std=0.02
        )

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(
            self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0
        )
        nn.init.constant_(
            self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0
        )
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(
        self, encoder_outputs, inputs, quantized_neighbor_agents_past=None
    ):
        """Run decoder forward and compute losses during training.

        Args:
            encoder_outputs: Encoder feature dict.
            inputs: Original input feature dict.
            quantized_neighbor_agents_past: Quantized neighbor past tensor.

        Returns:
            Decoder outputs, or a (loss_dict, decoder_outputs) tuple in
            training mode when criterion is set.
        """
        decoder_outputs = self.decoder(
            encoder_outputs, inputs, quantized_neighbor_agents_past
        )
        if self.criterion is not None and self.training:
            loss_dict, decoder_outputs = self.criterion(
                decoder_outputs, inputs
            )
            return loss_dict, decoder_outputs

        return decoder_outputs


@OBJECT_REGISTRY.register
class DiffusionPlannerIrInfer(nn.Module):
    """DiffusionPlanner IR inference wrapper."""

    def __init__(
        self,
        ir_model: nn.Module,
        deploy_model: nn.Module = None,
        convert_pipeline=None,
    ):
        """Initialize DiffusionPlannerIrInfer.

        Args:
            ir_model: Wrapped IR model used for inference.
            deploy_model: Optional deploy model.
            convert_pipeline: Optional result conversion pipeline.
        """
        super(DiffusionPlannerIrInfer, self).__init__()
        self.ir_model = ir_model

    def process_input(self, data):
        """Prepare model inputs by adding noise and casting flags.

        Args:
            data: Raw input feature dict.

        Returns:
            Processed input feature dict.
        """
        device = data["ego_current_state"].device
        noise = torch.randn(
            (1, 11, 80, 4),
        ).to(device)
        data["noise"] = noise
        data["route_lanes_has_speed_limit"] = data[
            "route_lanes_has_speed_limit"
        ].to(torch.bool)
        return data

    def process_output(self, outputs):
        """Convert IR outputs to a prediction dict.

        Args:
            outputs: Raw IR model outputs.

        Returns:
            Dict containing the prediction tensor.
        """
        if "_output_0" in outputs:
            hbir_outputs = {"prediction": outputs["_output_0"]}
        else:
            hbir_outputs = outputs
        return hbir_outputs

    def forward(self, data):
        """Run IR inference end to end.

        Args:
            data: Raw input feature dict.

        Returns:
            Prediction dict.
        """
        inputs = self.process_input(data)
        hbir_outputs = self.ir_model(inputs)
        results = self.process_output(hbir_outputs)
        return results


@OBJECT_REGISTRY.register
class DiffusionPlannerHbmInfer(nn.Module):
    """DiffusionPlanner HBM inference wrapper."""

    def __init__(
        self,
        ir_model: nn.Module,
        deploy_model: nn.Module = None,
        convert_pipeline=None,
    ):
        """Initialize DiffusionPlannerHbmInfer.

        Args:
            ir_model: Wrapped IR model used for inference.
            deploy_model: Optional deploy model.
            convert_pipeline: Optional result conversion pipeline.
        """
        super(DiffusionPlannerHbmInfer, self).__init__()
        self.ir_model = ir_model

    def process_input(self, data):
        """Prepare model inputs by adding noise and casting flags.

        Args:
            data: Raw input feature dict.

        Returns:
            Processed input feature dict.
        """
        device = data["ego_current_state"].device
        noise = torch.randn(
            (1, 11, 80, 4),
        ).to(device)
        data["noise"] = noise
        data["route_lanes_has_speed_limit"] = data[
            "route_lanes_has_speed_limit"
        ].to(torch.bool)
        return data

    def process_output(self, outputs):
        """Wrap raw outputs into a prediction dict.

        Args:
            outputs: Raw model outputs.

        Returns:
            Dict containing the prediction tensor.
        """
        hbir_outputs = {"prediction": outputs}
        return hbir_outputs

    def forward(self, data):
        """Run HBM inference end to end.

        Args:
            data: Raw input feature dict.

        Returns:
            Prediction dict.
        """
        inputs = self.process_input(data)
        hbir_outputs = self.ir_model(inputs)
        results = self.process_output(hbir_outputs)
        return results


@OBJECT_REGISTRY.register
class TorchWrapper(nn.Module):
    """Wrapper that exposes a plugin module as an nn.Module."""

    def __init__(self, model):
        """Initialize TorchWrapper.

        Args:
            model: Plugin model to wrap.
        """
        super().__init__()
        self.plugin_wrapper = model

    def forward(self, data):
        """Run the wrapped plugin model.

        Args:
            data: Input feature dict.

        Returns:
            Model outputs.
        """
        out = self.plugin_wrapper(data)
        return out


@OBJECT_REGISTRY.register
class TorchIrmodule(nn.Module):
    """IR wrapper module containing a TorchWrapper."""

    def __init__(self, model):
        """Initialize TorchIrmodule.

        Args:
            model: Plugin model to wrap.
        """
        super().__init__()
        self.ir_model = TorchWrapper(model)

    def forward(self, data):
        """Run the wrapped IR model.

        Args:
            data: Input feature dict.

        Returns:
            Model outputs.
        """
        out = self.ir_model(data)
        return out
