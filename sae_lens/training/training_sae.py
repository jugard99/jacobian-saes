"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import json
import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Literal, Optional, overload

import einops
import torch
from jaxtyping import Float, Int
from torch import nn
from transformer_lens.components.transformer_block import TransformerBlock
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae import SAE, SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import (
    handle_config_defaulting,
    read_sae_from_disk,
)
from sae_lens.training.mlp_with_act_grads import MLPWithActGrads

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


@dataclass
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    sae_in2: torch.Tensor
    sae_out2: torch.Tensor
    feature_acts2: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    mse_loss: float
    l1_loss: float
    ghost_grad_loss: float
    auxiliary_reconstruction_loss: float = 0.0
    jacobian_loss: float = 0.0
    mse_loss2: float = 0.0
    l1_loss2: float = 0.0


@dataclass(kw_only=True)
class TrainingSAEConfig(SAEConfig):

    # Sparsity Loss Calculations
    l1_coefficient: float
    lp_norm: float
    use_ghost_grads: bool
    normalize_sae_decoder: bool
    noise_scale: float
    decoder_orthogonal_init: bool
    mse_loss_normalization: Optional[str]
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False
    scale_sparsity_penalty_by_decoder_norm: bool = False
    use_jacobian_loss: bool = False
    jacobian_coefficient: float = 5e2
    mlp_out_mse_coefficient: float = 1.0

    @classmethod
    def from_sae_runner_config(
        cls, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSAEConfig":

        return cls(
            # base config
            architecture=cfg.architecture,
            is_pair=cfg.use_jacobian_loss,
            d_in=cfg.d_in,
            d_sae=cfg.d_sae,  # type: ignore
            dtype=cfg.dtype,
            device=cfg.device,
            model_name=cfg.model_name,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            activation_fn_str=cfg.activation_fn,
            activation_fn_kwargs=cfg.activation_fn_kwargs,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            finetuning_scaling_factor=cfg.finetuning_method is not None,
            sae_lens_training_version=cfg.sae_lens_training_version,
            context_size=cfg.context_size,
            dataset_path=cfg.dataset_path,
            prepend_bos=cfg.prepend_bos,
            seqpos_slice=cfg.seqpos_slice,
            # Training cfg
            l1_coefficient=cfg.l1_coefficient,
            lp_norm=cfg.lp_norm,
            use_ghost_grads=cfg.use_ghost_grads,
            normalize_sae_decoder=cfg.normalize_sae_decoder,
            noise_scale=cfg.noise_scale,
            decoder_orthogonal_init=cfg.decoder_orthogonal_init,
            mse_loss_normalization=cfg.mse_loss_normalization,
            decoder_heuristic_init=cfg.decoder_heuristic_init,
            init_encoder_as_decoder_transpose=cfg.init_encoder_as_decoder_transpose,
            scale_sparsity_penalty_by_decoder_norm=cfg.scale_sparsity_penalty_by_decoder_norm,
            use_jacobian_loss=cfg.use_jacobian_loss,
            jacobian_coefficient=cfg.jacobian_coefficient,
            mlp_out_mse_coefficient=cfg.mlp_out_mse_coefficient,
            normalize_activations=cfg.normalize_activations,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAEConfig":
        # remove any keys that are not in the dataclass
        # since we sometimes enhance the config with the whole LM runner config
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }

        # ensure seqpos slice is tuple
        # ensure that seqpos slices is a tuple
        # Ensure seqpos_slice is a tuple
        if "seqpos_slice" in valid_config_dict:
            if isinstance(valid_config_dict["seqpos_slice"], list):
                valid_config_dict["seqpos_slice"] = tuple(
                    valid_config_dict["seqpos_slice"]
                )
            elif not isinstance(valid_config_dict["seqpos_slice"], tuple):
                valid_config_dict["seqpos_slice"] = (valid_config_dict["seqpos_slice"],)

        if (
            "use_jacobian_loss" in valid_config_dict
            and valid_config_dict["use_jacobian_loss"]
        ):
            valid_config_dict["is_pair"] = True

        return TrainingSAEConfig(**valid_config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "l1_coefficient": self.l1_coefficient,
            "lp_norm": self.lp_norm,
            "use_ghost_grads": self.use_ghost_grads,
            "normalize_sae_decoder": self.normalize_sae_decoder,
            "noise_scale": self.noise_scale,
            "decoder_orthogonal_init": self.decoder_orthogonal_init,
            "init_encoder_as_decoder_transpose": self.init_encoder_as_decoder_transpose,
            "mse_loss_normalization": self.mse_loss_normalization,
            "decoder_heuristic_init": self.decoder_heuristic_init,
            "scale_sparsity_penalty_by_decoder_norm": self.scale_sparsity_penalty_by_decoder_norm,
            "normalize_activations": self.normalize_activations,
            "use_jacobian_loss": self.use_jacobian_loss,
            "jacobian_coefficient": self.jacobian_coefficient,
            "mlp_out_mse_coefficient": self.mlp_out_mse_coefficient,
        }

    # this needs to exist so we can initialize the parent sae cfg without the training specific
    # parameters. Maybe there's a cleaner way to do this
    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "is_pair": self.is_pair,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation_fn_str": self.activation_fn_str,
            "activation_fn_kwargs": self.activation_fn_kwargs,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "dtype": self.dtype,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "device": self.device,
            "context_size": self.context_size,
            "prepend_bos": self.prepend_bos,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "normalize_activations": self.normalize_activations,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "sae_lens_training_version": self.sae_lens_training_version,
        }


class TrainingSAE(SAE):  # TODO rename to TrainingSAEPair
    """
    A SAE used for training. This class provides a `training_forward_pass` method which calculates
    losses used for training.
    """

    cfg: TrainingSAEConfig
    use_error_term: bool
    dtype: torch.dtype
    device: torch.device

    def __init__(
        self,
        cfg: TrainingSAEConfig,
        use_error_term: bool = False,
        transformer_block: Optional[TransformerBlock] = None,
        llm_cfg: Optional[HookedTransformerConfig] = None,
    ):
        base_sae_cfg = SAEConfig.from_dict(cfg.get_base_sae_cfg_dict())
        super().__init__(base_sae_cfg)
        self.cfg = cfg  # type: ignore

        self.encode_with_hidden_pre_fn = (
            self.encode_with_hidden_pre
            if cfg.architecture != "gated"
            else self.encode_with_hidden_pre_gated
        )

        self.check_cfg_compatibility()

        self.use_error_term = use_error_term

        self.initialize_weights_complex(False)

        # The training SAE will assume that the activation store handles
        # reshaping.
        self.turn_off_forward_pass_hook_z_reshaping()

        self.mse_loss_fn = self._get_mse_loss_fn()

        if cfg.use_jacobian_loss:
            self.initialize_weights_complex(True)
            assert llm_cfg is not None
            assert (
                cfg.architecture == "standard"
            ), "Jacobian loss is currently only supported with standard SAEs"
            assert (
                cfg.activation_fn_str == "topk"
            ), "Jacobian loss is currently only supported with topk due to efficiency concerns"
            assert transformer_block is not None and isinstance(
                transformer_block, TransformerBlock
            ), "You need to pass in a TransformerBlock into TrainingSAE to use the Jacobian loss"

            # copy the MLP weights into an MLPWithActGrads so we can use it in the Jacobian calc
            self.mlp = MLPWithActGrads(transformer_block.mlp.cfg)
            self.mlp.load_state_dict(transformer_block.mlp.state_dict())
            self.mlp.to(cfg.device)
            for param in self.mlp.parameters():
                param.requires_grad = False

            self.llm_cfg = llm_cfg
            assert (
                self.llm_cfg.n_key_value_heads is None
            ), "The type of model you're trying to use is not yet supported with Jacobian SAEs"
            assert (
                not self.llm_cfg.gated_mlp
            ), "Gated MLPs are not supported with Jacobian SAEs"  # TODO I gotta add support for this
            assert (
                self.llm_cfg.num_experts is None
            ), "MoE MLPs are not supported with Jacobian SAEs"
            # assert ( # TODO probably add this check
            #     self.llm_cfg.normalization_type is None
            #     or "Pre" in self.llm_cfg.normalization_type
            # ), "Only Pre normalization is supported"
            if self.llm_cfg.parallel_attn_mlp:
                assert not self.llm_cfg.use_attn_in
                assert not self.llm_cfg.use_split_qkv_input
                assert not self.llm_cfg.use_normalization_before_and_after
                self.pre_mlp_ln = transformer_block.ln2
                # TODO do we actually need the transfromer block? isn't it just the LN which we need regardless of parallel_attn_mlp
                for param in self.pre_mlp_ln.parameters():
                    param.requires_grad = False

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAE":
        return cls(TrainingSAEConfig.from_dict(config_dict))

    def check_cfg_compatibility(self):
        if self.cfg.architecture == "gated":
            assert (
                self.cfg.use_ghost_grads is False
            ), "Gated SAEs do not support ghost grads"
            assert self.use_error_term is False, "Gated SAEs do not support error terms"

    @overload
    def encode_standard(
        self,
        x: Float[torch.Tensor, "... d_in"],
        is_output_sae: bool,
        return_topk_indices: Literal[False],
    ) -> Float[torch.Tensor, "... d_sae"]: ...

    @overload
    def encode_standard(
        self,
        x: Float[torch.Tensor, "... d_in"],
        is_output_sae: bool,
        return_topk_indices: Literal[True],
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... k"]]: ...

    def encode_standard(
        self,
        x: Float[torch.Tensor, "... d_in"],
        is_output_sae: bool,
        return_topk_indices: bool = False,
    ) -> (
        Float[torch.Tensor, "... d_sae"]
        | tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... k"]]
    ):
        """
        Calcuate SAE features from inputs
        """
        if return_topk_indices:
            feature_acts, _, topk_indices = self.encode_with_hidden_pre_fn(
                x, is_output_sae, return_topk_indices=True
            )
            return feature_acts, topk_indices
        feature_acts, _ = self.encode_with_hidden_pre_fn(x, is_output_sae)
        return feature_acts

    @overload
    def encode_with_hidden_pre(
        self,
        x: Float[torch.Tensor, "... d_in"],
        is_output_sae: bool,
        return_topk_indices: Literal[False],
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]: ...

    @overload
    def encode_with_hidden_pre(
        self,
        x: Float[torch.Tensor, "... d_in"],
        is_output_sae: bool,
        return_topk_indices: Literal[True],
    ) -> tuple[
        Float[torch.Tensor, "... d_sae"],
        Float[torch.Tensor, "... d_sae"],
        Float[torch.Tensor, "... d_sae"],
    ]: ...

    def encode_with_hidden_pre(
        self,
        x: Float[torch.Tensor, "... d_in"],
        is_output_sae: bool,
        return_topk_indices: bool = False,
    ) -> (
        tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]
        | tuple[
            Float[torch.Tensor, "... d_sae"],
            Float[torch.Tensor, "... d_sae"],
            Int[torch.Tensor, "... k"],
        ]
    ):
        sae_in = self.process_sae_in(x, is_output_sae)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(
            sae_in @ self.get_W_enc(is_output_sae) + self.get_b_enc(is_output_sae)
        )
        hidden_pre_noised = hidden_pre + (
            torch.randn_like(hidden_pre) * self.cfg.noise_scale * self.training
        )

        if return_topk_indices:
            assert (
                self.cfg.activation_fn_str == "topk"
            ), "Return indices only makes sense with topk activation function"
            feature_acts, topk_indices = self.activation_fn(
                hidden_pre_noised, return_indices=True
            )
            feature_acts = self.hook_sae_acts_post(feature_acts)

            return feature_acts, hidden_pre_noised, topk_indices

        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised))

        return feature_acts, hidden_pre_noised

    def encode_with_hidden_pre_gated(
        self, x: Float[torch.Tensor, "... d_in"], is_output_sae: bool
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        raise NotImplementedError("Gated SAEs are not yet supported with Jacobian SAEs")
        sae_in = self.process_sae_in(x)

        # apply b_dec_to_input if using that method.
        sae_in = x - (self.get_b_dec(is_output_sae) * self.cfg.apply_b_dec_to_input)

        # Gating path with Heaviside step function
        gating_pre_activation = sae_in @ self.get_W_enc(is_output_sae) + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path with weight sharing
        magnitude_pre_activation = (
            sae_in @ (self.get_W_enc(is_output_sae) * self.r_mag.exp()) + self.b_mag
        )
        # magnitude_pre_activation_noised = magnitude_pre_activation + (
        #     torch.randn_like(magnitude_pre_activation) * self.cfg.noise_scale * self.training
        # )
        feature_magnitudes = self.activation_fn(
            magnitude_pre_activation
        )  # magnitude_pre_activation_noised)

        # Return both the gated feature activations and the magnitude pre-activations
        return (
            active_features * feature_magnitudes,
            magnitude_pre_activation,
        )  # magnitude_pre_activation_noised

    def forward(
        self, x: Float[torch.Tensor, "... d_in"], is_output_sae: bool
    ) -> Float[torch.Tensor, "... d_in"]:
        feature_acts, _ = self.encode_with_hidden_pre_fn(x, is_output_sae)
        sae_out = self.decode(feature_acts, is_output_sae)

        return sae_out

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        current_jacobian_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
        dead_neuron_mask2: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:
        sae_out, feature_acts, topk_indices, mse_loss, l1_loss = self.apply_sae(
            sae_in, False, current_l1_coefficient
        )

        # GHOST GRADS
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None:
            assert not self.cfg.use_jacobian_loss
            # first half of second forward pass
            _, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)
            ghost_grad_loss = self.calculate_ghost_grad_loss(
                x=sae_in,
                sae_out=sae_out,
                per_item_mse_loss=per_item_mse_loss,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
        else:
            ghost_grad_loss = 0.0

        if self.cfg.architecture == "gated":
            assert not self.cfg.use_jacobian_loss
            # Gated SAE Loss Calculation

            # Shared variables
            sae_in_centered = (
                self.reshape_fn_in(sae_in)
                - self.get_b_dec(False) * self.cfg.apply_b_dec_to_input
            )
            pi_gate = sae_in_centered @ self.get_W_enc(False) + self.b_gate
            pi_gate_act = torch.relu(pi_gate)

            # SFN sparsity loss - summed over the feature dimension and averaged over the batch
            l1_loss = (
                current_l1_coefficient
                * torch.sum(
                    pi_gate_act * self.get_W_dec(False).norm(dim=1), dim=-1
                ).mean()
            )

            # Auxiliary reconstruction loss - summed over the feature dimension and averaged over the batch
            via_gate_reconstruction = pi_gate_act @ self.get_W_dec(
                False
            ) + self.get_b_dec(False)
            aux_reconstruction_loss = torch.sum(
                (via_gate_reconstruction - sae_in) ** 2, dim=-1
            ).mean()

            loss = mse_loss + l1_loss + aux_reconstruction_loss

            jacobian_loss = torch.tensor(0.0)
        elif self.cfg.use_jacobian_loss:
            # Run the reconstructed activations through the MLP
            # mlp_out, mlp_act_grads = self.mlp(self.pre_mlp_ln(sae_out))
            #! Bypassing the first SAE for now
            # TODO gradually go from bypassing the first SAE to using it, ie always run both sae_in and sae_out through the MLP and have a scheduler go from 0 to 1 over time
            mlp_out, mlp_act_grads = self.mlp(self.pre_mlp_ln(sae_in))
            sae_out2, feature_acts2, topk_indices2, _mse_loss2, l1_loss2 = (
                self.apply_sae(mlp_out, True, current_l1_coefficient)
            )

            # Calculate the Jacobian
            # TODO make sure the LN doesn't mess this up
            wd1 = self.get_W_dec(False) @ self.mlp.W_in  # (d_sae, d_mlp)
            w2e = self.mlp.W_out @ self.get_W_enc(True)  # (d_mlp, d_sae)
            jacobian = einops.einsum(
                wd1[topk_indices],
                mlp_act_grads,
                w2e[:, topk_indices2],
                "... seq_pos k1 d_mlp, ... seq_pos d_mlp,"
                "d_mlp ... seq_pos k2 -> ... seq_pos k1 k2",
            )

            _jacobian_loss = jacobian.abs().mean()

            mse_loss2 = self.cfg.mlp_out_mse_coefficient * _mse_loss2

            jacobian_loss = current_jacobian_coefficient * _jacobian_loss

            loss = (
                mse_loss
                + l1_loss
                + ghost_grad_loss
                + jacobian_loss
                + mse_loss2
                + l1_loss2
            )

            # TODO aux loss a la Gao et al?
            aux_reconstruction_loss = torch.tensor(0.0)
        else:
            jacobian_loss = torch.tensor(0.0)
            mlp_out = torch.tensor(0.0)
            sae_out2 = torch.tensor(0.0)
            feature_acts2 = torch.tensor(0.0)
            mse_loss2 = torch.tensor(0.0)
            l1_loss2 = torch.tensor(0.0)
            aux_reconstruction_loss = torch.tensor(0.0)
            loss = mse_loss + l1_loss + ghost_grad_loss

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            sae_in2=mlp_out,
            sae_out2=sae_out2,
            feature_acts2=feature_acts2,
            loss=loss,
            mse_loss=mse_loss.item(),
            l1_loss=l1_loss.item(),
            ghost_grad_loss=(
                ghost_grad_loss.item()
                if isinstance(ghost_grad_loss, torch.Tensor)
                else ghost_grad_loss
            ),
            auxiliary_reconstruction_loss=aux_reconstruction_loss.item(),
            jacobian_loss=jacobian_loss.item(),
            mse_loss2=mse_loss2.item(),
            l1_loss2=l1_loss2.item(),
        )

    def apply_sae(
        self, sae_in: torch.Tensor, is_output_sae: bool, current_l1_coefficient: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Run through the SAE
        feature_acts, _, topk_indices = self.encode_with_hidden_pre_fn(
            sae_in, is_output_sae, return_topk_indices=True
        )
        sae_out = self.decode(feature_acts, is_output_sae)

        # Calculate the MSE and L1 losses
        mse_loss = self.mse_loss_fn(sae_out, sae_in).mean()
        if current_l1_coefficient == 0:
            l1_loss = torch.tensor(0.0)
        else:
            weighted_feature_acts = feature_acts * self.get_W_dec(is_output_sae).norm(
                dim=1
            )
            sparsity = weighted_feature_acts.norm(
                p=self.cfg.lp_norm, dim=-1
            )  # sum over the feature dimension
            l1_loss = (current_l1_coefficient * sparsity).mean()

        return sae_out, feature_acts, topk_indices, mse_loss, l1_loss

    def calculate_ghost_grad_loss(
        self,
        x: torch.Tensor,
        sae_out: torch.Tensor,
        per_item_mse_loss: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Ghost grads are not yet supported with Jacobian SAEs"
        )
        # 1.
        residual = x - sae_out
        l2_norm_residual = torch.norm(residual, dim=-1)

        # 2.
        # ghost grads use an exponentional activation function, ignoring whatever
        # the activation function is in the SAE. The forward pass uses the dead neurons only.
        feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
        ghost_out = (
            feature_acts_dead_neurons_only
            @ self.get_W_dec(is_output_sae)[dead_neuron_mask, :]
        )
        l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
        norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
        ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

        # 3. There is some fairly complex rescaling here to make sure that the loss
        # is comparable to the original loss. This is because the ghost grads are
        # only calculated for the dead neurons, so we need to rescale the loss to
        # make sure that the loss is comparable to the original loss.
        # There have been methodological improvements that are not implemented here yet
        # see here: https://www.lesswrong.com/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team#Improving_ghost_grads
        per_item_mse_loss_ghost_resid = self.mse_loss_fn(ghost_out, residual.detach())
        mse_rescaling_factor = (
            per_item_mse_loss / (per_item_mse_loss_ghost_resid + 1e-6)
        ).detach()
        per_item_mse_loss_ghost_resid = (
            mse_rescaling_factor * per_item_mse_loss_ghost_resid
        )

        return per_item_mse_loss_ghost_resid.mean()

    @torch.no_grad()
    def _get_mse_loss_fn(self) -> Any:
        def standard_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.mse_loss(preds, target, reduction="none")

        def batch_norm_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            target_centered = target - target.mean(dim=0, keepdim=True)
            normalization = target_centered.norm(dim=-1, keepdim=True)
            return torch.nn.functional.mse_loss(preds, target, reduction="none") / (
                normalization + 1e-6
            )

        if self.cfg.mse_loss_normalization == "dense_batch":
            return batch_norm_mse_loss_fn
        else:
            return standard_mse_loss_fn

    @classmethod
    def load_from_pretrained(
        cls,
        path: str,
        device: str = "cpu",
        dtype: str | None = None,
    ) -> "TrainingSAE":
        # get the config
        config_path = os.path.join(path, SAE_CFG_PATH)
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg_dict = handle_config_defaulting(cfg_dict)
        cfg_dict["device"] = device
        if dtype is not None:
            cfg_dict["dtype"] = dtype

        weight_path = os.path.join(path, SAE_WEIGHTS_PATH)
        cfg_dict, state_dict = read_sae_from_disk(
            cfg_dict=cfg_dict,
            weight_path=weight_path,
            device=device,
        )
        sae_cfg = TrainingSAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae

    def initialize_weights_complex(self, is_output_sae: bool):
        """ """

        if self.cfg.decoder_orthogonal_init:
            self.get_W_dec(is_output_sae).data = nn.init.orthogonal_(
                self.get_W_dec(is_output_sae).data.T
            ).T

        elif self.cfg.decoder_heuristic_init:
            self.set_W_dec(
                is_output_sae,
                nn.Parameter(
                    torch.rand(
                        self.cfg.d_sae,
                        self.cfg.d_in,
                        dtype=self.dtype,
                        device=self.device,
                    )
                ),
            )
            self.initialize_decoder_norm_constant_norm(False)
            # if self.cfg.use_jacobian_loss:
            #     self.initialize_decoder_norm_constant_norm(True)

        # Then we initialize the encoder weights (either as the transpose of decoder or not)
        if self.cfg.init_encoder_as_decoder_transpose:
            self.get_W_enc(is_output_sae).data = (
                self.get_W_dec(is_output_sae).data.T.clone().contiguous()
            )
        else:
            self.set_W_enc(
                is_output_sae,
                nn.Parameter(
                    torch.nn.init.kaiming_uniform_(
                        torch.empty(
                            self.cfg.d_in,
                            self.cfg.d_sae,
                            dtype=self.dtype,
                            device=self.device,
                        )
                    )
                ),
            )

        if self.cfg.normalize_sae_decoder:
            with torch.no_grad():
                # Anthropic normalize this to have unit columns
                self.set_decoder_norm_to_unit_norm(False)
                # if self.cfg.use_jacobian_loss:
                #     self.set_decoder_norm_to_unit_norm(True)

    ## Initialization Methods
    @torch.no_grad()
    def initialize_b_dec_with_precalculated(
        self, origin: torch.Tensor, is_output_sae: bool
    ):
        out = torch.tensor(origin, dtype=self.dtype, device=self.device)
        self.get_b_dec(is_output_sae).data = out

    @torch.no_grad()
    def initialize_b_dec_with_mean(
        self, all_activations: torch.Tensor, is_output_sae: bool
    ):
        previous_b_dec = self.get_b_dec(is_output_sae).clone().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.get_b_dec(is_output_sae).data = out.to(self.dtype).to(self.device)

    ## Training Utils
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self, is_output_sae: bool):
        self.get_W_dec(is_output_sae).data /= torch.norm(
            self.get_W_dec(is_output_sae).data, dim=1, keepdim=True
        )

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(
        self, is_output_sae: bool, norm: float = 0.1
    ):
        """
        A heuristic proceedure inspired by:
        https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        """
        # TODO: Parameterise this as a function of m and n

        # ensure W_dec norms at unit norm
        self.get_W_dec(is_output_sae).data /= torch.norm(
            self.get_W_dec(is_output_sae).data, dim=1, keepdim=True
        )
        self.get_W_dec(
            is_output_sae
        ).data *= norm  # will break tests but do this for now.

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self, is_output_sae: bool):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        assert self.get_W_dec(is_output_sae).grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.get_W_dec(is_output_sae).grad,
            self.get_W_dec(is_output_sae).data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.get_W_dec(is_output_sae).grad -= einops.einsum(
            parallel_component,
            self.get_W_dec(is_output_sae).data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
