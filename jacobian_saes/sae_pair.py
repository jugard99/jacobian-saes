import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple, TypeVar, Union, overload

import einops
import torch
from jaxtyping import Float, Int
from safetensors.torch import save_file
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint

from jacobian_saes.config import DTYPE_MAP
from jacobian_saes.toolkit.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    get_conversion_loader_name,
    handle_config_defaulting,
    read_sae_from_disk,
)
from jacobian_saes.toolkit.pretrained_saes_directory import (
    get_norm_scaling_factor,
    get_pretrained_saes_directory,
)

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"

T = TypeVar("T", bound="SAEPair")


@dataclass
class SAEPairConfig:
    # architecture details
    architecture: Literal["standard", "gated", "jumprelu"]

    # forward pass details.
    d_in: int
    d_sae: int
    activation_fn_str: str
    apply_b_dec_to_input: bool
    finetuning_scaling_factor: bool

    # dataset it was trained on details.
    context_size: int
    model_name: str
    hook_name: str
    hook_layer: int
    hook_head_index: Optional[int]
    prepend_bos: bool
    dataset_path: str
    dataset_trust_remote_code: bool
    normalize_activations: str

    # misc
    dtype: str
    device: str
    randomize_llm_weights: bool = False
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)
    is_pair: bool = False
    sparsity_metric: str = "l1"
    neuronpedia_id: Optional[str] = None
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)
    seqpos_slice: tuple[int | None, ...] = (None,)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEPairConfig":

        # rename dict:
        rename_dict = {  # old : new
            "hook_point": "hook_name",
            "hook_point_head_index": "hook_head_index",
            "hook_point_layer": "hook_layer",
            "activation_fn": "activation_fn_str",
            "use_jacobian_loss": "is_pair",
        }
        config_dict = {rename_dict.get(k, k): v for k, v in config_dict.items()}

        # use only config terms that are in the dataclass
        config_dict = {
            k: v
            for k, v in config_dict.items()
            if k in cls.__dataclass_fields__  # pylint: disable=no-member
        }

        if "seqpos_slice" in config_dict:
            config_dict["seqpos_slice"] = tuple(config_dict["seqpos_slice"])

        return cls(**config_dict)

    # def __post_init__(self):

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "is_pair": self.is_pair,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "dtype": self.dtype,
            "device": self.device,
            "model_name": self.model_name,
            "randomize_llm_weights": self.randomize_llm_weights,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "activation_fn_str": self.activation_fn_str,  # use string for serialization
            "activation_fn_kwargs": self.activation_fn_kwargs or {},
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "prepend_bos": self.prepend_bos,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "context_size": self.context_size,
            "normalize_activations": self.normalize_activations,
            "neuronpedia_id": self.neuronpedia_id,
            "model_from_pretrained_kwargs": self.model_from_pretrained_kwargs,
            "seqpos_slice": self.seqpos_slice,
        }


class SAEPair(HookedRootModule):
    """
    Core Sparse Autoencoder pair (SAEPair) class used for inference. For training, see `TrainingSAEPair`.
    """

    cfg: SAEPairConfig
    dtype: torch.dtype
    device: torch.device

    # analysis
    use_error_term: bool

    def __init__(
        self,
        cfg: SAEPairConfig,
        use_error_term: bool = False,
    ):
        super().__init__()

        self.cfg = cfg

        if cfg.model_from_pretrained_kwargs:
            warnings.warn(
                "\nThis SAE has non-empty model_from_pretrained_kwargs. "
                "\nFor optimal performance, load the model like so:\n"
                "model = HookedSAETransformer.from_pretrained_no_processing(..., **cfg.model_from_pretrained_kwargs)",
                category=UserWarning,
                stacklevel=1,
            )

        self.activation_fn = get_activation_fn(
            cfg.activation_fn_str, **cfg.activation_fn_kwargs or {}
        )
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)
        self.use_error_term = use_error_term

        if self.cfg.architecture == "standard":
            self.initialize_weights_basic(False)
            if self.cfg.is_pair:
                self.initialize_weights_basic(True)
            self.encode = self.encode_standard
        elif self.cfg.architecture == "gated":
            self.initialize_weights_gated()
            self.encode = self.encode_gated
        elif self.cfg.architecture == "jumprelu":
            self.initialize_weights_jumprelu()
            self.encode = self.encode_jumprelu
        else:
            raise ValueError(f"Invalid architecture: {self.cfg.architecture}")

        # handle presence / absence of scaling factor.
        if self.cfg.finetuning_scaling_factor:
            self.apply_finetuning_scaling_factor = (
                lambda x: x * self.finetuning_scaling_factor
            )
        else:
            self.apply_finetuning_scaling_factor = lambda x: x

        # set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()

        # handle hook_z reshaping if needed.
        # this is very cursed and should be refactored. it exists so that we can reshape out
        # the z activations for hook_z SAEs. but don't know d_head if we split up the forward pass
        # into a separate encode and decode function.
        # this will cause errors if we call decode before encode.
        if self.cfg.hook_name.endswith("_z"):
            self.turn_on_forward_pass_hook_z_reshaping()
        else:
            # need to default the reshape fns
            self.turn_off_forward_pass_hook_z_reshaping()

        # handle run time activation normalization if needed:
        if self.cfg.normalize_activations == "constant_norm_rescale":

            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                self.x_norm_coeff = (self.cfg.d_in**0.5) / x.norm(dim=-1, keepdim=True)
                x = x * self.x_norm_coeff
                return x

            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:  #
                x = x / self.x_norm_coeff
                del self.x_norm_coeff  # prevents reusing
                return x

            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out

        elif self.cfg.normalize_activations == "layer_norm":

            #  we need to scale the norm of the input and store the scaling factor
            def run_time_activation_ln_in(
                x: torch.Tensor, eps: float = 1e-5
            ) -> torch.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                x = x - mu
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + eps)
                self.ln_mu = mu
                self.ln_std = std
                return x

            def run_time_activation_ln_out(x: torch.Tensor, eps: float = 1e-5):
                return x * self.ln_std + self.ln_mu

            self.run_time_activation_norm_fn_in = run_time_activation_ln_in
            self.run_time_activation_norm_fn_out = run_time_activation_ln_out
        else:
            self.run_time_activation_norm_fn_in = lambda x: x
            self.run_time_activation_norm_fn_out = lambda x: x

        self.setup()  # Required for `HookedRootModule`s

    @property
    def W_enc(self):
        raise ValueError("Use get_W_enc instead")

    @property
    def b_enc(self):
        raise ValueError("Use get_b_enc instead")

    @property
    def W_dec(self):
        raise ValueError("Use get_W_dec instead")

    @property
    def b_dec(self):
        raise ValueError("Use get_b_dec instead")

    def get_W_enc(self, is_output_sae: bool) -> torch.Tensor:
        return self._W_enc_out if is_output_sae else self._W_enc

    def get_b_enc(self, is_output_sae: bool) -> torch.Tensor:
        return self._b_enc_out if is_output_sae else self._b_enc

    def get_W_dec(self, is_output_sae: bool) -> torch.Tensor:
        return self._W_dec_out if is_output_sae else self._W_dec

    def get_b_dec(self, is_output_sae: bool) -> torch.Tensor:
        return self._b_dec_out if is_output_sae else self._b_dec

    def set_W_enc(self, is_output_sae: bool, value: torch.Tensor):
        setattr(self, "_W_enc_out" if is_output_sae else "_W_enc", value)

    def set_b_enc(self, is_output_sae: bool, value: torch.Tensor):
        setattr(self, "_b_enc_out" if is_output_sae else "_b_enc", value)

    def set_W_dec(self, is_output_sae: bool, value: torch.Tensor):
        setattr(self, "_W_dec_out" if is_output_sae else "_W_dec", value)

    def set_b_dec(self, is_output_sae: bool, value: torch.Tensor):
        setattr(self, "_b_dec_out" if is_output_sae else "_b_dec", value)

    def initialize_weights_basic(self, is_output_sae: bool):

        # no config changes encoder bias init for now.
        self.set_b_enc(
            is_output_sae,
            nn.Parameter(
                torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            ),
        )

        # Start with the default init strategy:
        self.set_W_dec(
            is_output_sae,
            nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(
                        self.cfg.d_sae,
                        self.cfg.d_in,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            ),
        )

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

        # methdods which change b_dec as a function of the dataset are implemented after init.
        self.set_b_dec(
            is_output_sae,
            nn.Parameter(
                torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
            ),
        )

        # scaling factor for fine-tuning (not to be used in initial training)
        # TODO: Make this optional and not included with all SAEs by default (but maintain backwards compatibility)
        if self.cfg.finetuning_scaling_factor:
            self.finetuning_scaling_factor = nn.Parameter(
                torch.ones(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            )

    def initialize_weights_gated(self):
        raise NotImplementedError("Not implemented with Jacobian SAEs")

        # Initialize the weights and biases for the gated encoder
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_gate = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.r_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.b_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )

        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

    def initialize_weights_jumprelu(self):
        raise NotImplementedError("Not implemented with Jacobian SAEs")
        # The params are identical to the standard SAE
        # except we use a threshold parameter too
        self.threshold = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

    @overload
    def to(
        self: T,
        device: Optional[Union[torch.device, str]] = ...,
        dtype: Optional[torch.dtype] = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: torch.dtype, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: torch.Tensor, non_blocking: bool = ...) -> T: ...

    def to(self, *args: Any, **kwargs: Any) -> "SAEPair":  # type: ignore
        device_arg = None
        dtype_arg = None

        # Check args
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device_arg = arg
            elif isinstance(arg, torch.dtype):
                dtype_arg = arg
            elif isinstance(arg, torch.Tensor):
                device_arg = arg.device
                dtype_arg = arg.dtype

        # Check kwargs
        device_arg = kwargs.get("device", device_arg)
        dtype_arg = kwargs.get("dtype", dtype_arg)

        if device_arg is not None:
            # Convert device to torch.device if it's a string
            device = (
                torch.device(device_arg) if isinstance(device_arg, str) else device_arg
            )

            # Update the cfg.device
            self.cfg.device = str(device)

            # Update the .device property
            self.device = device

        if dtype_arg is not None:
            # Update the cfg.dtype
            self.cfg.dtype = str(dtype_arg)

            # Update the .dtype property
            self.dtype = dtype_arg

        # Call the parent class's to() method to handle all cases (device, dtype, tensor)
        return super().to(*args, **kwargs)

    # Basic Forward Pass Functionality.
    def forward(self, x: torch.Tensor, is_output_sae: bool) -> torch.Tensor:
        feature_acts = self.encode(x, is_output_sae)
        sae_out = self.decode(feature_acts, is_output_sae)

        # TEMP
        if self.use_error_term:
            with torch.no_grad():
                # Recompute everything without hooks to get true error term
                # Otherwise, the output with error term will always equal input, even for causal interventions that affect x_reconstruct
                # This is in a no_grad context to detach the error, so we can compute SAE feature gradients (eg for attribution patching). See A.3 in https://arxiv.org/pdf/2403.19647.pdf for more detail
                # NOTE: we can't just use `sae_error = input - x_reconstruct.detach()` or something simpler, since this would mean intervening on features would mean ablating features still results in perfect reconstruction.
                with _disable_hooks(self):
                    feature_acts_clean = self.encode(x, is_output_sae)
                    x_reconstruct_clean = self.decode(feature_acts_clean, is_output_sae)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            sae_out = sae_out + sae_error
        return self.hook_sae_output(sae_out)

    def encode_gated(
        self, x: Float[torch.Tensor, "... d_in"], is_output_sae: bool
    ) -> Float[torch.Tensor, "... d_sae"]:
        raise NotImplementedError("Not implemented with Jacobian SAEs")
        sae_in = self.process_sae_in(x)

        # Gating path
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path with weight sharing
        magnitude_pre_activation = self.hook_sae_acts_pre(
            sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        )
        feature_magnitudes = self.activation_fn(magnitude_pre_activation)

        feature_acts = self.hook_sae_acts_post(active_features * feature_magnitudes)

        return feature_acts

    def encode_jumprelu(
        self, x: Float[torch.Tensor, "... d_in"], is_output_sae: bool
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calculate SAE features from inputs
        """
        raise NotImplementedError("Not implemented with Jacobian SAEs")
        sae_in = self.process_sae_in(x)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        feature_acts = self.hook_sae_acts_post(
            self.activation_fn(hidden_pre) * (hidden_pre > self.threshold)
        )

        return feature_acts

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
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Int[torch.Tensor, "... k"]]: ...

    def encode_standard(
        self,
        x: Float[torch.Tensor, "... d_in"],
        is_output_sae: bool,
        return_topk_indices: bool = False,
    ) -> (
        Float[torch.Tensor, "... d_sae"]
        | tuple[Float[torch.Tensor, "... d_sae"], Int[torch.Tensor, "... k"]]
    ):
        """
        Calculate SAE features from inputs
        """
        sae_in = self.process_sae_in(x, is_output_sae)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(
            sae_in @ self.get_W_enc(is_output_sae) + self.get_b_enc(is_output_sae)
        )
        if return_topk_indices:
            assert (
                self.cfg.activation_fn_str == "topk"
            ), "Return indices only makes sense with topk activation function"
            feature_acts, topk_indices = self.activation_fn(
                hidden_pre, return_indices=True
            )
            feature_acts = self.hook_sae_acts_post(feature_acts)

            return feature_acts, topk_indices

        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))

        return feature_acts

    def process_sae_in(
        self, sae_in: Float[torch.Tensor, "... d_in"], is_output_sae: bool
    ) -> Float[torch.Tensor, "... d_sae"]:
        sae_in = sae_in.to(self.dtype)
        print(f"Sae in to process_sae_in: f{sae_in.shape}")
        sae_in = self.reshape_fn_in(sae_in)
        print(f"Sae in to process_sae_in post reshape: f{sae_in.shape}")
        sae_in = self.hook_sae_input(sae_in)
        print(f"Sae in to process_sae_in post hook input: f{sae_in.shape}")
        sae_in = self.run_time_activation_norm_fn_in(sae_in)
        print(f"Sae in to process_sae_in post activ norm fn: f{sae_in.shape}")
        sae_in = sae_in - (
            self.get_b_dec(is_output_sae) * self.cfg.apply_b_dec_to_input
        )
        return sae_in

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"], is_output_sae: bool
    ) -> Float[torch.Tensor, "... d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""
        # "... d_sae, d_sae d_in -> ... d_in",
        sae_out = self.hook_sae_recons(
            self.apply_finetuning_scaling_factor(feature_acts)
            @ self.get_W_dec(is_output_sae)
            + self.get_b_dec(is_output_sae)
        )

        # handle run time activation normalization if needed
        # will fail if you call this twice without calling encode in between.
        sae_out = self.run_time_activation_norm_fn_out(sae_out)

        # handle hook z reshaping if needed.
        sae_out = self.reshape_fn_out(sae_out, self.d_head)  # type: ignore

        return sae_out

    @torch.no_grad()
    def fold_W_dec_norm(self, is_output_sae: bool):
        W_dec_norms = self.get_W_dec(is_output_sae).norm(dim=-1).unsqueeze(1)
        self.get_W_dec(is_output_sae).data = (
            self.get_W_dec(is_output_sae).data / W_dec_norms
        )
        self.get_W_enc(is_output_sae).data = (
            self.get_W_enc(is_output_sae).data * W_dec_norms.T
        )
        if self.cfg.architecture == "gated":
            raise NotImplementedError("Not implemented with Jacobian SAEs")
            self.r_mag.data = self.r_mag.data * W_dec_norms.squeeze()
            self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
            self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()
        else:
            self.get_b_enc(is_output_sae).data = (
                self.get_b_enc(is_output_sae).data * W_dec_norms.squeeze()
            )

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(
        self, activation_norm_scaling_factor: float, is_output_sae: bool
    ):
        self.get_W_enc(is_output_sae).data = (
            self.get_W_enc(is_output_sae).data * activation_norm_scaling_factor
        )
        # previously weren't doing this.
        self.get_W_dec(is_output_sae).data = (
            self.get_W_dec(is_output_sae).data / activation_norm_scaling_factor
        )

        # once we normalize, we shouldn't need to scale activations.
        self.cfg.normalize_activations = "none"

    def save_model(self, path: str, sparsity: Optional[torch.Tensor] = None):

        if not os.path.exists(path):
            os.mkdir(path)

        # generate the weights
        save_file(self.state_dict(), f"{path}/{SAE_WEIGHTS_PATH}")

        # save the config
        config = self.cfg.to_dict()

        with open(f"{path}/{SAE_CFG_PATH}", "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            save_file(sparsity_in_dict, f"{path}/{SPARSITY_PATH}")  # type: ignore

    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu", dtype: str | None = None
    ) -> "SAEPair":

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

        sae_cfg = SAEPairConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        if not hasattr(sae, "mlp"):
            state_dict = {k: v for k, v in state_dict.items() if "mlp" not in k}
        sae.load_state_dict(state_dict)

        return sae

    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
    ) -> Tuple["SAEPair", dict[str, Any], Optional[torch.Tensor]]:
        """

        Load a pretrained SAE from the Hugging Face model hub.

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.
            return_sparsity_if_present: If True, will return the log sparsity tensor if it is present in the model directory in the Hugging Face model hub.
        """

        # get sae directory
        sae_directory = get_pretrained_saes_directory()

        # get the repo id and path to the SAE
        if release not in sae_directory:
            if "/" not in release:
                raise ValueError(
                    f"Release {release} not found in pretrained SAEs directory, and is not a valid huggingface repo."
                )
        elif sae_id not in sae_directory[release].saes_map:
            # If using Gemma Scope and not the canonical release, give a hint to use it
            if (
                "gemma-scope" in release
                and "canonical" not in release
                and f"{release}-canonical" in sae_directory
            ):
                canonical_ids = list(
                    sae_directory[release + "-canonical"].saes_map.keys()
                )
                # Shorten the lengthy string of valid IDs
                if len(canonical_ids) > 5:
                    str_canonical_ids = str(canonical_ids[:5])[:-1] + ", ...]"
                else:
                    str_canonical_ids = str(canonical_ids)
                value_suffix = f" If you don't want to specify an L0 value, consider using release {release}-canonical which has valid IDs {str_canonical_ids}"
            else:
                value_suffix = ""

            valid_ids = list(sae_directory[release].saes_map.keys())
            # Shorten the lengthy string of valid IDs
            if len(valid_ids) > 5:
                str_valid_ids = str(valid_ids[:5])[:-1] + ", ...]"
            else:
                str_valid_ids = str(valid_ids)

            raise ValueError(
                f"ID {sae_id} not found in release {release}. Valid IDs are {str_valid_ids}."
                + value_suffix
            )
        sae_info = sae_directory.get(release, None)
        config_overrides = sae_info.config_overrides if sae_info is not None else None
        neuronpedia_id = (
            sae_info.neuronpedia_id[sae_id] if sae_info is not None else None
        )

        conversion_loader_name = get_conversion_loader_name(sae_info)
        conversion_loader = NAMED_PRETRAINED_SAE_LOADERS[conversion_loader_name]

        cfg_dict, state_dict, log_sparsities = conversion_loader(
            release,
            sae_id=sae_id,
            device=device,
            force_download=False,
            cfg_overrides=config_overrides,
        )

        sae = cls(SAEPairConfig.from_dict(cfg_dict))
        sae.load_state_dict(state_dict)
        sae.cfg.neuronpedia_id = neuronpedia_id

        # Check if normalization is 'expected_average_only_in'
        if cfg_dict.get("normalize_activations") == "expected_average_only_in":
            norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
            if norm_scaling_factor is not None:
                sae.fold_activation_norm_scaling_factor(norm_scaling_factor, False)
                if getattr(sae.cfg, "use_jacobian_loss", False):
                    sae.fold_activation_norm_scaling_factor(norm_scaling_factor, True)
                cfg_dict["normalize_activations"] = "none"
            else:
                warnings.warn(
                    f"norm_scaling_factor not found for {release} and {sae_id}, but normalize_activations is 'expected_average_only_in'. Skipping normalization folding."
                )

        return sae, cfg_dict, log_sparsities

    def get_name(self):
        model_name = self.cfg.model_name
        if self.cfg.randomize_llm_weights:
            model_name += "-randomized"
        sae_name = f"sae_pair_{model_name}_{self.cfg.hook_layer}_{self.cfg.d_sae}"
        return sae_name

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEPair":
        return cls(SAEPairConfig.from_dict(config_dict))

    def turn_on_forward_pass_hook_z_reshaping(self):

        assert self.cfg.hook_name.endswith(
            "_z"
        ), "This method should only be called for hook_z SAEs."

        def reshape_fn_in(x: torch.Tensor):
            self.d_head = x.shape[-1]  # type: ignore
            self.reshape_fn_in = lambda x: einops.rearrange(
                x, "... n_heads d_head -> ... (n_heads d_head)"
            )
            return einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")

        self.reshape_fn_in = reshape_fn_in

        self.reshape_fn_out = lambda x, d_head: einops.rearrange(
            x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
        )
        self.hook_z_reshaping_mode = True

    def turn_off_forward_pass_hook_z_reshaping(self):
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x
        self.d_head = None
        self.hook_z_reshaping_mode = False


class TopK(nn.Module):
    def __init__(
        self, k: int, postact_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()
    ):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    @overload
    def forward(
        self, x: torch.Tensor, return_indices: Literal[False]
    ) -> torch.Tensor: ...

    @overload
    def forward(
        self, x: torch.Tensor, return_indices: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward(
        self, x: torch.Tensor, return_indices: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        if return_indices:
            return result, topk.indices
        return result


def get_activation_fn(
    activation_fn: str, **kwargs: Any
) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_fn == "relu":
        return torch.nn.ReLU()
    elif activation_fn == "tanh-relu":

        def tanh_relu(input: torch.Tensor) -> torch.Tensor:
            input = torch.relu(input)
            input = torch.tanh(input)
            return input

        return tanh_relu
    elif activation_fn == "topk":
        assert "k" in kwargs, "TopK activation function requires a k value."
        k = kwargs.get("k", 1)  # Default k to 1 if not provided
        postact_fn = kwargs.get(
            "postact_fn", nn.ReLU()
        )  # Default post-activation to ReLU if not provided

        return TopK(k, postact_fn)
    else:
        raise ValueError(f"Unknown activation function: {activation_fn}")


_blank_hook = nn.Identity()


@contextmanager
def _disable_hooks(sae: SAEPair):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae.hook_dict.keys():
            setattr(sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae.hook_dict.items():
            setattr(sae, hook_name, hook)
