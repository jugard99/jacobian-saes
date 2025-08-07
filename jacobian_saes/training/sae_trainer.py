import contextlib
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule

import wandb
from jacobian_saes import __version__
from jacobian_saes.config import LanguageModelSAERunnerConfig
from jacobian_saes.evals import EvalConfig, run_evals
from jacobian_saes.training.activations_store import ActivationsStore
from jacobian_saes.training.optim import LinearScheduler, get_lr_scheduler
from jacobian_saes.training.training_sae_pair import TrainingSAEPair, TrainStepOutput

# used to map between parameters which are updated during finetuning and the config str.
FINETUNING_PARAMETERS = {
    "scale": ["scaling_factor"],
    "decoder": ["scaling_factor", "W_dec", "b_dec"],
    "unrotated_decoder": ["scaling_factor", "b_dec"],
}


@dataclass
class TrainSAEOutput:
    sae: TrainingSAEPair
    checkpoint_path: str
    log_feature_sparsities: torch.Tensor


class SAETrainer:
    """
    Core SAE class used for inference. For training, see TrainingSAE.
    """

    def __init__(
        self,
        model: HookedRootModule,
        sae: TrainingSAEPair,
        activation_store: ActivationsStore,
        save_checkpoint_fn,  # type: ignore
        cfg: LanguageModelSAERunnerConfig,
    ) -> None:

        self.model = model
        self.sae = sae
        self.activation_store = activation_store
        self.save_checkpoint = save_checkpoint_fn
        self.cfg = cfg

        self.n_training_steps: int = 0
        self.n_training_tokens: int = 0
        self.started_fine_tuning: bool = False

        self.checkpoint_thresholds = []
        if self.cfg.n_checkpoints > 0:
            self.checkpoint_thresholds = list(
                range(
                    0,
                    cfg.total_training_tokens,
                    cfg.total_training_tokens // self.cfg.n_checkpoints,
                )
            )[1:]

        self.act_freq_scores = torch.zeros(
            cast(int, cfg.d_sae),
            device=cfg.device,
        )
        self.n_forward_passes_since_fired = torch.zeros(
            cast(int, cfg.d_sae),
            device=cfg.device,
        )
        if self.cfg.use_jacobian_loss:
            self.act_freq_scores2 = torch.zeros(
                cast(int, cfg.d_sae),
                device=cfg.device,
            )
            self.n_forward_passes_since_fired2 = torch.zeros(
                cast(int, cfg.d_sae),
                device=cfg.device,
            )

        self.n_frac_active_tokens = 0
        # we don't train the scaling factor (initially)
        # set requires grad to false for the scaling factor
        for name, param in self.sae.named_parameters():
            if "scaling_factor" in name:
                param.requires_grad = False

        self.optimizer = Adam(
            sae.parameters(),
            lr=cfg.lr,
            betas=(
                cfg.adam_beta1,
                cfg.adam_beta2,
            ),
        )
        assert cfg.lr_end is not None  # this is set in config post-init
        self.lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler_name,
            lr=cfg.lr,
            optimizer=self.optimizer,
            warm_up_steps=cfg.lr_warm_up_steps,
            decay_steps=cfg.lr_decay_steps,
            training_steps=self.cfg.total_training_steps,
            lr_end=cfg.lr_end,
            num_cycles=cfg.n_restart_cycles,
        )
        self.l1_scheduler = LinearScheduler(
            warm_up_steps=cfg.l1_warm_up_steps,
            final_value=cfg.l1_coefficient,
        )
        self.jacobian_scheduler = LinearScheduler(
            warm_up_steps=cfg.jacobian_warm_up_steps,
            final_value=cfg.jacobian_coefficient,
        )

        # Setup autocast if using
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.autocast)

        if self.cfg.autocast:
            self.autocast_if_enabled = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.cfg.autocast,
            )
        else:
            self.autocast_if_enabled = contextlib.nullcontext()

        # Set up eval config

        self.trainer_eval_config = EvalConfig(
            batch_size_prompts=self.cfg.eval_batch_size_prompts,
            n_eval_reconstruction_batches=self.cfg.n_eval_batches,
            n_eval_sparsity_variance_batches=self.cfg.n_eval_batches,
            compute_ce_loss=True,
            compute_l2_norms=True,
            compute_sparsity_metrics=True,
            compute_variance_metrics=True,
            compute_kl=False,
            compute_featurewise_weight_based_metrics=False,
        )

    @property
    def feature_sparsity(self) -> torch.Tensor:
        return self.act_freq_scores / self.n_frac_active_tokens

    @property
    def feature_sparsity2(self) -> torch.Tensor:
        return self.act_freq_scores2 / self.n_frac_active_tokens

    @property
    def log_feature_sparsity(self) -> torch.Tensor:
        return torch.log10(self.feature_sparsity + 1e-10).detach().cpu()

    @property
    def log_feature_sparsity2(self) -> torch.Tensor:
        return torch.log10(self.feature_sparsity2 + 1e-10).detach().cpu()

    @property
    def current_l1_coefficient(self) -> float:
        return self.l1_scheduler.current_value

    @property
    def current_jacobian_coefficient(self) -> float:
        return self.jacobian_scheduler.current_value

    @property
    def dead_neurons(self) -> torch.Tensor:
        return (self.n_forward_passes_since_fired > self.cfg.dead_feature_window).bool()

    @property
    def dead_neurons2(self) -> torch.Tensor:
        return (
            self.n_forward_passes_since_fired2 > self.cfg.dead_feature_window
        ).bool()

    def fit(self) -> TrainingSAEPair:

        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE")

        self._estimate_norm_scaling_factor_if_needed()

        # Train loop
        while self.n_training_tokens < self.cfg.total_training_tokens:
            # Do a training step.
            layer_acts = self.activation_store.next_batch()[:, 0, :].to(self.sae.device)
            self.n_training_tokens += self.cfg.train_batch_size_tokens
            print(f"SAETrainer Layer acts shape: {layer_acts.shape}")
            step_output = self._train_step(sae=self.sae, sae_in=layer_acts)

            if self.cfg.log_to_wandb:
                self._log_train_step(step_output)
                self._run_and_log_evals()

            self._checkpoint_if_needed()
            self.n_training_steps += 1
            self._update_pbar(step_output, pbar)

            ### If n_training_tokens > sae_group.cfg.training_tokens, then we should switch to fine-tuning (if we haven't already)
            self._begin_finetuning_if_needed()

        # fold the estimated norm scaling factor into the sae weights
        if self.activation_store.estimated_norm_scaling_factor is not None:
            self.sae.fold_activation_norm_scaling_factor(
                self.activation_store.estimated_norm_scaling_factor, False
            )
            if self.cfg.use_jacobian_loss:
                if self.activation_store.estimated_norm_scaling_factor != 1.0:
                    raise NotImplementedError(
                        "We're not yet estimating the norm scaling factor for the post-MLP SAE"
                    )

        # save final sae group to checkpoints folder
        self.save_checkpoint(
            trainer=self,
            checkpoint_name=f"final_{self.n_training_tokens}",
            wandb_aliases=["final_model"],
        )

        pbar.close()
        return self.sae

    @torch.no_grad()
    def _estimate_norm_scaling_factor_if_needed(self) -> None:
        if self.cfg.normalize_activations == "expected_average_only_in":
            self.activation_store.estimated_norm_scaling_factor = (
                self.activation_store.estimate_norm_scaling_factor()
            )
            if self.cfg.use_jacobian_loss:
                raise NotImplementedError(
                    "We're not yet estimating the norm scaling factor for the post-MLP SAE"
                )
        else:
            self.activation_store.estimated_norm_scaling_factor = 1.0

    def _train_step(
        self,
        sae: TrainingSAEPair,
        sae_in: torch.Tensor,
    ) -> TrainStepOutput:

        sae.train()
        # Make sure the W_dec is still zero-norm
        if self.cfg.normalize_sae_decoder:
            sae.set_decoder_norm_to_unit_norm(False)
            if self.cfg.use_jacobian_loss:
                sae.set_decoder_norm_to_unit_norm(True)

        # log and then reset the feature sparsity every feature_sampling_window steps
        sample_every_n = (
            self.cfg.feature_sampling_window * self.cfg.gradient_accumulation_steps
        )
        if (self.n_training_steps + 1) % sample_every_n == 0:
            if self.cfg.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                wandb.log(sparsity_log_dict, step=self.n_training_steps)
            self._reset_running_sparsity_stats()

        # for documentation on autocasting see:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with self.autocast_if_enabled:

            train_step_output = self.sae.training_forward_pass(
                sae_in=sae_in,
                dead_neuron_mask=self.dead_neurons,
                dead_neuron_mask2=self.dead_neurons2,
                current_l1_coefficient=self.current_l1_coefficient,
                current_jacobian_coefficient=self.current_jacobian_coefficient,
            )

            with torch.no_grad():
                did_fire = (train_step_output.feature_acts > 0).float().sum(-1) > 0
                self.n_forward_passes_since_fired += 1
                self.n_forward_passes_since_fired[did_fire] = 0
                self.act_freq_scores += (
                    (train_step_output.feature_acts.abs() > 0).float().sum(0)
                )
                if self.cfg.use_jacobian_loss:
                    print(f"Output feature acts shape at did fire: {train_step_output.feature_acts2.shape}")
                    did_fire2 = (train_step_output.feature_acts2 > 0).float().sum(
                        -1
                    ) > 0
                    self.n_forward_passes_since_fired2 += 1
                    self.n_forward_passes_since_fired2[did_fire2] = 0
                    self.act_freq_scores2 += (
                        (train_step_output.feature_acts2.abs() > 0).float().sum(0)
                    )
                self.n_frac_active_tokens += self.cfg.train_batch_size_tokens

        # Divide by the number of accumulation steps
        loss = train_step_output.loss / self.cfg.gradient_accumulation_steps

        # Scaler will rescale gradients if autocast is enabled
        self.scaler.scale(loss).backward()  # loss.backward() if not autocasting

        # If it's the step where we should step the optimizer
        if (self.n_training_steps + 1) % self.cfg.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)  # needed to clip correctly
            # TODO: Work out if grad norm clipping should be in config / how to test it.
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            self.scaler.step(
                self.optimizer
            )  # just ctx.optimizer.step() if not autocasting
            self.scaler.update()

            if self.cfg.normalize_sae_decoder:
                sae.remove_gradient_parallel_to_decoder_directions(False)
                if self.cfg.use_jacobian_loss:
                    sae.remove_gradient_parallel_to_decoder_directions(True)

            self.optimizer.zero_grad()

        self.lr_scheduler.step()
        self.l1_scheduler.step()
        self.jacobian_scheduler.step()

        return train_step_output

    @torch.no_grad()
    def _log_train_step(self, step_output: TrainStepOutput):
        log_every_n = (
            self.cfg.wandb_log_frequency * self.cfg.gradient_accumulation_steps
        )
        if (self.n_training_steps + 1) % log_every_n == 0:
            wandb.log(
                self._build_train_step_log_dict(
                    output=step_output,
                    n_training_tokens=self.n_training_tokens,
                ),
                step=self.n_training_steps,
            )

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_tokens: int,
    ) -> dict[str, Any]:
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts

        # metrics for currents acts
        l0 = (feature_acts > 0).float().sum(-1).mean()
        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / total_variance

        if self.cfg.use_jacobian_loss:
            sae_in2 = output.sae_in2
            sae_out2 = output.sae_out2
            feature_acts2 = output.feature_acts2
            l0_2 = (feature_acts2 > 0).float().sum(-1).mean()
            per_token_l2_loss2 = (sae_out2 - sae_in2).pow(2).sum(dim=-1).squeeze()
            total_variance2 = (sae_in2 - sae_in2.mean(0)).pow(2).sum(-1)
            explained_variance2 = 1 - per_token_l2_loss2 / total_variance2
        else:
            l0_2 = torch.tensor(0.0)
            explained_variance2 = torch.tensor(0.0)

        log_dict = {
            # losses
            "losses/mse_loss": output.mse_loss,
            "losses/l1_loss": output.l1_loss
            / (self.current_l1_coefficient if self.current_l1_coefficient != 0 else 1),
            "losses/auxiliary_reconstruction_loss": output.auxiliary_reconstruction_loss,
            "losses/jacobian_loss": output.jacobian_loss
            / (
                self.current_jacobian_coefficient
                if self.current_jacobian_coefficient != 0
                else 1
            ),
            "losses/mse_loss2": output.mse_loss2
            / (
                self.cfg.mlp_out_mse_coefficient
                if self.cfg.mlp_out_mse_coefficient != 0
                else 1
            ),
            "losses/l1_loss2": output.l1_loss2
            / (self.current_l1_coefficient if self.current_l1_coefficient != 0 else 1),
            "losses/overall_loss": output.loss.item(),
            # variance explained
            "metrics/explained_variance": explained_variance.mean().item(),
            "metrics/explained_variance_std": explained_variance.std().item(),
            "metrics/explained_variance2": explained_variance2.mean().item(),
            "metrics/explained_variance_std2": explained_variance2.std().item(),
            "metrics/l0": l0.item(),
            "metrics/l0_2": l0_2.item(),
            # sparsity
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/mean_passes_since_fired2": self.n_forward_passes_since_fired2.mean().item(),
            "sparsity/dead_features": self.dead_neurons.sum().item(),
            "sparsity/dead_features2": self.dead_neurons2.sum().item(),
            "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
            "details/current_l1_coefficient": self.current_l1_coefficient,
            "details/current_jacobian_coefficient": self.current_jacobian_coefficient,
            "details/n_training_tokens": n_training_tokens,
        }
        # Log ghost grad if we're using them
        if self.cfg.use_ghost_grads:
            ghost_grad_loss = output.ghost_grad_loss
            if isinstance(ghost_grad_loss, torch.Tensor):
                ghost_grad_loss = ghost_grad_loss.item()

            log_dict["losses/ghost_grad_loss"] = ghost_grad_loss

        return log_dict

    @torch.no_grad()
    def _run_and_log_evals(self):
        # record loss frequently, but not all the time.
        eval_every_n = (
            self.cfg.wandb_log_frequency
            * self.cfg.eval_every_n_wandb_logs
            * self.cfg.gradient_accumulation_steps
        )
        if (self.n_training_steps + 1) % eval_every_n == 0:
            self.sae.eval()
            eval_metrics, _ = run_evals(
                sae=self.sae,
                activation_store=self.activation_store,
                model=self.model,
                eval_config=self.trainer_eval_config,
                model_kwargs=self.cfg.model_kwargs,
            )  # not calculating featurwise metrics here.

            # Remove eval metrics that are already logged during training
            eval_metrics.pop("metrics/explained_variance", None)
            eval_metrics.pop("metrics/explained_variance_std", None)
            eval_metrics.pop("metrics/l0", None)
            eval_metrics.pop("metrics/l1", None)
            eval_metrics.pop("metrics/mse", None)

            # Remove metrics that are not useful for wandb logging
            eval_metrics.pop("metrics/total_tokens_evaluated", None)

            W_dec_norm_dist = (
                self.sae.get_W_dec(False).detach().float().norm(dim=1).cpu().numpy()
            )
            eval_metrics["weights/W_dec_norms"] = wandb.Histogram(W_dec_norm_dist)  # type: ignore
            W_dec_out_norm_dist = (
                self.sae.get_W_dec(True).detach().float().norm(dim=1).cpu().numpy()
            )
            eval_metrics["weights/W_dec_norms2"] = wandb.Histogram(W_dec_out_norm_dist)  # type: ignore

            if self.sae.cfg.architecture == "standard":
                b_e_dist = self.sae.get_b_enc(False).detach().float().cpu().numpy()
                eval_metrics["weights/b_e"] = wandb.Histogram(b_e_dist)  # type: ignore
                b_e_out_dist = self.sae.get_b_enc(True).detach().float().cpu().numpy()
                eval_metrics["weights/b_e2"] = wandb.Histogram(b_e_out_dist)  # type: ignore
            elif self.sae.cfg.architecture == "gated":
                raise NotImplementedError("Not yet implemented for Jacobian SAEs")
                b_gate_dist = self.sae.b_gate.detach().float().cpu().numpy()
                eval_metrics["weights/b_gate"] = wandb.Histogram(b_gate_dist)  # type: ignore
                b_mag_dist = self.sae.b_mag.detach().float().cpu().numpy()
                eval_metrics["weights/b_mag"] = wandb.Histogram(b_mag_dist)  # type: ignore

            wandb.log(
                eval_metrics,
                step=self.n_training_steps,
            )
            self.sae.train()

    @torch.no_grad()
    def _build_sparsity_log_dict(self) -> dict[str, Any]:
        log_dict = {
            "metrics/mean_log10_feature_sparsity": self.log_feature_sparsity.mean().item(),
            "plots/feature_density_line_chart": wandb.Histogram(self.log_feature_sparsity.numpy()),  # type: ignore
            "sparsity/below_1e-5": (self.feature_sparsity < 1e-5).sum().item(),
            "sparsity/below_1e-6": (self.feature_sparsity < 1e-6).sum().item(),
        }

        if self.cfg.use_jacobian_loss:
            log_dict.update(
                {
                    "metrics/mean_log10_feature_sparsity2": self.log_feature_sparsity2.mean().item(),
                    "plots/feature_density_line_chart2": wandb.Histogram(self.log_feature_sparsity2.numpy()),  # type: ignore
                    "sparsity/below_1e-5_2": (self.feature_sparsity2 < 1e-5)
                    .sum()
                    .item(),
                    "sparsity/below_1e-6_2": (self.feature_sparsity2 < 1e-6)
                    .sum()
                    .item(),
                }
            )

        return log_dict

    @torch.no_grad()
    def _reset_running_sparsity_stats(self) -> None:

        self.act_freq_scores = torch.zeros(
            self.cfg.d_sae,  # type: ignore
            device=self.cfg.device,
        )
        if self.cfg.use_jacobian_loss:
            self.act_freq_scores2 = torch.zeros(
                self.cfg.d_sae,  # type: ignore
                device=self.cfg.device,
            )
        self.n_frac_active_tokens = 0

    @torch.no_grad()
    def _checkpoint_if_needed(self):
        if (
            self.checkpoint_thresholds
            and self.n_training_tokens > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint(
                trainer=self,
                checkpoint_name=self.n_training_tokens,
            )
            self.checkpoint_thresholds.pop(0)

    @torch.no_grad()
    def _update_pbar(self, step_output: TrainStepOutput, pbar: tqdm, update_interval: int = 100):  # type: ignore
        update_every_n = update_interval * self.cfg.gradient_accumulation_steps
        if self.n_training_steps % update_every_n == 0:
            description = f"{self.n_training_steps}| MSE {step_output.mse_loss:.1e} | "
            if self.cfg.use_jacobian_loss:
                description += f"MSE_out {step_output.mse_loss2:.1e} | "
                description += f"Jacobian {step_output.jacobian_loss:.1e}"
            else:
                description += f"L1 {step_output.l1_loss:.3f}"
            pbar.set_description(description)
            pbar.update(update_interval * self.cfg.train_batch_size_tokens)

    def _begin_finetuning_if_needed(self):
        if (not self.started_fine_tuning) and (
            self.n_training_tokens > self.cfg.training_tokens
        ):
            self.started_fine_tuning = True

            # finetuning method should be set in the config
            # if not, then we don't finetune
            if not isinstance(self.cfg.finetuning_method, str):
                return

            raise NotImplementedError(
                "Not yet implemented -- update FINETUNING_PARAMETERS"
            )

            for name, param in self.sae.named_parameters():
                if name in FINETUNING_PARAMETERS[self.cfg.finetuning_method]:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            self.finetuning = True
