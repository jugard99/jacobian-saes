import argparse
import os
import sys

import torch

# sys.path.append(os.path.join(os.getcwd(), "SAELens"))
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

parser = argparse.ArgumentParser(description="Train a Jacobian SAE")
parser.add_argument("--jacobian-coeff", "-j", type=float, default=3e3, help="Jacobian coefficient")
parser.add_argument("--lr", "-l", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--tokens", "-t", type=float, default=300_000_000, help="Total number of training tokens")
parser.add_argument("--no-norm", dest="norm", action="store_false", help="Disable normalization")
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available(): # Doesn't work right now for some reason
#     device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


total_training_tokens = int(args.tokens)
batch_size = 4096
total_training_steps = (total_training_tokens // batch_size) + 1
total_training_tokens = total_training_steps * batch_size # to get the exact number
context_size = 1024

lr_warm_up_steps = total_training_steps // 100  # 1% of training
lr_decay_steps = total_training_steps // 5  # 20% of training
jacobian_warm_up_steps = total_training_steps // 20  # 5% of training

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="pythia-70m",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_name="blocks.3.hook_resid_pre",
    hook_layer=3,
    d_in=512,
    activation_fn="topk",
    activation_fn_kwargs={"k": 32},
    dataset_path="apollo-research/monology-pile-uncopyrighted-tokenizer-EleutherAI-gpt-neox-20b",
    is_dataset_tokenized=True,
    streaming=True,
    # SAE Parameters
    expansion_factor=32,
    b_dec_init_method="zeros",
    apply_b_dec_to_input=False,
    normalize_sae_decoder=True,
    # scale_sparsity_penalty_by_decoder_norm=True,
    # decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    normalize_activations="expected_average_only_in" if args.norm else "none",
    # Training Parameters
    lr=args.lr,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    l1_coefficient=0,  # we're using TopK so we don't need this
    use_jacobian_loss=True,
    jacobian_coefficient=args.jacobian_coeff,
    jacobian_warm_up_steps=jacobian_warm_up_steps,
    train_batch_size_tokens=batch_size,
    context_size=context_size,
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=16,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-6,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=True,
    wandb_project="jacobian_saes_jac_coef_sweep1",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)

sparse_autoencoder = SAETrainingRunner(cfg).run()
