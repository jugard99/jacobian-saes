from transformer_lens import HookedTransformer
from jacobian_saes.training.activations_store import ActivationsStore
import torch

def get_head_hooks(
        model_name: str,
        hook1: str,
        hook2: str,
        headindex: int
):
    # Gets requested hooks based on params
    tokengen = ActivationsStore._iterate_raw_dataset_tokens()
    model = HookedTransformer.from_pretrained(model_name)
    for tokens in tokengen:
        if isinstance(tokens,str):
            tokens = model.to_tokens(tokens)
        elif isinstance(tokens[0],str):
            tokens = "".join(tokens)
            tokens = model.to_tokens(tokens)
        else:
            tokens = torch.tensor(tokens).unsqueeze(0)
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: hook1 in name or hook2 in name)
        K = cache[hook1][:,:,headindex,:]
        V = cache[hook2][:,:,headindex,:]
        yield K, V
