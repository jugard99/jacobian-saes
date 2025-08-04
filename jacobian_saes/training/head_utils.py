from transformer_lens import HookedTransformer
from jacobian_saes.training.activations_store import ActivationsStore
import torch

def get_head(
        model_name: str,
        hook:str,
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
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: hook in name)
        hooked = cache[hook][:,:,headindex,:]
        yield hooked

def get_W_E(
        model_name: str
):
    model = HookedTransformer.from_pretrained(model_name)
    tokengen = ActivationsStore._iterate_raw_dataset_tokens()
    for tokens in tokengen:
        if isinstance(tokens,str):
            tokens = model.to_tokens(tokens)
        elif isinstance(tokens[0],str):
            tokens = "".join(tokens)
            tokens = model.to_tokens(tokens)
        else:
            tokens = torch.tensor(tokens).unsqueeze(0)
        E = model.W_E[tokens]
        yield E

