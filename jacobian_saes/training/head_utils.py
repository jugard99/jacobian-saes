from transformer_lens import HookedTransformer
from jacobian_saes.training.activations_store import ActivationsStore
import torch
import einops

def attn_with_act_grads(self,
                                   E:torch.tensor):
    print(f"Attn with act grads has run with input E of shape: {E.shape}")

    model_name = self.cfg.model_name
    model = HookedTransformer.from_pretrained(model_name)
    # Get query, key and value matrices (Def 4 head just cuz of error) #self.cfg.hook_head_index]
    W_Q = model.W_Q[self.cfg.hook_layer][4]
    W_K = model.W_K[self.cfg.hook_layer][4]
    W_V = model.W_V[self.cfg.hook_layer][4]
    print(f"W_Q shape: {W_Q.shape}")
    # Get K and V values
    K = E @ W_K
    V = E @ W_V
    q = E @ W_Q
    print(f"E Shape: {E.shape}, K Shape: {K.shape}, W_K Shape: {W_K.shape}")
    print(f"q shape is, before sum: {q.shape}")
    # Actually get all queries
    # Now do einsum for attention pattern
    S = einops.einsum(q, K, "l1 d_h,l2 d_h->l1 l2")
    # Go l1 d_h, l2 d_h -> l1 l2
    # Softmax and jacobian of softmax
    # Apply causal mask
    mask = torch.triu(torch.ones(S.shape, dtype=torch.bool), diagonal=1).to(self.cfg.device)
    S.masked_fill_(mask, -1e9)
    print(S.shape)
    with torch.no_grad():
        A = torch.softmax(S, dim=-1)
    """jacA = torch.diag_embed(A) - einops.einsum(A,A,"l1 l2, l1 l3-> l1 l2 l3")
    jacA = jacA.sum(0)"""
    jacA = torch.sum(
        A.unsqueeze(2) * (torch.eye(A.shape[1], device=A.device) - A.unsqueeze(1)),
        dim=0
    )
    z = einops.einsum(A, V, "l1 l2,l2 d_h->l1 d_h")
    q = q.sum(0)
    z = z.sum(0)
    print(f"q shape after sum: {q.shape}")
    # l1 l2, l2 d_h -> l1 d_h
    return q, z, (V, K, jacA)


def compute_head_jacobian(
        self, V: torch.tensor, K: torch.tensor, jacA: torch.tensor, topk_indices: torch.tensor,
        topk_indices2: torch.tensor):
    W_dec = self.get_W_dec(False)
    W_enc = self.get_W_enc(True)
    print(f"W_enc shape: {W_enc.shape}")
    print(f"Topk indices 2 shape: {topk_indices2.shape}")
    wd1 = W_dec[topk_indices] @ V.T
    w2e = K @ W_enc[:, topk_indices2]
    J = einops.einsum(
        wd1, jacA, w2e,
        "d_s1 seq,seq seq2,seq2 d_s2-> d_s1 d_s2"
    )

    return J
