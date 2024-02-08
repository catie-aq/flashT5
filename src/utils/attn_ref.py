import torch

def attn_ref(q, k, v, b, sm_scale, dropout_p=0.0, causal=False, upcast=False):
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        b = b.float()

    if (b.shape[0] != q.shape[0]) or (b.shape[1] != q.shape[1]):
        b = b.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])

    M = torch.tril(torch.ones((q.shape[2], k.shape[2]), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3))
    p *= sm_scale
    if b is not None:
        p += b

    if causal:
        p[:, :, M == 0] = float("-inf")

    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    if dropout_p > 0.0:
        p = torch.dropout(p, dropout_p, train=True)

    ref_out = torch.matmul(p, v)
    return ref_out
