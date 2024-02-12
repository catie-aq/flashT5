import torch
import copy

from src.utils.fa2_lib.fa2_compilable import flash_attn_kvpacked_func, flash_attn_func, flash_attn_qkvpacked_func
from flash_attn import flash_attn_func as flash_attn_func_orig

class TestFA2(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward_ref(self, q, k, v, attn_bias):
        #return torch.nn.functional.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=attn_bias).permute(0, 2, 1, 3)
        out = flash_attn_func_orig(q, k, v, attn_bias=attn_bias)
        return out

    def forward(self, q, k, v, attn_bias):

        out = flash_attn_func(q, k, v, attn_bias=attn_bias)
        return out

    def forward_kv(self, q, kv, attn_bias):

        out = flash_attn_kvpacked_func(q, kv, attn_bias=attn_bias)
        return out

    def forward_qkv(self, qkv, attn_bias):

        out = flash_attn_qkvpacked_func(qkv, attn_bias=attn_bias)
        return out


head_dim=32
num_heads=12
seqlen=128
batch_size=16

model = TestFA2().cuda()
model_compiled = copy.deepcopy(model)
model_compiled = torch.compile(model_compiled)

q = torch.randn(batch_size, seqlen, num_heads, head_dim).cuda().to(torch.bfloat16)
q.requires_grad = True
k = torch.randn(batch_size, seqlen, num_heads, head_dim).cuda().to(torch.bfloat16)
k.requires_grad = True
v = torch.randn(batch_size, seqlen, num_heads, head_dim).cuda().to(torch.bfloat16)
v.requires_grad = True

kv = torch.randn(batch_size, seqlen, 2, num_heads, head_dim).cuda().to(torch.bfloat16)
kv.requires_grad = True

qkv = torch.randn(batch_size, seqlen, 3, num_heads, head_dim).cuda().to(torch.bfloat16)
qkv.requires_grad = True

attn_bias = torch.randn(batch_size, num_heads, seqlen, seqlen).cuda().to(torch.bfloat16)
attn_bias.requires_grad = True
#attn_bias = None

# Q, K, V
out_ref = model.forward_ref(q, k, v, attn_bias)
out = model_compiled(q, k, v, attn_bias)

print(torch.allclose(out, out_ref))

dout = torch.rand_like(out)

if attn_bias is not None:
    (dq, dk, dv, ds) = torch.autograd.grad(out, (q, k, v, attn_bias), dout)
    (dq_ref, dk_ref, dv_ref, ds_ref) = torch.autograd.grad(out_ref, (q, k, v, attn_bias), dout)
    print(torch.allclose(ds, ds_ref))
else:
    (dq, dk, dv) = torch.autograd.grad(out, (q, k, v), dout)
    (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(out_ref, (q, k, v), dout)

# KV packed
out_ref = model.forward_kv(q, kv, attn_bias)
out = model_compiled.forward_kv(q, kv, attn_bias)

print(torch.allclose(out, out_ref))

dout = torch.rand_like(out)

if attn_bias is not None:
    (dq, dkv, ds) = torch.autograd.grad(out, (q, kv, attn_bias), dout)
    (dq_ref, dkv_ref, ds_ref) = torch.autograd.grad(out_ref, (q, kv, attn_bias), dout)
    print(torch.allclose(ds, ds_ref))
else:
    (dq, dkv) = torch.autograd.grad(out, (q, kv), dout)
    (dq_ref, dkv_ref) = torch.autograd.grad(out_ref, (q, kv), dout)

print(torch.allclose(dq, dq_ref))
print(torch.allclose(dkv, dkv_ref))

# QKV packed

out_ref = model.forward_qkv(qkv, attn_bias)
out = model_compiled.forward_qkv(qkv, attn_bias)

print(torch.allclose(out, out_ref))

dout = torch.rand_like(out)

if attn_bias is not None:
    (dqkv, ds) = torch.autograd.grad(out, (qkv, attn_bias), dout)
    (dqkv_ref, ds_ref) = torch.autograd.grad(out_ref, (qkv, attn_bias), dout)
    print(torch.allclose(ds, ds_ref))
else:
    dqkv = torch.autograd.grad(out, qkv, dout)[0]
    dqkv_ref = torch.autograd.grad(out_ref, qkv, dout)[0]

print(torch.allclose(dqkv, dqkv_ref))
