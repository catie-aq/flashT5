from safetensors import safe_open
from safetensors.torch import save_file

import sys
import re

## Usage: python convert_huggingface_t5 <path_from_huggingface_model.safetensors> <path_to_output_model.safetensors>

tensors = {}
with safe_open(sys.argv[1], framework="pt", device=0) as f:
    for k in f.keys():
        new_k = re.sub(".layer.*.SelfAttention.q", ".self_attention_layer.self_attention.Wq", k)
        new_k = re.sub(".layer.*.SelfAttention.k", ".self_attention_layer.self_attention.Wk", new_k)
        new_k = re.sub(".layer.*.SelfAttention.v", ".self_attention_layer.self_attention.Wv", new_k)
        new_k = re.sub(".layer.*.SelfAttention.o", ".self_attention_layer.self_attention.o", new_k)
        new_k = re.sub(".layer.*.EncDecAttention.q", ".cross_attention_layer.cross_attention.Wq", new_k)
        new_k = re.sub(".layer.*.EncDecAttention.k", ".cross_attention_layer.cross_attention.Wk", new_k)
        new_k = re.sub(".layer.*.EncDecAttention.v", ".cross_attention_layer.cross_attention.Wv", new_k)
        new_k = re.sub(".layer.*.EncDecAttention.o", ".cross_attention_layer.cross_attention.o", new_k)
        new_k = re.sub(".layer.*.SelfAttention.relative_attention_bias.", ".self_attention_layer.self_attention.pe_encoding.relative_attention_bias.", new_k)
        new_k = new_k.replace(".layer.0.layer_norm.", ".self_attention_layer.layer_norm.")
        if "encoder" in new_k:
            new_k = new_k.replace(".layer.1.layer_norm.", ".ff_layer.layer_norm.")
        else:
            new_k = new_k.replace(".layer.1.layer_norm.", ".cross_attention_layer.layer_norm.")
        new_k = new_k.replace(".layer.2.layer_norm.", ".ff_layer.layer_norm.")
        new_k = re.sub(".layer.*.DenseReluDense.", ".ff_layer.", new_k)
        new_k = new_k.replace(".wi_", ".act.wi_")
        tensors[new_k] = f.get_tensor(k)

save_file(tensors, sys.argv[2], metadata={'format': 'pt'})
