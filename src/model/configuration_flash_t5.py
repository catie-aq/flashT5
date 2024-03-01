import sys
from collections import OrderedDict
from typing import Mapping
import logging

from transformers import T5Config

AUTO_MAP = {
    "AutoModel": "modeling_flash_t5.FlashT5ForConditionalGeneration",
    "AutoModelForSeq2SeqLM": "modeling_flash_t5.FlashT5ForConditionalGeneration"
}

class FlashT5Config(T5Config):

    model_type = "flash_t5"

    def __init__(
        self,
        decoder_start_token_id=0,
        pad_token_id=-100,
        use_glu_mlp=False,
        position_encoding_type="t5",
        use_randomized_position_encoding=False,
        label_smoothing=0.0,
        z_loss=None,
        use_flash_attention=None,
        max_sequence_length=1024,
        attention_dropout_rate=0.0,
        alibi_mode="symetric",
        use_triton_layernorm=False,
        use_triton_crossentropy=False,
        use_triton_gated_mlp=False,
        use_gelu_act=True,
        use_full_bias_size=False,
        rotary_emb_fraction=1.0,
        rotary_base=10000,
        rotary_interleaved=False,
        rotary_scale_base=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.use_glu_mlp = use_glu_mlp
        self.position_encoding_type = position_encoding_type
        self.use_randomized_position_encoding = use_randomized_position_encoding
        self.label_smoothing = label_smoothing
        self.z_loss = z_loss
        self.use_flash_attention = use_flash_attention
        self.max_sequence_length = max_sequence_length
        self.alibi_mode = alibi_mode
        self.attention_dropout_rate = attention_dropout_rate
        self.use_triton_layernorm = use_triton_layernorm
        self.use_triton_crossentropy = use_triton_crossentropy
        self.use_triton_gated_mlp = use_triton_gated_mlp
        self.use_gelu_act = use_gelu_act
        self.use_full_bias_size = use_full_bias_size
        self.rotary_base = rotary_base
        self.rotary_interleaved = rotary_interleaved
        self.rotary_scale_base = rotary_scale_base
        self.rotary_emb_fraction = rotary_emb_fraction

        self.auto_map = AUTO_MAP

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

# Register model in Auto API
try:
    FlashT5Config.register_for_auto_class()
    for key, value in AUTO_MAP.items():
        str_to_class(value.split(".")[-1]).register_for_auto_class(key)
except:
    logging.warn("AutoRegister isn't available.")
