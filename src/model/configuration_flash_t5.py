import sys
from collections import OrderedDict
from typing import Mapping
import logging

from transformers import T5Config

AUTO_MAP = {
    "AutoModel": "modeling_flash_t5.FlashT5ForConditionalGeneration"
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
        use_flash_attention=False,
        max_sequence_length=1024,
        attention_dropout_rate=0.0,
        alibi_mode="symetric",
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
