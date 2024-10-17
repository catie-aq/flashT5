# From: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

import copy
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.modeling_utils import ModuleUtilsMixin
from transformers.modeling_outputs import ModelOutput, Seq2SeqModelOutput, BaseModelOutput, Seq2SeqLMOutput
from transformers import PreTrainedModel

try:
    from .ops.rms_norm import fast_rms_layernorm
except ImportError:
    fast_rms_layernorm = None

try:
    from .ops.cross_entropy_loss import cross_entropy_loss as fast_cross_entropy_loss
except ImportError:
    fast_cross_entropy_loss = None

try:
    from .ops.flash_attention_v2_bias import flash_attention_v2_bias
except ImportError:
    flash_attention_v2_bias = None

try:
    from flash_attn import flash_attn_kvpacked_func, flash_attn_func
except ImportError:
    flash_attn_kvpacked_func, flash_attn_func = None, None

from ..utils.attn_ref import attn_ref

from .configuration_flash_t5 import FlashT5Config
from ..utils.positional_encoding import ALiBiPositionalEncoding, RelativePositionalEncoding, RotaryPositionalEncoding, FIRE

class FlashT5CrossEntropyLoss(nn.Module):
    def __init__(self, z_loss_factor=0.0, label_smoothing=0.0, use_triton_crossentropy=False, inplace_backward=False):

        super().__init__()

        if use_triton_crossentropy and fast_cross_entropy_loss is None:
            raise ImportError("fast_cross_entropy_loss is not available")

        self.use_triton_crossentropy = use_triton_crossentropy
        self.z_loss_factor = z_loss_factor
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward

        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def compute_zloss(self, logits: torch.Tensor, z_loss: float):
        logits_sum = torch.logsumexp(logits, dim=-1, keepdim=True)
        log_z = torch.squeeze(logits_sum, axis=-1)
        total_z_loss = z_loss * torch.square(log_z)
        return total_z_loss.mean()

    def forward(self, logits, labels):

        if self.use_triton_crossentropy:
            return fast_cross_entropy_loss(logits, labels, \
                                           lse_square_scale=self.z_loss_factor, \
                                           label_smoothing=self.label_smoothing, \
                                           inplace_backward=self.inplace_backward \
                                          )[0].mean()

        # use standard method
        batch, seq_len, d = logits.shape
        logits_flatten = logits.float().view(batch*seq_len, d) # Must cast to float32 for numerical stability
        labels_flatten = labels.view(-1)
        loss = self.cross_entropy_loss(logits_flatten, labels_flatten)
        z_loss = 0.0
        if self.z_loss_factor != 0.0:
            z_loss = self.compute_zloss(logits_flatten[labels_flatten != -100],
                                   z_loss=self.z_loss_factor)
        return loss + z_loss

class FlashT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, use_triton_layernorm=False):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()

        if use_triton_layernorm and fast_rms_layernorm is None:
            raise ImportError("fast_rms_layernorm is not available")

        self.use_triton_layernorm = use_triton_layernorm
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        if self.use_triton_layernorm:
            return fast_rms_layernorm(hidden_states, self.weight, self.variance_epsilon)

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class FlashT5DenseAct(nn.Module):
    def __init__(self, config: FlashT5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = torch.nn.GELU(approximate='tanh') if config.use_gelu_act else torch.nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        return hidden_states

class FlashT5DenseGatedAct(nn.Module):
    def __init__(self, config: FlashT5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = torch.nn.GELU(approximate='tanh') if config.use_gelu_act else torch.nn.ReLU()

        self.use_gelu_act = config.use_gelu_act

    def forward(self, hidden_states):

        hidden_act = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_act * hidden_linear
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class FlashT5LayerFF(nn.Module):
    def __init__(self, config: FlashT5Config):
        super().__init__()
        if config.use_glu_mlp:
            self.act = FlashT5DenseGatedAct(config)
        else:
            self.act = FlashT5DenseAct(config)

        self.layer_norm = FlashT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon, use_triton_layernorm=config.use_triton_layernorm)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states).type_as(hidden_states)
        forwarded_states = self.act(forwarded_states)
        forwarded_states = self.wo(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class FlashT5Attention(nn.Module, ModuleUtilsMixin):
    def __init__(self, config: FlashT5Config, has_positional_encoding=False, is_causal=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_positional_encoding = has_positional_encoding
        self.is_causal = is_causal
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.p_dropout = config.attention_dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.attention_type = config.attention_type
        self.position_encoding_type = config.position_encoding_type
        self.max_sequence_length = config.max_sequence_length
        self.softmax_scale = config.attention_scale if config.attention_scale is not None else 1.0/math.sqrt(self.n_heads)
        self.use_full_bias_size = config.use_full_bias_size
        self.use_masking = config.use_masking

        if self.use_masking and not self.use_full_bias_size:
            raise ValueError("Masking can only be used with full batch size.")

        if self.attention_type == "triton" and flash_attention_v2_bias is None:
            raise ImportError("flash_attention_triton is not available")
        elif self.attention_type.startswith("fa2") and flash_attn_func is None:
            raise ImportError("Flash Attention 2 is not available")

        if self.attention_type == "fa2_rpe" and self.position_encoding_type != "t5":
             raise ValueError("fa2_rpe is not compatible with non-T5 position encoding")

        assert (self.p_dropout == 0.0) or (self.attention_type != "triton"), "Triton attention does not support dropout"

        self.pe_encoding = None
        if self.position_encoding_type == "ALiBi" and has_positional_encoding:
            # build alibi matrix with an upper bound on seq length
            self.pe_encoding = ALiBiPositionalEncoding(self.max_sequence_length,
                                                       self.n_heads,
                                                       config.alibi_mode,
                                                       randomized_position=config.use_randomized_position_encoding)
        elif self.position_encoding_type == "t5" and has_positional_encoding:
            self.pe_encoding = RelativePositionalEncoding(self.relative_attention_num_buckets,
                                                          self.relative_attention_max_distance,
                                                          self.n_heads,
                                                          self.max_sequence_length,
                                                          bidirectional=(not self.is_decoder),
                                                          randomized_position=config.use_randomized_position_encoding)
        elif self.position_encoding_type == "RoPE":
            self.pe_encoding = RotaryPositionalEncoding(int(self.key_value_proj_dim * config.rotary_emb_fraction),
                                                        self.max_sequence_length,
                                                        config.rotary_base,
                                                        config.rotary_interleaved,
                                                        config.rotary_scale_base,
                                                        randomized_position=config.use_randomized_position_encoding)
        elif self.position_encoding_type == "FIRE" and has_positional_encoding:
            self.pe_encoding = FIRE(num_heads=self.n_heads,
                                    mlp_width=config.fire_mlp_width,
                                    init_c=0.1,
                                    init_L=self.relative_attention_max_distance)

        self.Wq = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.Wk = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.Wv = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        batch_size, seq_length = hidden_states.shape[:2]
        key_length = seq_length if key_value_states is None else key_value_states.shape[1]
        q = self.Wq(hidden_states)
        if key_value_states is None:
            k = self.Wk(hidden_states)
            v = self.Wv(hidden_states)
        else:
            k = self.Wk(key_value_states)
            v = self.Wv(key_value_states)

        q = q.view(batch_size, seq_length, self.n_heads, self.key_value_proj_dim)
        k = k.view(batch_size, key_length, self.n_heads, self.key_value_proj_dim)
        v = v.view(batch_size, key_length, self.n_heads, self.key_value_proj_dim)

        if position_bias is None and self.pe_encoding is not None and self.attention_type != "fa2_rpe":
            q, k, v, position_bias = self.pe_encoding(q, k, v)

        if position_bias is not None and self.use_full_bias_size:
            position_bias = position_bias.expand(q.shape[0], q.shape[2], q.shape[1], k.shape[1])
            if self.attention_type == "fa2_bias" or self.attention_type == "triton":
                position_bias = position_bias.contiguous()

        if position_bias is not None and mask is not None and self.use_masking:
            mask = mask.unsqueeze(1)
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(3)
            position_bias = torch.where(mask, position_bias, torch.finfo(hidden_states.dtype).min)

        if self.attention_type == "fa2_bias":
            output = flash_attn_func(q, k, v, dropout_p=self.p_dropout, softmax_scale=self.softmax_scale, \
                                    attn_bias=position_bias, causal=self.is_causal)
        elif self.attention_type == "fa2_rpe":
            output = flash_attn_func(q, k, v, dropout_p=self.p_dropout, softmax_scale=self.softmax_scale, \
                                    rpe_weights=self.pe_encoding.relative_attention_bias.weight.t(), \
                                    rpe_max_distance=self.relative_attention_max_distance, \
                                    causal=self.is_causal)
        elif self.attention_type == "triton":
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            output = flash_attention_v2_bias(q, k, v, position_bias, self.is_causal, self.softmax_scale)
            output = output.permute(0, 2, 1, 3)
        else: # use flash attention
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            output = attn_ref(q, k, v, position_bias, dropout_p=self.p_dropout, sm_scale=self.softmax_scale, causal=self.is_causal)
            output = output.permute(0, 2, 1, 3)

        output = self.o(output.reshape(output.shape[0], output.shape[1], self.inner_dim))
        return (output, position_bias)


class FlashT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_positional_encoding=False):
        super().__init__()
        self.self_attention = FlashT5Attention(config, has_positional_encoding=has_positional_encoding, is_causal=config.is_decoder)
        self.layer_norm = FlashT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon, use_triton_layernorm=config.use_triton_layernorm)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states).type_as(hidden_states)
        attention_output = self.self_attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class FlashT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attention = FlashT5Attention(config, has_positional_encoding=False)
        self.layer_norm = FlashT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon, use_triton_layernorm=config.use_triton_layernorm)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.cross_attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]
        return outputs


class FlashT5Block(nn.Module):
    def __init__(self, config, has_positional_encoding=False):
        super().__init__()
        self.is_decoder = config.is_decoder

        self.self_attention_layer = FlashT5LayerSelfAttention(config, has_positional_encoding=has_positional_encoding)

        if self.is_decoder:
            self.cross_attention_layer = FlashT5LayerCrossAttention(config)

        self.ff_layer = FlashT5LayerFF(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
    ):
        self_attention_outputs = self.self_attention_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[1:]  # Relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.cross_attention_layer(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
            )
            hidden_states = cross_attention_outputs[0]

            # Keep relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Apply Feed Forward layer
        hidden_states = self.ff_layer(hidden_states)

        outputs = (hidden_states,) + attention_outputs
        return outputs  # hidden-states, (self-attention position bias), (cross-attention position bias)

class FlashT5Stack(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, embed_tokens):
        super().__init__()
        assert embed_tokens is not None

        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [FlashT5Block(config, has_positional_encoding=bool(i == 0)) for i in range(config.num_layers)]
        )

        self.final_layer_norm = FlashT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon, use_triton_layernorm=config.use_triton_layernorm)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
    ) -> BaseModelOutput:
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if torch.is_autocast_enabled() and input_ids.device.type == 'cuda':
            inputs_embeds = inputs_embeds.to(torch.get_autocast_gpu_dtype())

        # Masking
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device, dtype=torch.bool)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.bool
            )

        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for _, layer_module in enumerate(self.block):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
            )

            # We share the position biases between the layers - the first layer store them
            position_bias = layer_outputs[1]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[2]

            hidden_states = layer_outputs[0]

        hidden_states = self.final_layer_norm(hidden_states).type_as(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states
        )

class FlashT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FlashT5Config
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["FlashT5Block"]
    _keep_in_fp32_modules = []

    def _init_weights(self, module):
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, FlashT5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (FlashT5ForConditionalGeneration)):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** -0.5)
        elif isinstance(module, FlashT5DenseGatedAct):
            d_ff, d_model = module.wi_0.weight.data.size()
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
        elif isinstance(module, FlashT5LayerFF):
            d_ff, d_model = module.wo.weight.data.size()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((d_ff) ** -0.5))
        elif isinstance(module, FlashT5Attention):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.Wq.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.Wk.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.Wv.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_positional_encoding:
                if hasattr(module.pe_encoding, "relative_attention_bias"):
                    module.pe_encoding.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class FlashT5Model(FlashT5PreTrainedModel):

    def __init__(self, config: FlashT5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = FlashT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FlashT5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask
        )

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
        )

class FlashT5ForConditionalGeneration(FlashT5PreTrainedModel):

    def __init__(self, config: FlashT5Config):
        super().__init__(config)
        config.is_encoder_decoder = False
        assert not config.tie_word_embeddings

        self.config = config
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        self.encoder = FlashT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = FlashT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.loss_fct = FlashT5CrossEntropyLoss(z_loss_factor=config.z_loss,
                                                label_smoothing=config.label_smoothing,
                                                use_triton_crossentropy=config.use_triton_crossentropy,
                                                inplace_backward=config.crossentropy_inplace_backward)

        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # do nothing
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        return model_inputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_length = 32,
        **kwargs,
    ) -> torch.LongTensor:
        """
            input_ids: B x L_encoder, int64
            attention_mask: B x L_encoder, int64
                1 for tokens to attend to, 0 for tokens to ignore

            Generation:
                Starts with 0, ends with 1, padding is 0

            # For 20 input/outputs, the diff between my implementation and HF is 9.8s vs 11.4s
        """
        B, _ = input_ids.size()
        labels = torch.zeros(B, 1, dtype=torch.long, device=input_ids.device)
        encoder_hidden_states = None

        for _ in range(max_length):
            out = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=labels,
                encoder_hidden_states=encoder_hidden_states,
            )
            encoder_hidden_states = out.encoder_hidden_states
            top_labels = out.logits[:, -1].argmax(-1).unsqueeze(-1)
            labels = torch.cat([labels, top_labels], dim=-1)

            if (labels == 1).sum(-1).clamp(min=0, max=1).sum().item() == B:
                break

        labels[:, -1] = 1

        # Mask out the padding, i.e., all positions after the first 1 with 0
        B, L = labels.size()
        mask = torch.arange(L, device=labels.device).unsqueeze(0) <= (labels == 1).long().argmax(-1).unsqueeze(-1)
        labels = labels.masked_fill(~mask, 0)

        return labels

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> Seq2SeqLMOutput:
        """
            input_ids: B x L_encoder, int64
            attention_mask: B x L_encoder, int64
                1 for tokens to attend to, 0 for tokens to ignore
            labels: B x L_decoder, int64
        """
        if encoder_hidden_states is None:
            encoder_hidden_states = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]

        hidden_states = encoder_hidden_states

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(lm_logits, labels)

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            encoder_hidden_states=encoder_hidden_states
        )
