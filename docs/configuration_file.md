# Configuration File Documentation

This document describes the structure and options available in the configuration file. We focus only on
additional options. We refer to the [T5 documentation](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Config) for the other options.


  - `attention_dropout_rate`: Dropout rate for the attention layer (float, between 0 and 1).

  - `dropout_rate`: Dropout rate for the MLP (float, between 0 and 1).

  - `d_ff`: Dimensionality of the feedforward layer (int).

  - `d_kv`: Dimensionality of the key and value vectors (int).

  - `d_model`: Dimensionality of the model (int).

  - `decoder_start_token_id`: Id of the token used as the decoder start token (int).

  - `label_smoothing`: Label smoothing value to pass to the crossentropy (float, between 0 and 1).

  - `max_sequence_length`: Maximum sequence length - currently not used (int).

  - `model_type`: Type of the model (string).

  - `num_heads`: Number of attention heads (int).

  - `num_layers`: Number of layers (int).

  - `position_encoding_type`: Type of the position encoding (string).

  - `relative_attention_max_distance`: Maximum distance for relative attention (int) - for T5 RPE position encoding only.

  - `relative_attention_num_buckets`: Number of buckets for relative attention (int) - for T5 RPE position encoding only.

  - `alibi_mode`: Mode for the alibi bias (string) - for alibi position encoding only. Could be `symetric` or `asymetric` for bidirectional attention.

  - `rotary_emb_fraction`: Fraction of the rotary embeddings to use (float, between 0 and 1) - for RoPE position encoding only.

  - `rotary_base`: Base for the rotary embeddings (int) - for RoPE position encoding only.

  - `rotary_interleaved`: Whether to use an interleaved rotary embedding (bool) - for RoPE position encoding only.

  - `rotary_scale_base`: Base for the rotary scale (float, between 0 and 1) - for RoPE position encoding only.

  - `fire_mlp_width`: Width of the MLP in the FIRE position encoding (int) - for FIRE position encoding only.

  - `tie_word_embeddings`: Whether to tie the word embeddings (bool).

  - `attention_type`: Type of the attention (string). Could be `fa2_rpe` to use the FlashAttention2 with T5 RPE patch or `fa2_bias` to use the FlashAttention2 with the bias patch, `triton` to use the Triton flash-attention with bias support or `ref` to use a reference implementation of attention in Torch (default).

  - `use_glu_mlp`: Whether to use the GLU MLP (bool).

  - `use_randomized_position_encoding`: Whether to use randomized position encoding (bool).

  - `z_loss`: Z-loss value (float, between 0 and 1).

  - `use_triton_layernorm`: Whether to use the Triton layernorm (bool).

  - `use_triton_crossentropy`: Whether to use the Triton crossentropy (bool).

  - `crossentropy_inplace_backward`: Inplace backward reuse the logits tensor for the backward pass - currently experimental (bool).

  - `use_gelu_act`: Whether to use the GELU activation.

  - `use_full_bias_size`: Whether to use a bias size of (batch_size, num_heads, seq_len, seq_len) instead of (1, 1, seq_len, seq_len) (bool).

  - `use_masking`: Whether to use masking - needs to be used together with full_bias_size (bool).

  - `attention_scale`: Attention scale (float).
