import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

from flash_attn.layers.rotary import apply_rotary_emb_qkv_, apply_rotary_emb_func, apply_rotary_emb_kv_

class RelativePositionalEncoding(nn.Module):

    def __init__(self, relative_attention_num_buckets, relative_attention_max_distance, n_heads, max_sequence_length, bidirectional=True, randomized_position=False):

        super().__init__()

        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.n_heads = n_heads
        self.max_sequence_length = max_sequence_length
        self.bidirectional = bidirectional
        self.randomized_position = randomized_position

        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device

        if self.randomized_position:
            context_position = torch.arange(self.max_sequence_length, dtype=torch.long, device=device)
            context_indices_rand, _ = torch.sort(torch.randperm(self.max_sequence_length)[:query_length])
            context_indices_rand[0] = 0 # root the first element of the sequence
            context_position = context_position[context_indices_rand][:, None]

            memory_position = torch.arange(self.max_sequence_length, dtype=torch.long, device=device)
            memory_indices_rand, _ = torch.sort(torch.randperm(self.max_sequence_length)[:key_length])
            memory_indices_rand[0] = 0 # root the first element of the sequence
            memory_position = memory_position[memory_indices_rand][None, :]
        else:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]

        relative_position = memory_position - context_position  # shape (query_length, key_length)

        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(self, q, k=None, v=None):

        query_length = q.shape[1]
        key_length = k.shape[1] if k is not None else query_length
        bias = self.compute_bias(query_length, key_length, device=q.device).to(q.dtype)

        return q, k, v, bias


class ALiBiPositionalEncoding(nn.Module):

    def __init__(self, max_sequence_length, num_heads, mode='symetric', randomized_position=False):

        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.num_heads = num_heads
        self.mode = mode
        self.randomized_position = randomized_position

        self.alibi_bias = self.build_alibi_bias_matrix(num_heads, max_sequence_length, mode)

    @staticmethod
    def fill_with_neg_inf(t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float("-inf")).type_as(t)

    def get_slopes(self, n):

        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
        else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround.
            return get_slopes_power_of_2(closest_power_of_2) + self.get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    def build_symetric_alibi_bias_matrix(self, num_heads, maxpos):

        context_position = torch.arange(maxpos)[:, None]
        memory_position = torch.arange(maxpos)[None, :]

        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(num_heads, -1,-1)

        slopes = torch.Tensor(self.get_slopes(num_heads)) * -1
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        return alibi.view(1, num_heads, maxpos, maxpos)

    def build_asymetric_alibi_bias_matrix(self, num_heads, maxpos):
        _future_mask_right = torch.triu(self.fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1).unsqueeze(0).repeat(num_heads // 2, 1, 1)
        _future_mask_left = torch.tril(self.fill_with_neg_inf(torch.zeros([maxpos, maxpos])), -1).unsqueeze(0).repeat(num_heads // 2, 1, 1)

        nonsym_mask = torch.cat((_future_mask_right, _future_mask_left), dim = 0).unsqueeze(0)
        slopes = torch.Tensor(self.get_slopes(num_heads // 2)) * -1

        context_position = torch.arange(maxpos)[:, None]
        memory_position = torch.arange(maxpos)[None, :]

        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(num_heads // 2, -1,-1)

        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        alibi = alibi.view(1, num_heads // 2, maxpos, maxpos)
        alibi = alibi.repeat(1, 2, 1, 1)

        return alibi.view(1, num_heads, maxpos, maxpos) + nonsym_mask.view(1, num_heads, maxpos, maxpos)


    def build_alibi_bias_matrix(self, num_heads, maxpos, mode='symetric'):
        if mode == 'symetric':
            return self.build_symetric_alibi_bias_matrix(num_heads, maxpos)
        elif mode == 'asymetric':
            return self.build_asymetric_alibi_bias_matrix(num_heads, maxpos)
        else:
            raise ValueError("ALiBi mode " + mode + " is not implemented.")

    def forward(self, q, k=None, v=None):

        query_length = q.shape[1]
        key_length = k.shape[1] if k is not None else query_length
        assert (self.alibi_bias.shape[1] < query_length) & (self.alibi_bias.shape[1] < key_length), "Sequence length larger than allowed alibi bound"

        if self.randomized_position:
            query_indices_rand, _ = torch.sort(torch.randperm(self.max_sequence_length)[:query_length])
            key_indices_rand, _ = torch.sort(torch.randperm(self.max_sequence_length)[:key_length])

            # ground sequences
            query_indices_rand[0] = 0
            key_indices_rand[0] = 0

            bias = self.alibi_bias[:, :, query_indices_rand, key_indices_rand].to(q.device)

        else:
            bias = self.alibi_bias[:, :, :query_length, :key_length].to(q.device)

        return q, k, v, bias

class RotaryPositionalEncoding(nn.Module):

    def __init__(self, dim,
        max_sequence_length,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        randomized_position=False):

        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.randomized_position = randomized_position

        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        self.scale_base = scale_base

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        scale = (
            (torch.arange(0, dim, 2, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            inv_freq = self._compute_inv_freq(device=device)

            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            t = torch.arange(seqlen, device=device, dtype=dtype)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, q, k=None, v=None):

        if self._cos_cached is None:
            self._update_cos_sin_cache(self.max_sequence_length, device=q.device, dtype=q.dtype)

        if k is None and v is None:
            if self.scale is None:
                q = apply_rotary_emb_qkv_(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=0
                )
            else:
                q = apply_rotary_emb_qkv_(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=0
                )
        elif v is None and k is not None:
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=0
            )

            if self.scale is None:
                k = apply_rotary_emb_kv_(
                    k,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=0,
                )
            else:
                k = apply_rotary_emb_kv_(
                    k,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=0,
                )
        else:
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=0
            )
            if self.scale is None:
                k = apply_rotary_emb_func(
                    k,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=0,
                )
                v = apply_rotary_emb_func(
                    v,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=0,
                )
            else:
                k = apply_rotary_emb_func(
                    k,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=0,
                )
                v = apply_rotary_emb_func(
                    v,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=0,
                )

        return q, k, v, None
