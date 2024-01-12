from typing import Dict, List
import numpy as np
from dataclasses import dataclass
from transformers import BatchEncoding
from transformers.data.data_collator import DataCollatorMixin
from transformers import AutoTokenizer
#import numba
import torch

class DataCollatorForUL2MLM(DataCollatorMixin):

    def __init__(self, tokenizer: AutoTokenizer,
                max_length: int,
                max_labels_length: int,
                batch_size: int,
                denoiser_list: List,
                denoiser_proportions: List,
                causal: bool = False):

        super().__init__()

        self.denoiser_proportions = denoiser_proportions
        if sum(self.denoiser_proportions) != 1.0:
            self.denoiser_proportions = [x / sum(self.denoiser_proportions) for x in self.denoiser_proportions]
        self.denoiser_list = denoiser_list #(mu, r, n, prefix)

        self.tokenizer = tokenizer
        self.prefixes = [tokenizer.encode(denoiser_list[i]["prefix"], return_tensors='np').flatten()[:-1] for i in range(len(denoiser_list))]

        # this assume the extra tokens ids are contiguous
        self.extra_ids = sorted([tokenizer.all_special_ids[i] for i, x in enumerate(tokenizer.all_special_tokens) if "extra" in x], reverse=True)
        self.max_length = max_length
        self.batch_size = batch_size
        self.max_labels_length = max_labels_length

        max_prefix_len = max([len(x) for x in self.prefixes])
        self.denoiser_optimal_len = [self.compute_input_and_target_lengths(max_length-max_prefix_len, x["r"], x["mu"]) for x in self.denoiser_list]

        self.causal = causal

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:

        input_batch_size = len(examples)

        # Sample a list of denoiser
        denoisers_sample = np.random.choice(range(len(self.denoiser_list)), input_batch_size, p=self.denoiser_proportions)

        # truncate lengths
        truncated_examples = []
        for i, x in enumerate(examples):
            max_len = self.denoiser_optimal_len[denoisers_sample[i]][0]
            if x["length"] > max_len:
                new_input_ids = x["input_ids"][:, :max_len]
                new_input_ids = x["input_ids"][:, :max_len]
                new_length = np.array(max_len)
                truncated_examples.append({"input_ids": new_input_ids, "length": new_length})
            else:
                truncated_examples.append(x)

        examples = truncated_examples

        spans_noise_masks = [self.random_spans_noise_mask(x["length"], self.denoiser_list[denoisers_sample[i]])
                for i, x in enumerate(examples)]

        input_ids_sentinel = [self.create_sentinel_ids(x.astype(np.int8)) for x in spans_noise_masks]
        labels_sentinel = [self.create_sentinel_ids((~x).astype(np.int8)) for x in spans_noise_masks]

        input_ids = [self.filter_input_ids(x["input_ids"], input_ids_sentinel[i], np.expand_dims(self.prefixes[denoisers_sample[i]], axis=0)) for i, x in enumerate(examples)]
        labels = [self.filter_input_ids(x["input_ids"], labels_sentinel[i], with_eos=False) for i, x in enumerate(examples)]

        def is_special_token(x):
            return (x <= self.extra_ids[0]) & (x >= self.extra_ids[-1])

        # Generate batch by greedy concatenate small length (to avoid large padding)
        batch_inputs = []
        batch_labels = []
        included_elements = [False for _ in range(len(input_ids))]

        for i in range(self.batch_size):
            concatenated_inputs = None
            concatenated_labels = None

            for i, x in enumerate(input_ids):
                if included_elements[i] == False:
                    item_labels = labels[i]
                    size_inputs = x.shape[1]
                    size_labels = item_labels.shape[1]

                    if concatenated_inputs is None:
                        concatenated_inputs = x
                        concatenated_labels = item_labels
                        included_elements[i] = True
                    elif ((concatenated_inputs.shape[1] + size_inputs < self.max_length)
                          and (concatenated_labels.shape[1] + size_labels < self.max_labels_length)):

                        # if we have too much labels used already, pass
                        num_labels = is_special_token(concatenated_inputs).sum()
                        num_new_labels = is_special_token(x).sum()
                        if (num_labels + num_new_labels) >= len(self.extra_ids):
                            continue

                        concatenated_inputs = np.concatenate([concatenated_inputs, x], axis=1)
                        concatenated_labels = np.concatenate([concatenated_labels, item_labels], axis=1)
                        included_elements[i] = True

            batch_inputs.append(concatenated_inputs)
            batch_labels.append(concatenated_labels)

            # reset the list in the unlikely event that the list has been completly filled
            if sum(included_elements) == len(included_elements):
                included_elements = [False for _ in range(len(input_ids))]

        labels = [np.where(is_special_token(x), self.extra_ids[0] - np.cumsum(is_special_token(x)) + 1, x) for x in batch_labels]
        input_ids = [np.where(is_special_token(x), self.extra_ids[0] - np.cumsum(is_special_token(x)) + 1, x) for x in batch_inputs]

        # add a final eos in labels to terminate generating
        labels = [np.concatenate([x, np.full((1,1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1) for x in labels]

        if self.causal:
            # left pad inputs_ids and right pad labels
            labels = np.concatenate([np.pad(x, ((0,0), (0, self.max_labels_length - x.shape[1])), constant_values=self.tokenizer.pad_token_id) for x in labels], axis=0)
            input_ids = np.concatenate([np.pad(x, ((0,0), (self.max_length - x.shape[1], 0)), constant_values=self.tokenizer.pad_token_id) for x in input_ids], axis=0)
        else:
            # right pad everything
            labels = np.concatenate([np.pad(x, ((0,0), (0, self.max_labels_length - x.shape[1])), constant_values=self.tokenizer.pad_token_id) for x in labels], axis=0)
            input_ids = np.concatenate([np.pad(x, ((0,0), (0, self.max_length - x.shape[1])), constant_values=self.tokenizer.pad_token_id) for x in input_ids], axis=0)

        batch = {}
        causal_labels = None
        if self.causal == False:
            batch["input_ids"] = torch.from_numpy(input_ids)
            causal_labels = torch.from_numpy(labels)
        else:
            batch["input_ids"] = torch.from_numpy(np.concatenate([input_ids, labels], axis=-1))
            causal_labels = batch["input_ids"].clone()

        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id)
        causal_labels[causal_labels == self.tokenizer.pad_token_id] = -100 # ignore padding indices for the loss
        batch["labels"] = causal_labels

        return batch

    def compute_input_and_target_lengths(self, inputs_length, noise_density, mean_noise_span_length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

        [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
        Training parameters to avoid padding with random_spans_noise_mask.
        When training a model with random_spans_noise_mask, we would like to set the other
        training hyperparmeters in a way that avoids padding.
        This function helps us compute these hyperparameters.
        We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
        and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
        This function tells us the required number of tokens in the raw example (for split_tokens())
        as well as the length of the encoded targets. Note that this function assumes
        the inputs and targets will have EOS appended and includes that in the reported length.

        Args:
            inputs_length: an integer - desired length of the tokenized inputs sequence
            noise_density: a float
            mean_noise_span_length: a float
        Returns:
            tokens_length: length of original text in tokens
            targets_length: an integer - length in tokens of encoded targets sequence
        """

        def _tokens_length_to_inputs_length_targets_length(tokens_length):
            num_noise_tokens = int(round(tokens_length * noise_density))
            num_nonnoise_tokens = tokens_length - num_noise_tokens
            num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
            # inputs contain all nonnoise tokens, sentinels for all noise spans
            # and one EOS token.
            _input_length = num_nonnoise_tokens + num_noise_spans + 1
            _output_length = num_noise_tokens + num_noise_spans + 1
            return _input_length, _output_length

        tokens_length = inputs_length

        # case of causal LM
        if noise_density == 0.0:
            return (self.max_labels_length + int(self.max_labels_length // mean_noise_span_length), inputs_length)

        while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
            tokens_length += 1

        inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

        # minor hack to get the targets length to be equal to inputs length
        # which is more likely to have been set to a nice round number.
        if noise_density == 0.5 and targets_length > inputs_length:
            tokens_length -= 1
            targets_length -= 1
        return tokens_length, targets_length

    def random_spans_noise_mask(self, sequence_length, denoiser_params):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        mean_noise_span_length = denoiser_params["mu"]
        noise_density = denoiser_params["r"]
        max_num_spans = denoiser_params["max_spans"]

        if max_num_spans == 1:
            # force the span to be at the beginning of the sequence
            prefix_span = int(np.round(sequence_length / mean_noise_span_length))
            masked_span = sequence_length - prefix_span
            interleaved_span_lengths = np.array([prefix_span, masked_span])
        else:
            num_noise_tokens = int(np.round(sequence_length * noise_density))
            # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
            num_noise_tokens = min(max(num_noise_tokens, 1), sequence_length - 1)
            num_noise_spans = min(max_num_spans, int(np.round(num_noise_tokens / mean_noise_span_length)))

            # avoid degeneracy by ensuring positive number of noise spans
            num_noise_spans = max(num_noise_spans, 1)
            num_nonnoise_tokens = sequence_length - num_noise_tokens

            # pick the lengths of the noise spans and the non-noise spans
            def _random_segmentation(num_items, num_segments):
                """Partition a sequence of items randomly into non-empty segments.
                Args:
                    num_items: an integer scalar > 0
                    num_segments: an integer scalar in [1, num_items]
                Returns:
                    a Tensor with shape [num_segments] containing positive integers that add
                    up to num_items
                """
                mask_indices = np.arange(num_items - 1) < (num_segments - 1)
                np.random.shuffle(mask_indices)
                first_in_segment = np.pad(mask_indices, [[1, 0]])
                segment_id = np.cumsum(first_in_segment)
                # count length of sub segments assuming that list is sorted
                _, segment_length = np.unique(segment_id, return_counts=True)
                return segment_length

            noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
            nonnoise_span_lengths = _random_segmentation(
                num_nonnoise_tokens, num_noise_spans
            )

            interleaved_span_lengths = np.reshape(
                np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
                [num_noise_spans * 2],
            )

        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((sequence_length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise


    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[0] = mask_indices[0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (self.extra_ids[0] - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids, prefixes=None, with_eos=True):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed

        input_ids = input_ids[input_ids != self.tokenizer.eos_token_id]
        input_ids = input_ids[input_ids >= 0].reshape((batch_size, -1))

        if prefixes is not None:
            input_ids = np.concatenate(
                [prefixes, input_ids], axis=-1
            )

        if with_eos:
            input_ids = np.concatenate(
                [input_ids,  np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
            )

        return input_ids
