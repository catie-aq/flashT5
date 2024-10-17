# Finetuning

> [!warning]
>  We are currently benchmarking our pre-trained models in French (see next section) to analyze the quality of our models and also whether our head implementations are correct. **So this work is still WIP**.

For the [classic T5](https://huggingface.co/docs/transformers/model_doc/t5), four different heads are available on Hugging Face: [`T5ForConditionalGeneration`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForConditionalGeneration), [`T5ForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForSequenceClassification) [`T5ForTokenClassification`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForTokenClassification) and [`T5ForQuestionAnswering`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForQuestionAnswering).
You can find the adaptation of the first head in this [file](https://github.com/catie-aq/flashT5/blob/684d02640464ea8bd2339689ce37da2d4e3b5f0b/src/model/modeling_flash_t5.py#L593) and that of the last three in this [file](https://github.com/catie-aq/flashT5/blob/main/src/model/custom_heads_flash_t5.py).

What we can say/observe at this stage is:
- We tested the `FlashT5ForConditionalGeneration` head on a text summarization task, in particular on the dataset [orange_sum](https://huggingface.co/datasets/orange_sum). The outputs of this dataset are 32 tokens. That's why for this [line](https://github.com/catie-aq/flashT5/blob/684d02640464ea8bd2339689ce37da2d4e3b5f0b/src/model/modeling_flash_t5.py#L640) we set `max_length = 32`. You'll need to set this value manually if you want to generate a different length.
For this head we've based ourselves on the [nanoT5 implementation](https://github.com/PiotrNawrot/nanoT5/blob/1c82d67bf8dea635be68a3b2a68a43b68b665193/nanoT5/utils/t5_model.py#L407) and not the Hugging Face one, as the latter is much faster (1 epoch of `FlashT5ForConditionalGeneration` takes us 6 min on FAT5-base vs. 3h30 on MT5-small).
The hyperparameters recommended in the [T5 documentation](https://huggingface.co/docs/transformers/model_doc/t5) (i.e. lr = `1e-4` or `3e-4`) don't seem to be the most suitable for this task for the FAT5 (= we match the results of Barthez, who introduced the `orange_sum` dataset, in 3 epochs against 30 but then reach a plateau). We need to carry out a search for hyperparameters.
For all the other tasks described below, a lr of `1e-4` gives the best results in the experiments we have carried out.
- For the `FlashT5ForTokenClassification`, we based ourselves on the implementation available on Hugging Face. This uses only the encoder. Thus, the number of parameters finetuned for this task are halved, and we obtain models with 67.1M parameters for the small version, 138M for the base version and 436M for the large version. This is something to bear in mind when benchmarking.
- For the `ForSequenceClassification`, the implementation available in Hugging Face is based on the encoder and decoder. This seems to us to be sub-optimal, so we've developed an encoder-only head.
Thus, the number of parameters finetuned for this task are halved, and we obtain models with 67.4M parameters for the small version, 138M for the base version and 436M for the large version. This is something to bear in mind when benchmarking.
- For the `T5ForQuestionAnswering`, the implementation available in Hugging Face is based on the encoder and decoder. This seems to us to be sub-optimal, so we've developed an encoder-only head.


> [!warning]
> Once your model has been finetuned, if you want to upload the weights to the Hugging Face Hub using the `push_to_hub` function, the latter won't load all the files you need to be able to reuse the model later. You'll have to perform a second upload yourself, where you'll load the missing files (these files are listed in the PR below). This is due to a bug in the `transformers` library. It has been reported and you can follow its progress in this [PR]( https://github.com/huggingface/transformers/issues/29714).
