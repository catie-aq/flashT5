# Finetuning

> [!warning]
> The information presented on this page should only be provisional, awaiting a port to a Hugging Face `transformers`, on which we are currently working with their team. 

## Some information about finetuning heads
For the [classic T5](https://huggingface.co/docs/transformers/model_doc/t5), five different heads are available on Hugging Face:
- [`T5ForConditionalGeneration`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForConditionalGeneration)
- [`T5EncoderModel`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel)
- [`T5ForSequenceClassification`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForSequenceClassification)
- [`T5ForQuestionAnswering`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForQuestionAnswering)
- [`T5ForTokenClassification`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForTokenClassification)
  
We've adapted these heads to our FAT5. You can find the adaptation two first ones in this [file](https://github.com/catie-aq/flashT5/blob/86cef9295dfdac016a2f36a1ea1e1ea9419a7f6f/src/model/modeling_flash_t5.py#L604) and that of the last three in this [file](https://github.com/catie-aq/flashT5/blob/86cef9295dfdac016a2f36a1ea1e1ea9419a7f6f/src/model/custom_heads_flash_t5.py#L19).  

To sum up:
- `FAT5T5EncoderModel` is unchanged from its Hugging Face implementation. 
- `FlashT5ForConditionalGeneration` is based on the [nanoT5 implementation](https://github.com/PiotrNawrot/nanoT5/blob/1c82d67bf8dea635be68a3b2a68a43b68b665193/nanoT5/utils/t5_model.py#L407) and not the Hugging Face one, as the latter is much faster (1 epoch of `FlashT5ForConditionalGeneration` takes us 15 min vs. 3h30 on MT5-small on the same dataset). 
- `FlashT5ForSequenceClassification` uses only the encoder part of the architecture, whereas the Hugging Face implementation uses the encoder-decoder. The result is a model that roughly be finetuned twice faster, with only 67.4M parameters instead of 147M and so taking less RAM space (270 MB vs. 587 MB).
- `FlashT5ForQuestionAnswering` uses only the encoder part of the architecture, whereas the Hugging Face implementation uses the encoder-decoder.
- `FlashT5ForTokenClassification` is unchanged from its Hugging Face implementation because it already uses only the encoder part of the architecture. In fact, we've used the same convention as this head for the SequenceClassification and QuestionAnswering heads, to ensure a certain coherence between all classification heads format.

<br>

## The finetuning itself
The finetuning of the model is performed in the classic way as can be done with a standard T5 (see for example theses [ressources](https://huggingface.co/docs/transformers/model_doc/t5#resources)).    
To perform the finetunings shown in the [blog post](https://huggingface.co/spaces/CATIE-AQ/FAT5-report), we use a learning rate of `1e-4` for all heads.

> [!warning]
> Once your model has been finetuned, if you want to upload the weights to the Hugging Face Hub using the `push_to_hub` function, the latter won't load all the files you need to be able to reuse the finetuned model. You'll have to perform a second upload yourself, where you'll load the missing files (these files are listed in the PR mentioned below or you can check the ["Files and versions"](https://huggingface.co/CATIE-AQ/FAT5-small/tree/main) on our pretrained FAT5 in French). This is due to a bug in the `transformers` library with custom model. It has been reported and you can follow its progress in this [PR](https://github.com/huggingface/transformers/issues/29714).
> With the integration into `transformers`, this problem should disappear, as FAT5 will no longer be a custom model.
