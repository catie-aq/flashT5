# Training FAT5 on MiniPile

## Introduction

This example demonstrates how to train a FAT5 model on the MiniPile dataset.

## Train a tokenizer and pretokenize the minipile dataset

The following script trains a tokenizer on the MiniPile dataset.

```bash
python train_tokenizer.py
```

following by the full tokenization of the minipile dataset:

```bash
python pretokenize_minipile.py
```

## Train a FAT5 model

The model can be trained with the following command:

```bash
python train_fat5_minipile.py config/flash-t5-small-minipile.yaml
```
