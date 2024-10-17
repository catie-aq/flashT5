import datasets
import sys
from transformers import AutoTokenizer

max_length = 1e16
name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(name)

culturax_train = datasets.load_dataset("uonlp/CulturaX", "fr", split="train[:-50000]")
culturax_valid = datasets.load_dataset("uonlp/CulturaX", "fr", split="train[-50000:]")

def tokenize_with_length(x):
    tokens_dict = tokenizer(x['text'], padding='do_not_pad', truncation=False, max_length=max_length, return_special_tokens_mask=True, return_tensors='pt')
    tokens_dict["length"] = tokens_dict["attention_mask"].sum()
    return tokens_dict

tokenized_valid = culturax_valid.map(lambda x: tokenize_with_length(x), remove_columns=culturax_train.column_names, batched=False, num_proc=128)
tokenized_valid.save_to_disk("output/culturax_tokenized_" + name + "/valid")

tokenized_train = culturax_train.map(lambda x: tokenize_with_length(x), remove_columns=culturax_train.column_names, batched=False, num_proc=128)
tokenized_train.save_to_disk("output/culturax_tokenized_" + name + "/train")
