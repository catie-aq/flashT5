import datasets
import sys
from transformers import AutoTokenizer

max_length = 1e16
name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(name)

#wiki = datasets.load_dataset("graelo/wikipedia", "20230601.fr", beam_runner='DirectRunner', use_auth_token=True, split="train", cache_dir="/mnt/storage2/data_boris/")
news_train = datasets.load_dataset("eckendoerffer/news_fr", split="train")
news_valid = datasets.load_dataset("eckendoerffer/news_fr", split="validation")
news_test = datasets.load_dataset("eckendoerffer/news_fr", split="test")

def tokenize_with_length(x):
    tokens_dict = tokenizer(x['text'], padding='do_not_pad', truncation=False, max_length=max_length, return_special_tokens_mask=True, return_tensors='pt')
    tokens_dict["length"] = tokens_dict["attention_mask"].sum()
    return tokens_dict

tokenized_train = news_train.map(lambda x: tokenize_with_length(x), remove_columns=['text'], num_proc=64)
tokenized_valid = news_valid.map(lambda x: tokenize_with_length(x), remove_columns=['text'], num_proc=64)
tokenized_test = news_test.map(lambda x: tokenize_with_length(x), remove_columns=['text'], num_proc=64)

tokenized_train.save_to_disk("output/news_tokenized_nopad_" + name + "/train")
tokenized_valid.save_to_disk("output/news_tokenized_nopad_" + name + "/valid")
tokenized_test.save_to_disk("output/news_tokenized_nopad_" + name + "/test")
