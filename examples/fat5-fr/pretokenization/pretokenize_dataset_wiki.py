import sys
import datasets

from transformers import AutoTokenizer

max_length = 1e9
name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(name)

wiki = datasets.load_dataset("wikimedia/wikipedia", "20231101.fr", split="train")

def tokenize_with_length(x):
    tokens_dict = tokenizer(x['text'], padding='do_not_pad', truncation=False, max_length=max_length, return_special_tokens_mask=True, return_tensors='pt')
    tokens_dict["length"] = tokens_dict["attention_mask"].sum()
    return tokens_dict

tokenized_wiki = wiki.map(lambda x: tokenize_with_length(x), remove_columns=['id', 'url', 'title', 'text'], num_proc=64)

tokenized_wiki.save_to_disk("output/wikipedia_2023_" + name + "/train")
