import datasets
import sys
from transformers import AutoTokenizer

max_length = 1e16
name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(name)

df_code = datasets.load_from_disk("bigcode/the-stack-dedup").select_columns(["raw_content"]).take(25000000)

def tokenize_with_length(x):
    tokens_dict = tokenizer(x['raw_content'], padding='do_not_pad', truncation=False, max_length=max_length, return_special_tokens_mask=True, return_tensors='pt')
    tokens_dict["length"] = tokens_dict["attention_mask"].sum()
    return tokens_dict

tokenized_train = df_code.map(lambda x: tokenize_with_length(x), remove_columns=['raw_content'], batched=False, num_proc=64)
tokenized_train.save_to_disk("output/thestack_tokenized_" + name + "/train")
