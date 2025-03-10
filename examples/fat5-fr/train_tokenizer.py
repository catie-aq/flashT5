import datasets

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split, Sequence

from transformers import T5TokenizerFast

VOCAB_SIZE = 32768

if VOCAB_SIZE % 64 != 0:
    print("Performance warning : the vocab size should be a multiple of 64!")

df_fr = datasets.load_dataset("uonlp/CulturaX", "fr").rename_column("text", "raw_content").select_columns(['raw_content'])
df_code = datasets.load_dataset("bigcode/the-stack-dedup", split="train", streaming=True).rename_column("content", "raw_content").select_columns(['raw_content'])

df = datasets.concatenate_datasets([df_fr, df_code])

def batch_iterator(dataset, batch_size=1000):
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["raw_content"]

special_tokens_dict = ["<cls>", "<s>", "</s>", "<mask>", "<pad>", "<sep>", "<unk>"]

for i in range(256):
    special_tokens_dict.append("<extra_id_" + str(i) + ">")

# inspired by punct (arXiv:2402.01035v2) but with individual digits
pat_str = r" ?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens_dict, max_token_length=20, show_progress=True)
pre_tokenizer = Sequence([Split(pattern=Regex(pat_str), behavior="isolated")])
tokenizer.pre_tokenizer = pre_tokenizer

tokenizer.train_from_iterator(batch_iterator(df), trainer, length=len(df))

pretrained_tokenizer = T5TokenizerFast(tokenizer_object=tokenizer, clean_up_tokenization_spaces=False)
pretrained_tokenizer.save_pretrained("tokenizer-flasht5-french")
