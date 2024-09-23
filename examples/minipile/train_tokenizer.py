import datasets

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Digits, Whitespace

from transformers import T5TokenizerFast

# Load the dataset
df = datasets.load_dataset("JeanKaddour/minipile", cache_dir="/mnt/storage3/hf_cache")["train"].select_columns(['text'])

def get_training_corpus():
    print("Number of steps : " + str(len(df) // 1000))
    for start_idx in range(0, len(df) // 1000):
        samples = df[start_idx * 1000 : start_idx + 1000]
        yield samples["text"]

special_tokens_dict = ["<cls>", "</s>", "<mask>", "<pad>", "<sep>", "<unk>"]

# Add extra masking tokens for the FAT5 model
for i in range(256):
    special_tokens_dict.append("<extra_id_" + str(i) + ">")

training_corpus = get_training_corpus()
vocab_size = 32768
pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

# Train the tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens_dict, max_token_length=20)
pre_tokenizer = Sequence([Whitespace(), Digits(individual_digits=True)])
tokenizer.pre_tokenizer = pre_tokenizer

tokenizer.train_from_iterator(training_corpus, trainer)
pretrained_tokenizer = T5TokenizerFast(tokenizer_object=tokenizer, clean_up_tokenization_spaces=False)
pretrained_tokenizer.save_pretrained("tokenizer-fat5-minipile")
