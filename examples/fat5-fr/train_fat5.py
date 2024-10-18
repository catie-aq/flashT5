import os
import torch
import yaml
import sys
import glob

# Configure clearml
os.environ["CLEARML_PROJECT"]="Flash-T5-FR"
os.environ["CLEARML_LOG_MODEL"]="False"
os.environ["WANDB_PROJECT"] = "Flash-T5-FR"

import datasets

from transformers import Trainer, TrainingArguments, AutoTokenizer, HfArgumentParser
from transformers.integrations import ClearMLCallback

from src.model.configuration_flash_t5 import FlashT5Config
from src.data.data_collator_ul2 import DataCollatorForUL2MLM
from src.model.modeling_flash_t5 import FlashT5ForConditionalGeneration
from optimization import create_optimizer, create_wsd_scheduler, create_cosine_scheduler

import evaluate
#import clearml

torch.set_float32_matmul_precision("high")

# Initialize
with open(sys.argv[1], 'r') as config_file:
    config = yaml.safe_load(config_file)

tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])

df_train_culturax = datasets.load_from_disk("/mnt/data2/culturax_tokenized_tokenizer-flasht5-french/train", keep_in_memory=False).with_format("np")
df_train_news = datasets.load_from_disk("/mnt/data2/news_tokenized_nopad_tokenizer-flasht5-french/train", keep_in_memory=False).with_format("np")
df_train_justice = datasets.load_from_disk("/mnt/data2/justice_tokenized_tokenizer-flasht5-french/train", keep_in_memory=False).with_format("np")
df_train_stack = datasets.load_from_disk("/mnt/data2/thestack_tokenized_tokenizer-flasht5-french/train", keep_in_memory=False).with_format("np")
df_train_wiki = datasets.load_from_disk("/mnt/data2/wikipedia_2023_tokenizer-flasht5-french/train", keep_in_memory=False).with_format("np")

train_dataset = datasets.concatenate_datasets([df_train_culturax, df_train_news, df_train_justice, df_train_stack, df_train_wiki])

valid_news = datasets.load_from_disk("/mnt/data2/news_tokenized_nopad_tokenizer-flasht5-french/valid", keep_in_memory=False).with_format("np")
#valid_culturax = datasets.load_from_disk("/mnt/data2/culturax_tokenized_tokenizer-flasht5-french/valid", keep_in_memory=False).with_format("np")

valid_dataset = valid_news

train_dataset = train_dataset.remove_columns(["special_tokens_mask"])
valid_dataset = valid_dataset.remove_columns(["special_tokens_mask"])

config_collator = config["collator_args"]
model_config = config["model_args"]
config_training_arguments = config["training_args"]

data_collator = DataCollatorForUL2MLM(
    tokenizer=tokenizer,
    max_length=config_collator["max_token_length"],
    max_labels_length=config_collator["max_labels_length"],
    batch_size=config_collator["output_batch_size"],
    #denoiser_list=[{"mu": 3.0, "r": 0.15, "max_spans": 512, "prefix": ""}], # this corresponds to T5 noise parameters
    #denoiser_proportions=[1.0]
    denoiser_list=[{"mu": 3.0, "r": 0.15, "max_spans": config_collator["max_token_length"], "prefix": "[R]"},
                   {"mu": 8.0, "r": 0.15, "max_spans": config_collator["max_token_length"], "prefix": "[R]"},
                   {"mu": 4.0, "r": 0.0, "max_spans": 1, "prefix": "[S]"},
                   {"mu": 3.0, "r": 0.5, "max_spans": config_collator["max_token_length"], "prefix": "[X]"},
                   {"mu": 8.0, "r": 0.15, "max_spans": config_collator["max_token_length"], "prefix": "[X]"},
                   {"mu": 64.0, "r": 0.15, "max_spans": config_collator["max_token_length"], "prefix": "[X]"},
                   {"mu": 64.0, "r": 0.5, "max_spans": config_collator["max_token_length"], "prefix": "[X]"}],
    denoiser_proportions=[0.165, 0.165, 0.34, 0.0825, 0.0825, 0.0825, 0.0825],
    fixed_batch_size=True,
    min_size_inputs=5
)

# Set a configuration for our T5 model
model_config["vocab_size"] = tokenizer.vocab_size
model_config["pad_token_id"] = tokenizer.pad_token_id
config_hf_model = FlashT5Config.from_dict(config["model_args"])

# Initialize the model from a configuration without pretrained weights
model = FlashT5ForConditionalGeneration(config=config_hf_model)
print('Num parameters: ', model.num_parameters())

masked_accuracy = evaluate.load("accuracy")

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    labels = labels.flatten()
    predictions=logits.flatten()[labels > 0]
    labels = labels[labels > 0]

    return {'MaskedAccuracy': masked_accuracy.compute(predictions=predictions, references=labels)["accuracy"]}

parser = HfArgumentParser(TrainingArguments)
config_training_arguments["gradient_accumulation_steps"] = max(1, config_training_arguments["gradient_accumulation_steps"] // torch.cuda.device_count())
config_training_arguments["report_to"] = ["codecarbon"]
config_training_arguments["output_dir"] = config["model_name"] + "_v" + str(config["version"])
config_training_arguments["run_name"] = config["model_name"] + "_fr_" + \
    str(model_config["position_encoding_type"])
config_training_arguments["report_to"] = "wandb"
os.environ["CLEARML_TASK"]=config_training_arguments["run_name"]

training_args = parser.parse_dict(config_training_arguments)[0]

optimizer = create_optimizer(model,
                            lr=training_args.learning_rate,
                            betas=(training_args.adam_beta1, training_args.adam_beta2),
                            eps=training_args.adam_epsilon,
                            weight_decay=training_args.weight_decay,
                            foreach=True)
scheduler = create_cosine_scheduler(training_args.warmup_steps, training_args.warmup_ratio, training_args.max_steps, optimizer)

#clearml_callback = ClearMLCallback()
#clearml_callback._clearml_task = clearml.Task.init("Flash-T5", config_training_arguments["run_name"], continue_last_task=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
    #callbacks=[clearml_callback]
    )

is_already_trained = (len(set([os.path.dirname(p) for p in glob.glob(config_training_arguments["output_dir"] + "/*/*")])) != 0)
result = trainer.train(resume_from_checkpoint=(config["checkpoint_name"] and is_already_trained))
#trainer.evaluate()
