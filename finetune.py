import sys
import logging

import os
import datasets
from random import randrange
from datasets import load_dataset
from peft import LoraConfig
import torch
import transformers
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTConfig

os.environ["WANDB_DISABLED"] = "true"
#os.environ["ACCELERATE_MIXED_PRECISION"] = "yes"

logger = logging.getLogger(__name__)

args = SFTConfig(
        output_dir="./phi-3-mini-LoRA",
        evaluation_strategy="steps",
        do_eval=True,
        optim="adamw_torch",
        auto_find_batch_size=True,
        #per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        #per_device_eval_batch_size=8,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=1e-4,
        fp16 = True,
        eval_steps=100,
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        seed=42,
)

training_config = {
    "bf16": False,
    "fp16": True,
    "do_eval": False,
    "learning_rate": 5.0e-06,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "auto_find_batch_size":True,
    #"per_device_eval_batch_size": 4,
    #"per_device_train_batch_size": 4,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 0,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs":{"use_reentrant": False},
    "gradient_accumulation_steps": 1,
    "warmup_ratio": 0.2,
}

peft_config = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
    "modules_to_save": None,
}

#train_conf = TrainingArguments(**training_config)
peft_conf = LoraConfig(**peft_config)

## Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Log on each process a small summary
logger.warning(
    f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
    + f" distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
)
logger.info(f"Training/evaluation parameters {args}")
logger.info(f"PEFT parameters {peft_conf}")


compute_dtype = torch.float16
attn_implementation = 'eager'


################
# Modle Loading
################
checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
# checkpoint_path = "microsoft/Phi-3-mini-128k-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation=attn_implementation,  # loading the model with flash-attenstion support
    torch_dtype=compute_dtype,
    device_map=None
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

def apply_chat_template(
    example,
    tokenizer,
):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    return example

dataset_name = 'iamtarun/python_code_instructions_18k_alpaca'
dataset_split= "train"
dataset = load_dataset(dataset_name, split=dataset_split)

# The 'len' function is used to get the size of the dataset, which is then printed.
print(f"dataset size: {len(dataset)}")

# 'randrange' is a function from the 'random' module that generates a random number within the specified range.
# Here it's used to select a random example from the dataset, which is then printed.
print(dataset[randrange(len(dataset))])

# This code block defines two functions that are used to format the dataset for training a chat model.

# 'create_message_column' is a function that takes a row from the dataset and returns a dictionary 
# with a 'messages' key and a list of 'user' and 'assistant' messages as its value.
def create_message_column(row):
    # Initialize an empty list to store the messages.
    messages = []
    
    # Create a 'user' message dictionary with 'content' and 'role' keys.
    user = {
        "content": f"{row['instruction']}\n Input: {row['input']}",
        "role": "user"
    }
    
    # Append the 'user' message to the 'messages' list.
    messages.append(user)
    
    # Create an 'assistant' message dictionary with 'content' and 'role' keys.
    assistant = {
        "content": f"{row['output']}",
        "role": "assistant"
    }
    
    # Append the 'assistant' message to the 'messages' list.
    messages.append(assistant)
    
    # Return a dictionary with a 'messages' key and the 'messages' list as its value.
    return {"messages": messages}

# 'format_dataset_chatml' is a function that takes a row from the dataset and returns a dictionary 
# with a 'text' key and a string of formatted chat messages as its value.
def format_dataset_chatml(row):
    # 'tokenizer.apply_chat_template' is a method that formats a list of chat messages into a single string.
    # 'add_generation_prompt' is set to False to not add a generation prompt at the end of the string.
    # 'tokenize' is set to False to return a string instead of a list of tokens.
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}

# This code block is used to prepare the 'dataset' for training a chat model.

# 'dataset.map' is a method that applies a function to each example in the 'dataset'.
# 'create_message_column' is a function that formats each example into a 'messages' format suitable for a chat model.
# The result is a new 'dataset_chatml' with the formatted examples.
dataset_chatml = dataset.map(create_message_column)

# 'dataset_chatml.map' is a method that applies a function to each example in the 'dataset_chatml'.
# 'format_dataset_chatml' is a function that further formats each example into a single string of chat messages.
# The result is an updated 'dataset_chatml' with the further formatted examples.
dataset_chatml = dataset_chatml.map(format_dataset_chatml)

# This code block is used to split the 'dataset_chatml' into training and testing sets.

# 'dataset_chatml.train_test_split' is a method that splits the 'dataset_chatml' into a training set and a testing set.
# 'test_size' is a parameter that specifies the proportion of the 'dataset_chatml' to include in the testing set. Here it's set to 0.05, meaning that 5% of the 'dataset_chatml' will be included in the testing set.
# 'seed' is a parameter that sets the seed for the random number generator. This is used to ensure that the split is reproducible. Here it's set to 1234.
dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

# This line of code is used to display the structure of the 'dataset_chatml' after the split.
# It will typically show information such as the number of rows in the training set and the testing set.
dataset_chatml


trainer = SFTTrainer(
    model=model,
    args=args,
    peft_config=peft_conf,
    train_dataset=dataset_chatml['train'],
    eval_dataset=dataset_chatml['test'],
    max_seq_length=512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True,
    dataset_num_proc=10    
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()


#############
# Evaluation
#############
tokenizer.padding_side = 'left'
metrics = trainer.evaluate()
metrics["eval_samples"] = len(processed_test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# ############
# # Save model
# ############
trainer.save_model(train_conf.output_dir)
