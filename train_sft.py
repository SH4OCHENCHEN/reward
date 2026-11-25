
import os
import torch
import argparse
from utils import *
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, unpair_preference_dataset
import wandb
import sys


parser = argparse.ArgumentParser(description='sft train')
# model and dataset name
parser.add_argument('--model', type=str, default="qwen1.7b", help='model name')
parser.add_argument('--dataset', type=str, default="hh", help='dataset name, hh, tldr, orca')
# training setting
parser.add_argument('--batch_size', type=int, default=128, help='total batch size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='gradient accumulation steps')
parser.add_argument('--epochs', type=int, default=1, help='total epochs')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=10, help='max grad norm')
parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='lr scheduler type')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')

parser.add_argument('--debug', default=False, action="store_true")
parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16'], help='训练精度')
add_lora_args(parser)
args = parser.parse_args()

gpu_nums = torch.cuda.device_count()
per_device_train_batch_size = int(args.batch_size / gpu_nums / args.gradient_accumulation_steps)

# load model
model_name, tokenizer_name = return_model_name(args.model)
dtype, precision_kwargs = resolve_precision(args)

sft_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=dtype)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# dataset process
preference_train_dataset, preference_test_dataset = return_preference_dataset(args.dataset)
if args.debug:
    preference_train_dataset = preference_train_dataset.select(range(1000))

def sft_hh(example): 
    return {"messages": example["prompt"] + example["chosen"]}
def sft_tldr(example):
    return {"text": example["prompt"] + example["chosen"]}
def sft_orca(example):
    return {"text": example["prompt"] + example["chosen"]}

if args.dataset == 'hh':
    sft_train_dataset = preference_train_dataset.map(sft_hh).remove_columns(preference_train_dataset.column_names)
    sft_test_dataset = preference_test_dataset.map(sft_hh).remove_columns(preference_test_dataset.column_names)
    sft_train_dataset = sft_train_dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False, enable_thinking=False)})
elif args.dataset == 'tldr':
    sft_train_dataset = preference_train_dataset.map(sft_tldr).remove_columns(["prompt", "chosen", "rejected"])
    sft_test_dataset = preference_test_dataset.map(sft_tldr).remove_columns(["prompt", "chosen", "rejected"])
elif args.dataset == 'capybara':
    sft_train_dataset = preference_train_dataset
    sft_test_dataset = preference_test_dataset


if args.dataset == 'tldr':
    args.max_length = 512
elif args.dataset == 'hh':
    args.max_length = 512
elif args.dataset == 'capybara':
    args.max_length = 1024

# wandb
# entity / project 优先读环境变量 WANDB_ENTITY / WANDB_PROJECT；
# 不设 WANDB_ENTITY 时用 wandb 登录账号的默认 entity。
if int(os.environ.get("RANK", "0")) == 0:
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY", None),
        project=os.environ.get("WANDB_PROJECT", "unbiased-alignment"),
    )

# config
peft_config = build_lora_config(args)

sft_args = SFTConfig(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optim="adamw_torch_fused",
    num_train_epochs=args.epochs,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    max_length=args.max_length,
    save_strategy="no",
    report_to="wandb",
    max_grad_norm=args.max_grad_norm,
    learning_rate = args.lr,
    seed=args.seed,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    gradient_checkpointing=True,
    **precision_kwargs,
)


sft_trainer = SFTTrainer(
    model=sft_model,
    processing_class=tokenizer,
    train_dataset=sft_train_dataset,
    eval_dataset=sft_test_dataset,
    args=sft_args,
    peft_config=peft_config,
)

if int(os.environ.get("RANK", "0")) == 0:
    print('--- sft data example ---\n' + tokenizer.decode(sft_trainer.train_dataset[0]['input_ids']))
    if peft_config is not None:
        print(f'--- LoRA enabled: r={peft_config.r} alpha={peft_config.lora_alpha} '
              f'targets={peft_config.target_modules} ---')

# train and save
sft_trainer.train()

sft_model_path = f"./models/sft_{args.model}_{args.dataset}"
if peft_config is not None:
    # 合并 adapter 后再存，否则下游的 train_dpo.py / train_reward.py 加载不到完整权重
    save_merged_model(sft_trainer, sft_model_path, tokenizer)
else:
    sft_trainer.save_model(sft_model_path)

if int(os.environ.get("RANK", "0")) == 0:
    print("sft train end")
