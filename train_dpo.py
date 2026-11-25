import os
import torch
import argparse
from utils import *
from datasets import load_dataset, Dataset
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# from trl import SFTTrainer, DPOTrainer, DPOConfig
from methods import DPOTrainer, DPOConfig
import pprint
import random
import wandb


parser = argparse.ArgumentParser(description='dpo-base train')
# model and dataset name
parser.add_argument('--model', type=str, default="qwen1.7b", help='model name')
parser.add_argument('--dataset', type=str, default="hh", help='dataset name, hh, tldr, ufb')
parser.add_argument('--noise_rate', type=float, default=0, help='noise rate')
# loss setting
parser.add_argument('--loss', type=str, default='sigmoid', help='the loss functions, sigmoid, robust, unbiased...')
parser.add_argument('--para', type=float, default=0, help='loss parameter')
# train setting
parser.add_argument('--batch_size', type=int, default=128, help='total batch size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='gradient accumulation steps')
parser.add_argument('--precompute_ref_batch_size_multiple', type=int, default=4, help='precompute_ref_batch_size_multiple')
parser.add_argument('--epochs', type=int, default=3, help='epochs')
parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=10, help='max grad norm')
parser.add_argument('--seed', type=int, default=123, help='seed')
parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='lr scheduler type')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
parser.add_argument('--debug', default=False, action="store_true")
parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16'], help='训练精度')
add_lora_args(parser)
args = parser.parse_args()


# uniform format
if args.noise_rate in [0, 0.0]:
    args.noise_rate = 0
if args.para in [0, 0.0]:
    args.para = 0

validate_unbiased_para(args.loss, args.para)

gpu_nums = torch.cuda.device_count()
# print(gpu_nums)
per_device_train_batch_size = int(args.batch_size / gpu_nums / args.gradient_accumulation_steps)

dataset_name = return_dataset_name(args.dataset)
model_name = return_model_name(args.model)


sft_model_path = f"./models/sft_{args.model}_{args.dataset}"
if not os.path.exists(sft_model_path):
    raise ValueError('No exist sft model, train sft model first')

dtype, precision_kwargs = resolve_precision(args)
peft_config = build_lora_config(args)

dpo_model = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True, dtype=dtype)
if peft_config is None:
    ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True, dtype=dtype)
else:
    # LoRA 模式下 reference 用同一份模型关掉 adapter 来算，不需要再加载第二份权重。
    # 同时必须传 ref_model=None，否则 DPOTrainer 会直接抛 ValueError。
    ref_model = None
tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# dataset process
preference_train_dataset, preference_test_dataset = return_preference_dataset(args.dataset)
if args.debug:
    preference_train_dataset = preference_train_dataset.select(range(1000))

# datasets.disable_caching()
random.seed(args.seed)

if args.noise_rate > 0:
    preference_train_dataset = preference_train_dataset.map(lambda x: add_preference_noise(x, noise_rate=args.noise_rate))

if args.dataset == 'tldr':
    args.max_length = 512
    args.beta = 0.5
elif args.dataset == 'hh':
    args.max_length = 512
    args.beta = 0.1
elif args.dataset == 'ufb':
    args.max_length = 1024
    args.beta = 0.1


# wandb
# entity / project 优先读环境变量 WANDB_ENTITY / WANDB_PROJECT；
# 不设 WANDB_ENTITY 时用 wandb 登录账号的默认 entity。
if int(os.environ.get("RANK", "0")) == 0:
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY", None),
        project=os.environ.get("WANDB_PROJECT", "unbiased-alignment"),
    )

# config and train
dpo_args = DPOConfig(
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optim="adamw_torch_fused",
    num_train_epochs=args.epochs,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=20,
    eval_on_start=True,
    max_length=args.max_length,
    save_strategy="no",
    report_to="wandb",
    max_grad_norm=args.max_grad_norm,
    loss_type=args.loss,
    learning_rate=args.lr,
    seed=args.seed,
    beta=args.beta,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    precompute_ref_log_probs=True,
    precompute_ref_batch_size=per_device_train_batch_size*args.precompute_ref_batch_size_multiple,
    gradient_checkpointing=True,
    **precision_kwargs,
)
if args.loss == 'sigmoid':
    dpo_args.label_smoothing = args.para
elif args.loss == 'robust':
    dpo_args.label_smoothing = args.para
elif args.loss == 'drdpo':
    dpo_args.drdpo_beta = args.para
elif args.loss == 'unbiased':
    dpo_args.unbiased_a = args.para
elif args.loss == 'normal_unbiased':
    dpo_args.unbiased_a = args.para

dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=ref_model,
    processing_class=tokenizer,
    train_dataset=preference_train_dataset,
    eval_dataset=preference_test_dataset,
    args=dpo_args,
    peft_config=peft_config,
)
if int(os.environ.get("RANK", "0")) == 0:
    print('--- prompt example ---\n' + tokenizer.decode(dpo_trainer.train_dataset[0]['prompt_input_ids']))
    print('--- chosen response ---\n' + tokenizer.decode(dpo_trainer.train_dataset[0]['chosen_input_ids']))
    print('--- rejected response ---\n' + tokenizer.decode(dpo_trainer.train_dataset[0]['rejected_input_ids']))
    if peft_config is not None:
        print(f'--- LoRA enabled: r={peft_config.r} alpha={peft_config.lora_alpha} '
              f'targets={peft_config.target_modules} ---')
        print('--- reference model = 同一份模型关掉 adapter（不额外占显存）---')


dpo_model_path = f"./models/dpo_{args.model}_{args.dataset}_{args.loss}_para={args.para}_noise={args.noise_rate}"
dpo_trainer.train()
if peft_config is not None:
    # 合并 adapter 后再存，保证产物是一个完整可加载的模型
    save_merged_model(dpo_trainer, dpo_model_path, tokenizer)
else:
    dpo_trainer.save_model(dpo_model_path)
if int(os.environ.get("RANK", "0")) == 0:
    print("train end")