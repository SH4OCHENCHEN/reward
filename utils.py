import random
import numpy as np
from datasets import load_dataset, Dataset
from trl import extract_prompt
import logging
import datasets


def dpo_tldr(data_format1):

  data_format2 = {
      "prompt": [{"role": "user", "content": data_format1["prompt"]}],
      "chosen": [{"role": "assistant", "content": data_format1["chosen"]}],
      "rejected": [{"role": "assistant", "content": data_format1["rejected"]}],
  }
  return data_format2

def add_preference_noise(example, noise_rate=0):
    """
    以一定概率交换 chosen 和 rejected, 模拟人工标注噪声。
    """
    if random.random() < noise_rate:
        # 交换 chosen 和 rejected
        example['chosen'], example['rejected'] = example['rejected'], example['chosen']
    return example


def add_lora_args(parser):
    """给 argparse 挂上 LoRA 相关参数（三个 train 脚本共用）。"""
    group = parser.add_argument_group("LoRA")
    group.add_argument(
        '--use_lora', action="store_true",
        help='使用 LoRA 微调；不加这个开关则是全参微调'
    )
    group.add_argument('--lora_r', type=int, default=16, help='LoRA 秩 r')
    group.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha，一般取 2*r')
    group.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    group.add_argument(
        '--lora_target_modules', type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help='要加 LoRA 的模块名，逗号分隔'
    )


def build_lora_config(args, modules_to_save=None):
    """
    按命令行参数构造 LoraConfig；未开启 --use_lora 时返回 None。

    modules_to_save 用于让某个新建的模块（如奖励模型的 score head）参与训练，
    否则 LoRA 只会训练低秩矩阵，这个头会被冻结。
    """
    if not getattr(args, 'use_lora', False):
        return None

    from peft import LoraConfig

    target_modules = [m.strip() for m in args.lora_target_modules.split(',') if m.strip()]
    if not target_modules:
        raise ValueError('--lora_target_modules 不能为空')

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
        bias="none",
    )


def resolve_precision(args):
    """
    返回 (加载模型用的 dtype, 传给 Trainer Config 的精度参数)。

    bf16：纯 bf16 训练，参数/梯度/优化器状态都是 2 字节，显存最省（本仓库默认）。
    fp16：走 transformers 的 AMP，参数保持 fp32、前向 autocast 到 fp16、带 GradScaler，
          数值更稳但静态显存大约是 bf16 的两倍（参数 4B + 梯度 4B + AdamW 状态 8B）。
    """
    import torch

    if getattr(args, 'precision', 'bf16') == 'fp16':
        return torch.float32, {"fp16": True, "bf16": False}
    return torch.bfloat16, {"bf16": True, "fp16": False}


def validate_unbiased_para(loss, para):
    """无偏损失的 a 必须落在 [0, 1)，否则梯度方向会翻转。"""
    if loss in ('unbiased', 'normal_unbiased') and not (0.0 <= para < 1.0):
        raise ValueError(
            f'--loss {loss} 要求 --para (a) 落在 [0, 1)，当前为 {para}。'
            f'a = η/(1-η)：η=0.2 → a=0.25，η=0.4 → a=0.667。'
            f'a>=1 会让 normal=(1+√a)/(1-√a) 变号，模型朝错误方向优化。'
        )


def save_merged_model(trainer, output_path, tokenizer=None):
    """
    保存训练结果。开了 LoRA 时，trainer.save_model() 只会存 adapter，
    这里先把 adapter 合并回基座再存，保证产物是一个能直接 from_pretrained 的完整模型
    （下游的 DPO / 奖励模型脚本需要完整权重）。
    """
    model = trainer.accelerator.unwrap_model(trainer.model)
    if hasattr(model, 'merge_and_unload'):
        model = model.merge_and_unload()
    model.save_pretrained(output_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_path)



def get_logger(filename):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=filename, format=head, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger

def return_dataset_name(dataset):
    if dataset == 'hh':
        dataset_name = "trl-lib/hh-rlhf-helpful-base"
    elif dataset == 'tldr':
        dataset_name = "trl-lib/tldr-preference"
    elif dataset == 'ufb':
        dataset_name = "trl-lib/ultrafeedback_binarized"
    elif dataset == 'capybara':
        dataset_name = "trl-lib/Capybara"

    return dataset_name

def return_model_name(model):

    if model == 'qwen0.6b':
        model_name = "Qwen/Qwen3-0.6B-Base"
        tokenizer_name = "Qwen/Qwen3-0.6B-Base"
    if model == 'qwen1.7b':
        model_name = "Qwen/Qwen3-1.7B-Base"
        tokenizer_name = "Qwen/Qwen3-1.7B-Base"
    if model == 'qwen4b':
        model_name = "Qwen/Qwen3-4B-Base"
        tokenizer_name = "Qwen/Qwen3-4B-Base"
    if model == 'qwen8b':
        model_name = "Qwen/Qwen3-8B-Base"
        tokenizer_name = "Qwen/Qwen3-8B-Base"

    if model == 'llama3b':
        model_name = 'unsloth/Llama-3.2-3B'
        tokenizer_name = 'unsloth/Llama-3.2-3B-Instruct'
    if model == 'llama8b':
        model_name = 'unsloth/Meta-Llama-3.1-8B'
        tokenizer_name = 'unsloth/Meta-Llama-3.1-8B-Instruct'
    return model_name, tokenizer_name

def return_preference_dataset(dataset):
    dataset_name = return_dataset_name(dataset)

 
    if dataset == 'tldr':
        preference_train_dataset = load_dataset(dataset_name, split="train")
        # preference_train_dataset = preference_train_dataset.map(dpo_tldr)

        preference_test_dataset = load_dataset(dataset_name, split="validation")
        df = preference_test_dataset.to_pandas()
        df_deduplicated = df.drop_duplicates(subset=['prompt'], keep='first')
        preference_test_dataset = Dataset.from_pandas(df_deduplicated, preserve_index=False) 
        preference_test_dataset = preference_test_dataset.select(range(1000))
        # preference_test_dataset = preference_test_dataset.map(dpo_tldr)
    elif dataset == 'hh':
        preference_train_dataset = load_dataset(dataset_name, split="train")
        preference_test_dataset = load_dataset(dataset_name, split="test[:1000]")
    elif dataset == 'ufb':
        preference_train_dataset = load_dataset(dataset_name, split="train")
        preference_train_dataset = preference_train_dataset.map(extract_prompt)

        preference_test_dataset = load_dataset(dataset_name, split="test")
        preference_test_dataset = preference_test_dataset.map(extract_prompt)
    elif dataset == "capybara":
        preference_train_dataset = load_dataset(dataset_name, split="train")
        preference_test_dataset = load_dataset(dataset_name, split="test")
   

    return preference_train_dataset, preference_test_dataset





if __name__ == "__main__":

    # dataset_name = "trl-lib/hh-rlhf-helpful-base"
    # dpo_dataset = load_dataset(dataset_name, split="train")
    # sft_dataset = dpo_dataset.map(sft_hh, remove_columns=["prompt", "chosen", "rejected"])
    # print(sft_dataset[0])

    # dataset = load_dataset(dataset_name, split="test")
    # dataset = dataset.map(concat_prompt_hh, remove_columns=["chosen", "rejected"])
    # print(dataset[0])

    dataset_name = "trl-lib/ultrafeedback_binarized"
    dpo_dataset = load_dataset(dataset_name, split="train")
    # dpo_dataset = dpo_dataset.map(dpo_hh)
    print(dpo_dataset[0])
  
