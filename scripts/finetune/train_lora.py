"""
train_lora.py — QLoRA 微调脚本
===============================
使用 4-bit 量化 + LoRA 对 Qwen-2.5 进行领域微调。

硬件需求:
  - 最低: 16GB VRAM (4-bit Qwen-2.5-7B)
  - 推荐: 24GB VRAM (4-bit Qwen-2.5-14B)

运行:
  python scripts/finetune/train_lora.py --data data/finetune/train_data.jsonl --output models/lora-manufacturing

首次运行会自动下载基座模型 (~15GB)。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_data(data_path: str) -> list[dict]:
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"加载训练数据: {len(data)} 条")
    return data


def format_prompt(example: dict) -> str:
    """构建 instruction 格式的 prompt。"""
    return f"""<|im_start|>system
{example['instruction']}<|im_end|>
<|im_start|>user
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""


def train(args):
    """执行 QLoRA 微调。"""
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install torch transformers peft bitsandbytes datasets accelerate")
        sys.exit(1)

    # 1. 加载数据
    data = load_data(args.data)
    dataset = Dataset.from_list(data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        prompts = [format_prompt(ex) for ex in [dict(zip(examples.keys(), v)) for v in zip(*examples.values())]]
        result = tokenizer(prompts, truncation=True, max_length=args.max_length, padding=False)
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    # 2. 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. 加载基座模型
    print(f"加载基座模型: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA 配置
    lora_config = LoraConfig(
        r=args.lora_r,           # LoRA rank
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        report_to="none",  # 不用 wandb
    )

    # 6. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"开始训练: {len(dataset)} 样本, {args.epochs} epochs")
    trainer.train()

    # 7. 保存
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"模型已保存: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA 微调")
    parser.add_argument("--data", default="data/finetune/train_data.jsonl", help="训练数据路径")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="基座模型")
    parser.add_argument("--output", default="models/lora-manufacturing", help="输出目录")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    args = parser.parse_args()

    print(f"QLoRA 微调配置:")
    print(f"  基座模型: {args.model}")
    print(f"  训练数据: {args.data}")
    print(f"  输出目录: {args.output}")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  训练轮数: {args.epochs}, 批次: {args.batch_size}")
    print()

    train(args)


if __name__ == "__main__":
    main()
