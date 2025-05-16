#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_llama2_finegrained.py

用途：针对单一“属性-维度”的文本语料，微调语言模型；
例如：
  - Emotion-Joy: emotion_joy.json
  - Style-Humor: style_humor.json
  - Tone-Professional: tone_professional.json
  - Toxic: 有毒语料/无毒语料(根据需求)
运行示例:
  python train_llama2_finegrained.py \
    --train_json emotion_joy.json \
    --pretrained_model llama2 \
    --save_dir llama2_joy \
    --epochs 3 \
    --lr 5e-5 \
    --log_file train_joy.log
"""
import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


import argparse
import json
import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from tqdm import tqdm

#################################################################
# 1. 数据集定义
#################################################################
class LMJsonDataset(Dataset):
    """
    从 JSON 文件中加载文本(字段 'text')，对每段文本进行 tokenizer 编码，
    分块(block_size)后用于因果语言模型训练。
    """
    def __init__(self, json_file, tokenizer, block_size=128):
        self.examples = []
        self.tokenizer= tokenizer
        self.block_size= block_size

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = [item["text"] for item in data]

        print(f"[INFO] Loading & tokenizing from {json_file} ...")
        for txt in tqdm(texts):
            tokens = tokenizer.encode(txt, add_special_tokens=True)
            i = 0
            while i < len(tokens):
                chunk = tokens[i:i+block_size]
                self.examples.append(chunk)
                i += block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

class DataCollatorForLM:
    """
    用于Causal LM训练:
      1) 对齐不同长度的input_ids
      2) 将 pad 的部分标签设为 -100
    """
    def __init__(self, tokenizer):
        self.tokenizer= tokenizer

    def __call__(self, examples):
        max_len= max(len(x) for x in examples)
        input_ids_batch= []
        for ex in examples:
            pad_len= max_len - len(ex)
            if pad_len>0:
                ex= torch.cat([ex, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            input_ids_batch.append(ex.unsqueeze(0))
        input_ids_batch= torch.cat(input_ids_batch, dim=0)  # [batch, seq_len]

        labels= input_ids_batch.clone()
        labels[labels== self.tokenizer.pad_token_id]= -100
        return {"input_ids": input_ids_batch, "labels": labels}

#################################################################
# 2. 训练流程
#################################################################
def parse_args():
    parser= argparse.ArgumentParser(description="Fine-tune llama2 for a single attribute-dimension.")
    parser.add_argument("--train_json", type=str, required=True,
                        help="JSON文件，内部含 'text' 字段。只针对单一维度语料。")
    parser.add_argument("--pretrained_model", type=str, default="llama2",
                        help="基础模型名称或路径，如 llama2")
    parser.add_argument("--save_dir", type=str, default="./llama2_fine",
                        help="微调后模型保存目录")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--log_file", type=str, default="train_llama2.log")
    return parser.parse_args()

def main():
    args= parse_args()
    sys.stderr.write(f"[INFO] Logging to {args.log_file}\n")
    sys.stdout= open(args.log_file, "w", encoding="utf-8")

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] pretrained_model={args.pretrained_model}")
    print(f"[INFO] train_json={args.train_json}")

    # 1) 加载 tokenizer & model
    tokenizer= AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token= tokenizer.eos_token

    model= AutoModelForCausalLM.from_pretrained(args.pretrained_model)
    model.to(device)

    # 2) 构建数据集
    dataset= LMJsonDataset(args.train_json, tokenizer, block_size=args.block_size)
    data_collator= DataCollatorForLM(tokenizer)

    # 3) 训练配置
    training_args= TrainingArguments(
        output_dir= args.save_dir,
        overwrite_output_dir= True,
        num_train_epochs= args.epochs,
        per_device_train_batch_size= args.batch_size,
        save_steps= args.save_steps,
        logging_steps= args.logging_steps,
        learning_rate= args.lr,
        weight_decay= 0.01,
        do_train= True,
        do_eval= False,
        save_total_limit= 1
    )

    # 4) Trainer
    trainer= Trainer(
        model= model,
        args= training_args,
        train_dataset= dataset,
        data_collator= data_collator
    )

    print("[INFO] Starting training ...")
    trainer.train()

    print("[INFO] Saving final model to", args.save_dir)
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    print("[INFO] Done.")
    sys.stdout.close()


if __name__=="__main__":
    main()
