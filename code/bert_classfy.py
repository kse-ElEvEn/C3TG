#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_bert_multilabel.py

用途：训练 BERT 多分类模型 (单一属性-多类别)，输出 [batch, num_labels] 的概率分布。
比如 Emotion(6类)、Style(5类)、Tone(2类)、Topic(4类)、Toxic(6类) 等。

示例：
  python train_bert_multilabel.py \
      --train_json /data/emotion_output.json \
      --bert_base /data/bert-base-uncased \
      --save_path /data/emotion_model.pt \
      --num_labels 6 \
      --epochs 3
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

################################################################
# 1. 数据集
################################################################
class TextDataset(Dataset):
    """
    读取 JSON 数据：[{text:..., label:...}, ...]，label=0..N-1
    """
    def __init__(self, json_file, tokenizer, max_length=128):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.texts = []
        self.labels= []
        for item in data:
            self.texts.append(item['text'])
            self.labels.append(int(item['label']))

        self.tokenizer = tokenizer
        self.max_length= max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label= self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids= encoding['input_ids'].squeeze(0)
        attn_mask= encoding['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

################################################################
# 2. 模型
################################################################
class BERTMultiClassifier(nn.Module):
    """
    多分类 BERT：pooler_output -> dropout -> classifier -> [batch, num_labels]
    """
    def __init__(self, bert_base, num_labels):
        super(BERTMultiClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_base)
        self.dropout= nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs= self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output= outputs.pooler_output
        pooled_output= self.dropout(pooled_output)
        logits= self.classifier(pooled_output)
        return logits


################################################################
# 3. 训练和评估
################################################################
def train_and_eval(model, train_loader, val_loader, device, epochs=3, lr=2e-5):
    criterion= nn.CrossEntropyLoss()
    optimizer= optim.AdamW(model.parameters(), lr=lr)
    best_f1= 0.0

    for epoch in range(1, epochs+1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        model.train()
        total_loss= 0.0
        for batch in tqdm(train_loader, desc="Train"):
            input_ids= batch['input_ids'].to(device)
            attn_mask= batch['attention_mask'].to(device)
            labels= batch['labels'].to(device)

            logits= model(input_ids, attn_mask)
            loss= criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+= loss.item()
        avg_loss= total_loss / len(train_loader)
        print(f"[Epoch {epoch}] train loss={avg_loss:.4f}")

        # 验证
        model.eval()
        all_preds=[]
        all_labels=[]
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                input_ids= batch['input_ids'].to(device)
                attn_mask= batch['attention_mask'].to(device)
                labels= batch['labels'].to(device)

                logits= model(input_ids, attn_mask)
                preds= torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc= accuracy_score(all_labels, all_preds)
        f1= f1_score(all_labels, all_preds, average='macro')
        print(f"[Epoch {epoch}] val acc={acc:.4f}, f1={f1:.4f}")
        if f1> best_f1:
            best_f1= f1
    print(f"Training done. Best val F1={best_f1:.4f}")


################################################################
# 4. 命令行解析 & 主函数
################################################################
def parse_args():
    parser= argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--bert_base", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True,
                        help="保存训练好模型的 .pt 文件路径")
    parser.add_argument("--num_labels", type=int, required=True,
                        help="多分类标签总数 (如Emotion=6, Style=5, Tone=2, Topic=4, Toxic=6)")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--log_file", type=str, default="train_bert_multilabel.log")
    return parser.parse_args()

def main():
    args= parse_args()
    sys.stderr.write(f"[LOG] Writing to {args.log_file}\n")
    sys.stdout= open(args.log_file, "w", encoding='utf-8')

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    tokenizer= BertTokenizer.from_pretrained(args.bert_base)
    dataset= TextDataset(args.train_json, tokenizer, max_length=args.max_length)
    total_len= len(dataset)
    train_len= int(args.train_split * total_len)
    val_len= total_len- train_len
    train_set, val_set= random_split(dataset, [train_len, val_len])
    train_loader= DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader= DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)

    model= BERTMultiClassifier(args.bert_base, num_labels=args.num_labels).to(device)
    train_and_eval(model, train_loader, val_loader, device, args.epochs, args.lr)

    print(f"[INFO] Saving model to {args.save_path}")
    torch.save(model, args.save_path)
    print("[INFO] Done.")

if __name__=="__main__":
    main()
