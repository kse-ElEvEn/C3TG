# C³TG — Model Training and Main Scripts

This repository contains the core scripts for training attribute-specific classification models and language models, as well as the multi-attribute controlled text generation pipeline (C³TG).


## Scripts Overview

### 1. `train_bert_multilabel.py`

Train a BERT-based multi-class classifier for a single attribute. Outputs logits over `num_labels` classes.

Usage example:

```bash
python scripts/train_bert_multilabel.py \
  --train_json /path/to/emotion_joy.json \
  --bert_base bert-base-uncased \
  --save_path models/emotion_joy.pt \
  --num_labels 6 \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5
```

Key arguments:

* `--train_json`: JSON file with `[{"text":..., "label":...}, ...]` entries.
* `--bert_base`: Pretrained BERT model path or name.
* `--save_path`: Output path for the trained `.pt` model.
* `--num_labels`: Number of target classes (e.g., 6 for emotions).

### 2. `train_llama2_finegrained.py`

Fine-tune LLaMA2 for a single attribute-dimension on causal language modeling.

Usage example:

```bash
python scripts/train_llama2_finegrained.py \
  --train_json /path/to/style_humor.json \
  --pretrained_model llama2 \
  --save_dir models/llama2_humor \
  --epochs 3 \
  --batch_size 8 \
  --lr 5e-5
```

Key arguments:

* `--train_json`: JSON file with a `text` field in each entry.
* `--pretrained_model`: Base LM name or local path.
* `--save_dir`: Directory to save fine-tuned model and tokenizer.
* `--block_size`: Token block size for LM training.

### 3. `multi_attribute_tokenlm.py`

The main C³TG pipeline implementing multi-attribute controlled generation using:

1. Pre-iteration rewrite via KL-based mixing of base and attribute LMs.
2. Three iterative stages guided by energy computation (classifier\_diff and overlap\_diff).
3. BERT classifiers for feedback scoring.

Usage example:

```bash
python scripts/multi_attribute_tokenlm.py \
  --base_model_path path/to/llama2_base \
  --attributes "joy=0.8, humor=0.6, Knowledge=0.5" \
  --llama2_dir models/ \
  --emo_bert_path models/emotion_joy.pt \
  --sty_bert_path models/style_humor.pt \
  --ton_bert_path models/tone_casual.pt \
  --top_bert_path models/topic_knowledge.pt \
  --tox_bert_path models/toxic.pt \
  --energy_threshold 0.5
```

## Dependencies

* Python 3.8+
* PyTorch
* Transformers (Hugging Face)
* scikit-learn
* tqdm



