#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_attribute_tokenlm.py
   classifier_diff = sum( |C_{A_i}(x) - T_i| ), i∈用户指定属性
   overlap_diff = sum( |C_{A_j}(x) - C_{A_j}(x_prev)| ), j∈所有(或相应)属性
   alpha = classifier_diff / (classifier_diff + overlap_diff)
   beta  = overlap_diff / (classifier_diff + overlap_diff)
   E = alpha * classifier_diff + beta * overlap_diff

"""

import argparse
import re
import sys
import math
import torch
import torch.nn.functional as F
from openai import OpenAI
import openai

from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_classfy import BERTMultiClassifier
import torch


############################################################################
# A. 命令行解析
############################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="LLaMA2基础模型路径")
    parser.add_argument("--attributes", type=str, default="",
                        help='逗号分隔 维度=分数, 如 "joy=0.8, humor=0.6"')
    parser.add_argument("--distilgpt2_dir", type=str, required=True,
                        help="存放多个distilgpt2_{dim}模型的路径")
    # BERT 多分类
    parser.add_argument("--emo_bert_path", type=str, default=None)
    parser.add_argument("--sty_bert_path", type=str, default=None)
    parser.add_argument("--ton_bert_path", type=str, default=None)
    parser.add_argument("--top_bert_path", type=str, default=None)
    parser.add_argument("--tox_bert_path", type=str, default=None)

    parser.add_argument("--initial_prompt", type=str, default="Once upon a time...")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--log_file", type=str, default="multi_attr_chain.log")
    parser.add_argument("--openai_api_key", type=str, default="sk-xxxxx")

    # 能量阈值(可选提前停止)
    parser.add_argument("--energy_threshold", type=float, default=0.5,
                        help="若能量E<该阈值,可视为满足要求,提前结束后续阶段")
    return parser.parse_args()


############################################################################
# B. 日志 Tee
############################################################################
class Tee(object):
    def __init__(self, path, mode='w'):
        self.file = open(path, mode, encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


############################################################################
# C. 加载 LLaMA2 + DistilGPT2
############################################################################
def load_lm(path, device):
    # 加载 tokenizer 和 model，支持指定 device
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
    model.eval().to(device)
    return tokenizer, model

@torch.no_grad()
def get_probs(tokenizer, model, input_ids):
    out = model(input_ids)
    logits = out.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    return probs

def map_ids(src_tok, dst_tok, input_ids, device):
    text = src_tok.decode(input_ids[0])
    new_ids = dst_tok.encode(text, return_tensors='pt').to(device)
    return new_ids

def combine_probs(base_probs, attr_probs_list, lam_list):
    """
    最终概率 = softmax( lam0*log(base_probs) + sum_i(lam_i*log(attr_probs_list[i])) )
    """
    import torch
    log_p = torch.log(base_probs + 1e-12) * lam_list[0]
    idx = 1
    for p_attr in attr_probs_list:
        log_p += torch.log(p_attr + 1e-12) * lam_list[idx]
        idx += 1
    max_v = torch.max(log_p)
    exp_v = torch.exp(log_p - max_v)
    sum_exp = torch.sum(exp_v)
    final_p = exp_v / sum_exp
    return final_p

def compute_mapping(base_tok, dtok):
    """
    对于基础模型词表中的每个 token，尝试用属性模型的 tokenizer 得到单个 token 的 ID。
    若无法单一token对应，则返回 -1。
    """
    mapping = []
    for i in range(len(base_tok)):
        token_str = base_tok.convert_ids_to_tokens(i)
        encoded = dtok.encode(token_str, add_special_tokens=False)
        if len(encoded) == 1:
            mapping.append(encoded[0])
        else:
            mapping.append(-1)
    return torch.tensor(mapping, dtype=torch.long)

def map_probability(p_attr, mapping, device):
    base_size = mapping.shape[0]
    p_attr_mapped = torch.full((base_size,), 1e-12, device=device, dtype=p_attr.dtype)
    valid_mask = mapping >= 0
    p_attr_mapped[valid_mask] = p_attr[mapping[valid_mask]]
    return p_attr_mapped


@torch.no_grad()
def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, filter_value=-float('Inf')):

    # logits: [vocab_size]
    top_k = min(top_k, logits.size(-1))  # 防止k>词表大小
    # 按logits从大到小排序, 返回index
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    if sorted_indices_to_remove[0] == True:
        sorted_indices_to_remove[0] = False

    sorted_indices_to_remove[top_k:] = True

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits

def apply_repetition_penalty(logits, generated_ids, penalty=1.2):

    # 如果 penalty==1.0, 表示不做惩罚
    if penalty == 1.0:
        return logits

    # 对每个已经出现过的token进行惩罚
    for token_id in generated_ids:
        # logits[token_id] = logits[token_id] / penalty
        logits[token_id] /= penalty
    return logits


@torch.no_grad()
def kl_generate(
    prompt, max_len, device,
    base_tok, base_mod, dim_models_list, mappings,
    # 下面是新增的若干参数
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2
):
    input_ids = base_tok.encode(prompt, return_tensors='pt').to(device)
    generated_tokens = list(input_ids[0].cpu().numpy())  # 用于重复惩罚
    gen_text = prompt

    base_lam = 1.0
    lam_list = [base_lam] + [x[3] for x in dim_models_list]

    for step in range(max_len):
        # 1) 计算基础模型的概率
        out = base_mod(input_ids)
        logits = out.logits[:, -1, :].squeeze(0)  # [vocab_size]
        base_probs = torch.softmax(logits, dim=-1)

        # 2) 计算各属性模型的概率
        attr_probs_list = []
        for idx, (dname, dtok, dmod, dlam) in enumerate(dim_models_list):
            dmod_gpu = dmod.to(device)
            new_ids = map_ids(base_tok, dtok, input_ids, device)
            out2 = dmod_gpu(new_ids)
            logits2 = out2.logits[:, -1, :].squeeze(0)
            p_dim = torch.softmax(logits2, dim=-1)
            # 映射到基础词表
            mapped_p_dim = map_probability(p_dim, mappings[idx], device)
            attr_probs_list.append(mapped_p_dim)
            # 移回CPU
            dmod_gpu.cpu()
            torch.cuda.empty_cache()

        # 3) 合并概率
        log_p = torch.log(base_probs + 1e-12) * lam_list[0]
        idx_attr = 1
        for p_attr in attr_probs_list:
            log_p += torch.log(p_attr + 1e-12) * lam_list[idx_attr]
            idx_attr += 1

        # 4) 转为 logits 进行采样处理
        #    log_p 就是最终的 (log(p1)*lam1 + ... ), 其实不是一个真正logits,
        #    但足够近似可以当 logits 用了
        # 先做 repetition penalty:
        final_logits = log_p.clone()
        final_logits = apply_repetition_penalty(final_logits, generated_tokens, penalty=repetition_penalty)

        # 然后做 temperature 缩放
        if abs(temperature - 1.0) > 1e-5:
            final_logits = final_logits / temperature

        # 再做 top_k & top_p 截断
        final_logits = top_k_top_p_filtering(final_logits, top_k=top_k, top_p=top_p)

        # 5) 归一化 => 得到最终的 probability
        final_probs = torch.softmax(final_logits, dim=-1)

        # 6) 采样
        next_token = torch.multinomial(final_probs, 1)
        if next_token.item() == base_tok.eos_token_id:
            break

        # 7) 拼接
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        generated_tokens.append(next_token.item())
        gen_text += base_tok.decode([next_token.item()])

    return gen_text

# def kl_generate(prompt, max_len, device, base_tok, base_mod, dim_models_list, mappings):
#     """
#     生成过程中，利用预先计算的 mappings 将各属性模型的概率映射到基础模型词表空间。

#     dim_models_list: [ (dim_name, dtok, dmod, lam), ...]
#     mappings: 与 dim_models_list 顺序对应，每个为预计算的 mapping 张量
#     """
#     input_ids = base_tok.encode(prompt, return_tensors='pt').to(device)
#     gen_text = prompt
#     base_lam = 1.0
#     lam_list = [base_lam] + [x[3] for x in dim_models_list]

#     for step in range(max_len):
#         base_probs = get_probs(base_tok, base_mod, input_ids)
#         attr_probs_list = []
#         for idx, (dname, dtok, dmod, dlam) in enumerate(dim_models_list):
#             # 临时将 DistilGPT2 模型迁移到 GPU
#             temp_dmod = dmod.to(device)
#             new_ids = map_ids(base_tok, dtok, input_ids, device)
#             p_dim = get_probs(dtok, temp_dmod, new_ids)
#             mapped_p_dim = map_probability(p_dim, mappings[idx], device)
#             attr_probs_list.append(mapped_p_dim)

#             # 计算完移回 CPU, 释放显存
#             temp_dmod.cpu()
#             torch.cuda.empty_cache()

#         final_probs = combine_probs(base_probs, attr_probs_list, lam_list)
#         next_token = torch.multinomial(final_probs, 1)
#         if next_token.item() == base_tok.eos_token_id:
#             break
#         input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
#         gen_text += base_tok.decode(next_token.tolist())
#     return gen_text


############################################################################
# D. BERT多分类(5大属性)
############################################################################
def load_bert_classifier(pt_path, device):
    model = torch.load(pt_path, map_location=device)
    model.eval()
    return model

def get_multiclass_probs(model, tokenizer, text, device, max_len=128):
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len)
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(enc['input_ids'], enc['attention_mask'])
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    return probs.detach().cpu().numpy()

def compute_scores(
    text, device, shared_bert_tok,
    emo_bert=None, sty_bert=None, ton_bert=None, top_bert=None, tox_bert=None
):
    """
      emotion: [p_sadness, p_joy, p_love, p_anger, p_fear, p_surprise]
      style:   [p_humor, p_metaphor, p_polite, p_romance, p_sarcasm]
      tone:    [p_casual, p_professional]
      topic:   [p_world, p_sports, p_business, p_science]
      toxicity:[p_toxic, p_non_toxic]
    """
    results = {}
    if emo_bert:
        e_probs = get_multiclass_probs(emo_bert, shared_bert_tok, text, device)
        results["emotion"] = e_probs.tolist()
    if sty_bert:
        s_probs = get_multiclass_probs(sty_bert, shared_bert_tok, text, device)
        results["style"] = s_probs.tolist()
    if ton_bert:
        t_probs = get_multiclass_probs(ton_bert, shared_bert_tok, text, device)
        results["tone"] = t_probs.tolist()
    if top_bert:
        p_probs = get_multiclass_probs(top_bert, shared_bert_tok, text, device)
        results["topic"] = p_probs.tolist()
    if tox_bert:
        x_probs = get_multiclass_probs(tox_bert, shared_bert_tok, text, device)
        results["toxicity"] = x_probs.tolist()
    return results


############################################################################
# E. 调用 OpenAI GPT 接口的封装函数
############################################################################
def askChatGPT(prompt, model="gpt-4", temperature=0.9):
    messages = [{"role": "user", "content": prompt}]
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Warn] GPT-{model} call failed:", e)
        return None


############################################################################
# F. 计算 classifier_diff 与 overlap_diff
############################################################################

def build_attribute_score_map(scores_dict):
    # 先给所有可能的维度初始化0
    all_dims = [
        "sadness","joy","love","anger","fear","surprise",
        "humor","metaphor","polite","romance","sarcasm",
        "casual","professional",
        "world","sports","business","science",
        "toxic","non-toxic"
    ]
    score_map = {dn:0.0 for dn in all_dims}

    # 若对应分类器存在，则写入真实数值
    # emotion: sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5)
    # style:   humor(0), metaphor(1), polite(2), romance(3), sarcasm(4)
    # tone:    casual(0), professional(1)
    # topic:   world(0), sports(1), business(2), science(3)
    # toxicity: toxic(0), non-toxic(1)

    if "emotion" in scores_dict:
        e = scores_dict["emotion"]
        score_map["sadness"]   = e[0]
        score_map["joy"]       = e[1]
        score_map["love"]      = e[2]
        score_map["anger"]     = e[3]
        score_map["fear"]      = e[4]
        score_map["surprise"]  = e[5]

    if "style" in scores_dict:
        s = scores_dict["style"]
        score_map["humor"]     = s[0]
        score_map["metaphor"]  = s[1]
        score_map["polite"]    = s[2]
        score_map["romance"]   = s[3]
        score_map["sarcasm"]   = s[4]

    if "tone" in scores_dict:
        t = scores_dict["tone"]
        score_map["casual"]        = t[0]
        score_map["professional"]  = t[1]

    if "topic" in scores_dict:
        p = scores_dict["topic"]
        score_map["world"]    = p[0]
        score_map["sports"]   = p[1]
        score_map["business"] = p[2]
        score_map["science"]  = p[3]

    if "toxicity" in scores_dict:
        x = scores_dict["toxicity"]
        score_map["toxic"]     = x[0]
        score_map["non-toxic"] = x[1]

    return score_map


def compute_classifier_diff(score_map, user_targets):
    """
    classifier_diff = sum( |score_map[dim] - T_i| ), i∈用户指定属性
    """
    diff_val = 0.0
    for dim, target_val in user_targets.items():
        curr_score = score_map.get(dim, 0.0)
        diff_val += abs(curr_score - target_val)
    return diff_val


def compute_overlap_diff(score_map_now, score_map_prev):
    """
    overlap_diff = sum( |score_map_now[dim] - score_map_prev[dim]| ) over all dims
    """
    diff_val = 0.0
    all_dims = set(score_map_now.keys()).union(set(score_map_prev.keys()))
    for dim in all_dims:
        diff_val += abs(score_map_now[dim] - score_map_prev[dim])
    return diff_val


############################################################################
# G. Main + 先改写 + 三阶段
############################################################################
def main():
    args = parse_args()
    tee = Tee(args.log_file)
    sys.stdout = tee
    sys.stderr = tee

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_device = torch.device("cpu")
    print("[INFO] GPU device=", device)
    print("[INFO] args=", args)

    # 初始化OpenAI
    global client
    client = OpenAI(
        base_url="https://fast.aigcbest.top/v1",
        api_key=args.openai_api_key
    )

    # 1) 解析用户 attributes => user_targets
    user_targets = {}
    if args.attributes.strip():
        pairs = re.split(r'\s*,\s*', args.attributes.strip())
        for p in pairs:
            if '=' in p:
                k, v = p.split('=')
                user_targets[k.strip()] = float(v.strip())

    # 强制 non-toxic = 1
    if "non-toxic" not in user_targets:
        user_targets["non-toxic"] = 1.0

    print("[INFO] user-specified attributes (with forced non-toxic=1):", user_targets)

    # 2) 加载 LLaMA2(base)
    base_tok, base_mod = load_lm(args.base_model_path, device)

    # 3) 加载 DistilGPT2 模型(仅用户targets + toxicity)
    dim_models_list = []

    def load_dim_model(attr, lam):
        mapping = {
            # emotion
            "sadness":   "emotion_sadness",
            "joy":       "emotion_joy",
            "love":      "emotion_love",
            "anger":     "emotion_anger",
            "fear":      "emotion_fear",
            "surprise":  "emotion_surprise",
            # style
            "humor":     "style_humor",
            "metaphor":  "style_metaphor",
            "polite":    "style_polite",
            "romance":   "style_romance",
            "sarcasm":   "style_sarcasm",
            # tone
            "casual":       "tone_casual",
            "professional": "tone_professional",
            # topic
            "world":    "topic_world",
            "sports":   "topic_sports",
            "business": "topic_business",
            "science":  "topic_science",
            # toxicity
            "toxicity":  "toxic",
            "non-toxic": "toxic_non"
        }
        folder_name = mapping.get(attr, f"distilgpt2_{attr}")
        path = f"{args.distilgpt2_dir}/{folder_name}"
        tok_dim, mod_dim = load_lm(path, torch.device("cpu"))
        # lam 先直接用 user_targets[attr], 也可另行指定
        dim_models_list.append((attr, tok_dim, mod_dim, lam))

    # 根据 user_targets 加载
    for dname, val in user_targets.items():
        load_dim_model(dname, val)

    print("[INFO] DistilGPT2 dims loaded:")
    for item in dim_models_list:
        print("  ", item[0], "=> lam=", item[3])

    # 预计算词表映射
    mappings = []
    for (dname, dtok, dmod, dlam) in dim_models_list:
        mp = compute_mapping(base_tok, dtok).to(device)
        mappings.append(mp)

    # 4) 加载 BERT (5大属性) => CPU
    from transformers import BertTokenizer
    shared_bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")

    emo_bert = load_bert_classifier(args.emo_bert_path, bert_device) if args.emo_bert_path else None
    sty_bert = load_bert_classifier(args.sty_bert_path, bert_device) if args.sty_bert_path else None
    ton_bert = load_bert_classifier(args.ton_bert_path, bert_device) if args.ton_bert_path else None
    top_bert = load_bert_classifier(args.top_bert_path, bert_device) if args.top_bert_path else None
    tox_bert = load_bert_classifier(args.tox_bert_path, bert_device) if args.tox_bert_path else None


    # ========== Pre-Iteration Rewrite =============
    rewrite_prompt = (
        "Rewrite the following text according to specified attributes. "
        "Make sure the final text is non-toxic.\n\n"
        f"=== Original ===\n{args.initial_prompt}\n"
        "=== Rewritten ===\n"
    )
    text0 = kl_generate(
        prompt=rewrite_prompt,
        max_len=128,
        device=device,
        base_tok=base_tok,
        base_mod=base_mod,
        dim_models_list=dim_models_list,
        mappings=mappings,
        temperature=0.9,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2
    )
    # kl_generate(
    #     prompt=rewrite_prompt,
    #     max_len=args.max_length,
    #     device=device,
    #     base_tok=base_tok, base_mod=base_mod,
    #     dim_models_list=dim_models_list,
    #     mappings=mappings
    # )

    print("\n=== [Pre-Iteration Rewrite] ===")
    print("Rewrite Prompt:\n", rewrite_prompt)
    print("Result text0:\n", text0)

    # 计算 text0 的分数 => 作为 stage0
    scores0 = compute_scores(
        text0, bert_device, shared_bert_tok,
        emo_bert=emo_bert, sty_bert=sty_bert, ton_bert=ton_bert, top_bert=top_bert, tox_bert=tox_bert
    )
    score_map0 = build_attribute_score_map(scores0)


    # 为方便 overlap 计算，定义 text[-1] = text0（意味着 stage1 overlap=0）
    text_list = [None]*4  # 0..3
    text_list[0] = text0

    score_map_list = [None]*4
    score_map_list[0] = score_map0


    # ============== 定义每个阶段的过程函数 =====================
    def run_stage(i):

        txt_prev = text_list[i-1]
        score_map_prev = score_map_list[i-1]

        # overlap 参照text[i-2], 对 i=1 时 => i-2 = -1 => 我们当做 text[0]
        if i == 1:
            # overlap=0
            score_map_before_that = score_map_list[0]
        else:
            score_map_before_that = score_map_list[i-2]

        # 先计算当前文本(=text[i-1])的 classifier_diff, overlap_diff
        c_diff = compute_classifier_diff(score_map_prev, user_targets)
        o_diff = compute_overlap_diff(score_map_prev, score_map_before_that)

        # alpha,beta
        s = c_diff + o_diff
        if s > 1e-8:
            alpha = c_diff / s
            beta  = o_diff / s
        else:
            alpha = 0.0
            beta  = 0.0
        E_val = alpha * c_diff + beta * o_diff

        # 将这些信息 + 当前文本分数信息 => GPT-4, 让它输出"改写指令"
        # 还需要把分类器概率一并给出去,以便 GPT-4 可以分析具体维度
        # 你也可只给差值/简要信息,本示例中为方便,直接输出score_map
        stage_prompt = f"""
We are in Stage {i}.
Here is the current text:

"{txt_prev}"

Below is the classification result for this text (by a custom classifier):
{score_map_prev}

User attribute targets are:
{user_targets}

We define:
- classifier_diff = sum of absolute differences from target = {c_diff:.4f}
- overlap_diff = sum of absolute differences from previous stage's text = {o_diff:.4f}
Thus:
- alpha = classifier_diff / (classifier_diff+overlap_diff) = {alpha:.4f}
- beta = overlap_diff / (classifier_diff+overlap_diff) = {beta:.4f}
So the energy E = alpha*classifier_diff + beta*overlap_diff = {E_val:.4f}

Please produce a rewriting instruction (prompt) that, when followed, can reduce the above energy E,
i.e., reduce the difference from user targets while not unnecessarily disturbing the previously established attributes.
Remember we want to keep the text non-toxic.

Important: Only output the rewriting instruction. Do not rewrite the text yourself.
"""

        gpt4_response = askChatGPT(stage_prompt, model="gpt-4")
        if not gpt4_response:
            print(f"[Warn] GPT-4 stage{i} failed, using fallback => trivial prompt.")
            gpt4_response = f"Rewrite the text in a better way to reduce E.\n\n{txt_prev}"

        print(f"\n=== [Stage {i}] GPT-4 rewriting-instruction Prompt ===")
        print(stage_prompt)
        print(f"\n--- GPT-4 Output (rewriting instruction) ---\n{gpt4_response}\n")

        # 用 kl_generate() 执行该指令 => 生成 text_i
        text_new = kl_generate(
            prompt=gpt4_response,
            max_len=args.max_length,
            device=device,
            base_tok=base_tok, base_mod=base_mod,
            dim_models_list=dim_models_list,
            mappings=mappings
        )
        # 得到text_i, 计算评分
        scores_new = compute_scores(
            text_new, bert_device, shared_bert_tok,
            emo_bert=emo_bert, sty_bert=sty_bert, ton_bert=ton_bert, top_bert=top_bert, tox_bert=tox_bert
        )
        score_map_new = build_attribute_score_map(scores_new)

        # 再次计算本轮生成的 classifier_diff / overlap_diff / E
        # (因为 text_new 跟 txt_prev 可能不同, overlap 应该是 text_new vs text[i-1], etc.)
        c_diff2 = compute_classifier_diff(score_map_new, user_targets)
        o_diff2 = compute_overlap_diff(score_map_new, score_map_prev)
        s2 = c_diff2 + o_diff2
        if s2 > 1e-8:
            alpha2 = c_diff2 / s2
            beta2  = o_diff2 / s2
        else:
            alpha2 = 0.0
            beta2  = 0.0
        E_val2 = alpha2 * c_diff2 + beta2 * o_diff2

        print(f"--- After applying GPT-4 rewriting instruction, we used KL to generate: ---\n{text_new}\n")
        print(f"New scores: {scores_new}")
        print(f"=> classifier_diff={c_diff2:.4f}, overlap_diff={o_diff2:.4f}, alpha={alpha2:.4f}, beta={beta2:.4f}, E={E_val2:.4f}")

        text_list[i] = text_new
        score_map_list[i] = score_map_new

        # 若 E < threshold, 则可提前结束
        if E_val2 < args.energy_threshold:
            print(f"[INFO] E={E_val2:.4f} < threshold={args.energy_threshold}. Will end now.\n")
            return True  # 表示要结束
        else:
            return False

    # ============== 进行三次阶段迭代 =================
    end_now = run_stage(1)
    if not end_now:
        end_now = run_stage(2)
    if not end_now:
        end_now = run_stage(3)

    # 最终文本
    final_text = text_list[3] if text_list[3] is not None else text_list[2] if text_list[2] is not None else text_list[1]
    print("\n=== [Final Result] ===")
    print(final_text)

    sys.stdout.file.close()


if __name__ == "__main__":
    main()
