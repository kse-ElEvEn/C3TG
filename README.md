# C³TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation


> **Fine-grained, multi-attribute text control *without* model surgery** — leverage the power of large language models together with lightweight attribute classifiers to generate coherent, diverse, and **conflict‑free** text.



<img width="1239" alt="image" src="https://github.com/user-attachments/assets/4cfd9c1d-1621-4be6-a03b-152b7cc750c6" />



---

## ✨ Key Highlights

* **Two‑phase framework**: *Generation* (weighted KL fusion) + *Optimization* (energy‑based iterative refinement)
* **17+ controllable sub‑attributes** across *emotion, style, tone, topic,* and *toxicity*
* **Conflict‑aware**: overlap penalty resolves attribute interference while preserving fluency
* **Lightweight & flexible**: no architecture change; plug‑in classifiers fine‑tuned with LoRA
* **State‑of‑the‑art** accuracy and fluency on multiple open‑ended generation benchmarks

---

## 📑 Table of Contents

1. [Method Overview](#method-overview)
2. [Results](#results)
3. [Citation](#citation)
4. [License](#license)
5. [Contact & Contributing](#contact--contributing)

---

## 🔬 Method Overview


<img width="892" alt="image" src="https://github.com/user-attachments/assets/af5a8411-df23-4cc5-ac29-07b4342b7bcf" />


C³TG couples a **base language model** (LLaMA‑2 13B in our experiments) with a bank of **attribute classifiers** fine‑tuned on emotion, style, tone, topic, and toxicity labels. Generation unfolds in two tightly connected phases:

1. **Generation Phase**
   A weighted geometric mean combines the attribute‑specific priors $Q_i$ into a single token distribution $P$, minimising a *weighted KL divergence* objective

   $$
   \mathcal J[P] = \sum_{i=1}^{n} \lambda_i\, D_{\mathrm{KL}}\bigl(P\;\|\;Q_i\bigr).
   $$
2. **Optimization Phase**
   An **energy function** blends classifier deviations with an *overlap penalty* that discourages mutually contradicting attributes. A small‑step optimiser adjusts logits, and a feedback agent performs *three* local rewriting cycles for fine‑grained polish.

> **Why it matters**: The design achieves *simultaneous, conflict‑aware control* **without touching** the backbone model, making C³TG a drop‑in module for any decoder‑style LLM.

<img width="628" alt="image" src="https://github.com/user-attachments/assets/bc9a9d1f-f1dd-4295-a662-21b2db5a223a" />


---

## 📈 Results

### 1. Automatic Evaluation

| Dataset            | Accuracy ↑ | PPL ↓    | Dist‑2 ↑ | Dist‑3 ↑ | Toxicity ↓ |
| ------------------ | ---------- | -------- | -------- | -------- | ---------- |
| **ROCStories**     | **90.4**   | **4.04** | **0.74** | **0.43** | **0.12**   |
| **WritingPrompts** | **85.6**   | **3.68** | **0.55** | **0.29** | **0.24**   |
| **LOT (En)**       | **88.2**   | **4.11** | **0.60** | **0.35** | **0.10**   |

### 2. Conflict vs. Overlap (ROCStories)

| Method           | Conflict Avg ↓ | Conflict PPL ↓ | Conflict Drift ↓ | Overlap Avg ↓ | Overlap PPL ↓ | Overlap Drift ↓ |
| ---------------- | -------------- | -------------- | ---------------- | ------------- | ------------- | --------------- |
| **C³TG**         | **0.19**       | **5.02**       | **0.22**         | **0.14**      | **5.57**      | **0.19**        |
| Model Arithmetic | 0.27           | 10.48          | 0.38             | 0.22          | 11.03         | 0.31            |
| Prompt Steering  | 0.24           | 6.71           | 0.30             | 0.18          | 7.33          | 0.27            |

*Complete breakdowns (per‑attribute scores, human‑eval statistics) are provided in the `paper/` folder.*

---


> *“Life is a journey, and each obstacle is an opportunity to flex our courage muscles.”* – *C³TG demo output*
