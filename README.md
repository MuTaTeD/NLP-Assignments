# Natural Language Processing (CSAI8333) — Course Assignments

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Unsloth%20%7C%20HuggingFace-orange.svg)](https://github.com/unslothai/unsloth)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()

This repository contains academic coursework, experiments, and executed notebooks for **Special Topics in Natural Language Processing (CSAI8333)**, PhD in Computer Science, Fall 2026 at the **University of Central Punjab (UCP)**.

---

## 👥 Project Collaborators

* **Awais M. Lodhi** (ID: `L1F26PHDC0006`)
* **Sajid Saleem** (ID: `L1F26PHDC0001`)
* **Basit Ali Gillani** (ID: `L1F26PHDC0002`)

*Program:* PhD in Computer Science (Fall 2026)  
*Institution:* Faculty of Information Technology, University of Central Punjab (UCP), Lahore, Pakistan  

---

## 📂 Repository Structure

```
NLP-Assignments/
├── README.md
├── .gitignore
└── LLM Finetuning/
    ├── Gemma_2_2B_IT_Banking_Classification - (Awais Lodhi).ipynb
    ├── Gemma_2_2B_IT_Banking_Classification - (Sajid Saleem).ipynb
    ├── Llama_3_2_3B_Banking_Classification - (Awais Lodhi).ipynb
    ├── Mistral_7B_Banking_Classification - (Awais Lodhi).ipynb
    ├── Phi_3_5_mini_Banking_Classification - (Basit Gillani).ipynb
    ├── Qwen2_5_3B_Banking_Classification - (Basit Gillani).ipynb
    └── TinyLlama_1_1B_Banking_Classification - (Sajid Saleem).ipynb
```

---

## 🚀 Assignment #1: Parameter-Efficient Fine-Tuning (QLoRA) Across 6 LLMs/SLMs for Banking Intent Classification

### Overview
This project evaluates the performance of six modern open-weight Instruction-tuned Small and Large Language Models (1.1B to 7.2B parameters) for fine-grained intent classification using the standardized **Banking77** benchmark (77 categories, 13,083 queries).

Fine-tuning is conducted via **Quantized Low-Rank Adaptation (QLoRA)** using the accelerated **Unsloth** library (4-bit NF4 quantization, rank $r=16$, $\alpha=16$). Models are benchmarked pre- and post-fine-tuning across three prompting formulations:
1. **Baseline (Zero-Shot):** Direct intent extraction constrained strictly to canonical taxonomy.
2. **Few-Shot Exemplars:** In-context demonstrations providing representative query-to-label pairs.
3. **Feature-Aware:** Two-stage entity (card, pin, fee) and state (declined, pending) extraction.

### Executed Notebooks

All notebooks are pre-executed with complete execution outputs, loss logs, pre/post evaluation metrics on 2,080 test queries, and Gradio test interfaces:

| Model Architecture | Parameters | Contributor | Notebook Link | Post-Tuning Accuracy |
| :--- | :---: | :---: | :--- | :---: |
| **TinyLlama 1.1B Chat** | 1.1B | Sajid Saleem | [Open Notebook](./LLM%20Finetuning/TinyLlama_1_1B_Banking_Classification%20-%20(Sajid%20Saleem).ipynb) | **90.38%** |
| **Gemma 2 2B IT** *(Primary)* | 2.6B | Awais M. Lodhi | [Open Notebook](./LLM%20Finetuning/Gemma_2_2B_IT_Banking_Classification%20-%20(Awais%20Lodhi).ipynb) | **93.22%** |
| **Gemma 2 2B IT** *(Validation)* | 2.6B | Sajid Saleem | [Open Notebook](./LLM%20Finetuning/Gemma_2_2B_IT_Banking_Classification%20-%20(Sajid%20Saleem).ipynb) | **92.60%** |
| **Llama 3.2 3B Instruct** | 3.2B | Awais M. Lodhi | [Open Notebook](./LLM%20Finetuning/Llama_3_2_3B_Banking_Classification%20-%20(Awais%20Lodhi).ipynb) | **92.88%** |
| **Qwen2.5-3B-Instruct** | 3.1B | Basit Ali Gillani | [Open Notebook](./LLM%20Finetuning/Qwen2_5_3B_Banking_Classification%20-%20(Basit%20Gillani).ipynb) | **93.27%** |
| **Phi-3.5-mini-Instruct** | 3.8B | Basit Ali Gillani | [Open Notebook](./LLM%20Finetuning/Phi_3_5_mini_Banking_Classification%20-%20(Basit%20Gillani).ipynb) | **91.83%** |
| **Mistral 7B Instruct v0.3** | 7.2B | Awais M. Lodhi | [Open Notebook](./LLM%20Finetuning/Mistral_7B_Banking_Classification%20-%20(Awais%20Lodhi).ipynb) | **93.32%** |

---

## ⚡ Empirical Hardware & Resource Profiles (Nvidia Tesla T4)

* **Pre-Training Evaluation:** ~8–9 minutes per model across 2,080 test queries (3 prompting strategies).
* **Supervised QLoRA Fine-Tuning:** 1.5 to 3.0 hours per model on 10,003 training queries (2 epochs, batch size 8).
* **Post-Fine-Tuning Evaluation:** ~9–10 minutes per model.
* **VRAM Footprint:**
  * **TinyLlama 1.1B:** ~3.5 GB VRAM (edge-deployable)
  * **SLMs (Gemma 2, Llama 3.2, Qwen2.5, Phi-3.5):** ~6.0–7.0 GB VRAM
  * **Mistral 7B Instruct:** ~13.0 GB VRAM

---

## 🔧 Environment Setup & Execution

Each notebook is self-contained and configured for zero-setup execution on **Google Colab** or **Kaggle** with a single Nvidia T4 (15/16 GB) GPU:

```bash
# Core Dependencies
pip install unsloth "xformers<0.0.28" "trl<0.9.0" peft accelerate bitsandbytes datasets evaluate scikit-learn gradio
```

1. Open any notebook in Google Colab or Kaggle.
2. Select **GPU runtime** (T4 GPU recommended).
3. Insert your Hugging Face authentication token when prompted for gated models (Gemma 2, Llama 3.2).
4. Run all cells sequentially.
