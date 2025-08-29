# Transformer from Scratch (学习版)

本仓库内容基于 [Wayland Zhang](https://github.com/waylandzhang) 老师的开源仓库和课程视频整理学习。  
原始实现请参考：[nanoGPT](https://github.com/karpathy/nanoGPT) 和 [张老师的仓库](https://github.com/waylandzhang/transformer-from-scratch)。  

> ⚠️ 本仓库仅作个人学习与课程跟练使用，不涉及原创贡献，如需使用或引用，请以原仓库为准。

---

## 项目简介

这是一个 **Transformer 架构的 Large Language Model (LLM)** 训练 Demo，仅使用 _约 240 行代码_。  

通过该 Demo，可以从零开始理解如何用 PyTorch 训练一个简单的 LLM。  
代码简洁易懂，适合作为入门学习材料。

- 训练数据：约 450 KB [sample textbook](https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt)  
- 模型大小：约 51M  
- 参数量：约 1.3M  
- 硬件：单台 i7 CPU  
- 训练时间：约 20 分钟

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy requests torch tiktoken

### 2. 运行模型训练

```bash
python model.py

- 第一次运行会自动下载数据集并保存到 data 文件夹。

- 模型将在数据集上开始训练，并在控制台输出训练与验证 Loss，例如：

Step: 0 Training Loss: 11.68 Validation Loss: 11.681
Step: 20 Training Loss: 10.322 Validation Loss: 10.287
Step: 40 Training Loss: 8.689 Validation Loss: 8.783
...

- 5000 次迭代后，Loss 会下降到约 2.807，模型会保存为 model-ckpt.pt。
- 训练完成后，会在控制台输出模型生成文本示例，例如：
The salesperson to identify the other cost savings interaction towards a nextProps audience, ...
提示：可以修改 model.py 顶部的超参数，观察训练效果的变化。

### Step-by-Step Notebook

本仓库提供 step-by-step.ipynb，逐步展示 Transformer 的计算过程。
运行前需要安装额外依赖：

```bash
pip install matplotlib pandas

Notebook 中包含：

- [x] 输入嵌入矩阵示例  
- [x] 位置编码可视化  
- [x] 注意力矩阵与 Mask 操作可视化  

👉 帮助理解 Transformer **Decoder-only 架构** 的训练流程。


注意力矩阵与 Mask 操作可视化 帮助理解 Transformer Decoder-only 架构 的训练流程。 
其它内容 在 /GPT2 目录下，有一些示例代码演示如何微调预训练 GPT2 模型并进行推理。

## 推荐阅读

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy 的经典 GPT 教程  
- [Transformers from Scratch](https://blog.matdmiller.com/posts/2023-06-10_transformers/notebook.html) — Mat Miller 的简洁实现  
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文  
- [Transformer Architecture: LLM From Zero-to-Hero](https://medium.com/@waylandzhang/transformer-architecture-llms-zero-to-hero-98b1ee51a838) — 张老师的讲解文章  

