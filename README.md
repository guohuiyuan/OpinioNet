# OpinioNet: 基于大语言模型的方面-情感-类别联合抽取

OpinioNet 是一个专注于中文电商评论观点挖掘的项目。本项目采用**指令微调 (Instruction Fine-tuning)** 的方式，训练 **Qwen** 系列大型语言模型 (LLM) 来解决“用户观点四元组” (Aspect-Category-Opinion-Sentiment, ACOS) 的抽取任务。

该方案通过将结构化抽取任务转化为一个端到端的生成任务，有效处理了传统方法难以解决的隐式观点和复杂语义问题，取得了优异的性能。

---

## 快速开始

**1. 环境准备**
```bash
# 安装核心依赖
pip install -r requirements.txt
```

**2. 数据准备 (可选)**
项目已提供预处理好的微调数据。如需重新生成，请运行：
```bash
# 将原始 CSV 转换为 LLM 微调所需的 JSONL 格式
python convert_absa_to_jsonl.py
```

**3. 模型部署与配置**
- 使用 vLLM 或其他推理框架部署一个经过微调的 Qwen 模型。
- 修改 `llm_predict.py` 脚本，填入您的服务地址和 API-Key：
  ```python
  VLLM_API_BASE = "http://YOUR_VLLM_ENDPOINT/v1"
  VLLM_API_KEY = "YOUR_API_KEY"
  ```

**4. 运行预测**
```bash
# 对验证集进行预测
python llm_predict.py --mode val

# 对测试集进行预测，生成提交文件
python llm_predict.py --mode test
```
预测结果将保存在 `data/VALID/Result.csv` 或 `data/TEST/Result.csv`。

**5. 运行评估**
```bash
# 评估验证集上的 F1 分数
python llm_eval.py --pred ./data/VALID/Result.csv
```

**6. 结果融合 (可选)**
如果您有多个模型的预测结果，可以使用 `merge.py` 进行投票融合，以提升最终性能。
- 将不同模型的预测结果文件（如 `Result1.csv`, `Result2.csv`）放入 `data/TEST/` 目录。
- 修改 `merge.py` 脚本，使其读取您的预测文件。
- 运行脚本：
  ```bash
  python merge.py
  ```
  最终的融合结果将保存在 `data/TEST/submit.csv`。

---

## 技术方案详解

本方案的核心思想是将 ACOS 任务转化为一个**条件生成任务**。通过指令微调，教会大模型在理解评论后，直接以 JSON 格式生成所有观点四元组。

- **数据构建**: `convert_absa_to_jsonl.py` 脚本将 CSV 数据转换为 ChatML 格式的 `.jsonl` 文件，每一行都是一个包含 `system`, `user`, `assistant` 的对话样本。

- **推理与优化**:
    - `llm_predict.py` 是推理核心脚本，它负责调用 vLLM 服务，并集成了**断点续传**和**自洽性投票 (Self-Consistency)** 等高级功能来保证预测的稳定性和准确性。
    - `llm_eval.py` 用于精确评估生成的四元组，并能输出详细的错误分析报告。

- **结果融合**:
    - `merge.py` 脚本提供了一种简单的投票集成策略。它读取多个不同模型的预测结果，筛选出在多个结果中共同出现的“高置信度”四元组，并结合单一模型的结果作为补充，最终生成一个融合后的提交文件。

---

## 文件结构

```
OpinioNet/
├── README.md                 # 本文档
├── requirements.txt          # Python 依赖
├── data/                     # 数据目录
│   ├── SPLIT/                # 划分好的训练/验证集
│   ├── TEST/                 # 测试集
│   └── TRAIN/                # 原始训练集
├── result/                   # 实验结果与日志
│   ├── qwen3-8b/             # Qwen-8B 模型的实验记录
│   └── submit/               # 最终提交文件
├── convert_absa_to_jsonl.py  # 数据转换脚本
├── llm_predict.py            # LLM 推理脚本
├── llm_eval.py               # LLM 评估脚本
├── merge.py                  # 多模型结果融合脚本
└── ACOS_LLM_Report.md        # 详细技术报告
```

---

## 详细使用指南

1.  **数据转换**:
    ```bash
    python convert_absa_to_jsonl.py --reviews ./data/TRAIN/Train_reviews.csv --labels ./data/TRAIN/Train_labels.csv --output ./data/TRAIN/train.jsonl
    ```
2.  **模型微调**:
    使用上一步生成的 `train.jsonl` 文件，在您选择的平台上对 Qwen 等大模型进行全量微调。

3.  **推理预测**:
    - 部署微调好的模型。
    - 修改 `llm_predict.py` 中的 API 配置。
    - 运行预测：
      ```bash
      # 生成验证集结果
      python llm_predict.py --mode val

      # 生成测试集结果
      python llm_predict.py --mode test
      ```

4.  **性能评估**:
    ```bash
    python llm_eval.py --gt ./data/SPLIT/val_labels.csv --pred ./data/VALID/Result.csv --text_file ./data/SPLIT/val_reviews.csv
    ```

---

## 实验结果

基于 **Qwen-8B** 全量微调并结合自洽性投票的方案，在验证集上取得了 **F1-Score: 0.7983** 的优异成绩，具体提交记录可见 `result/submit/qwen3_8b_full_Result7983.csv`。这证明了 LLM 在复杂信息抽取任务上的巨大潜力。