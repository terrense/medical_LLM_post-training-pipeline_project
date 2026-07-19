# 环境规划：为什么要分开、分几个、什么时候建

用户要求（2026-07-19）：清洗数据和训练可能需要不同的 conda 环境，尤其是上了服务器之后。
原因很实际：data-juicer 的依赖链（pandas/numpy 版本、部分 filter 用到的轻量模型）经常和
OpenRLHF/vLLM/DeepSpeed 那套要求的 torch/CUDA 版本冲突；混在一个环境里装，任何一边升级
都可能把另一边装崩。这里先把规划写清楚，实际创建环境等到了服务器/GPU 机器上再做
（这台 Windows 本机没有需要装这些重依赖的理由）。

## 三个环境

1. **`.venv`（本机 Windows 开发环境，已存在）**
   纯 CPU，只有 pytest/pydantic/transformers(仅 tokenizer)/cryptography/datasketch 等
   轻量依赖，见 `requirements.lock.txt` Section 1。用于写代码、跑单元测试、验证 chat
   template/mask 逻辑。**不装 data-juicer，也不装 torch。**

2. **`cmedalign-clean`（服务器/GPU机器上，数据清洗专用）**
   装 data-juicer 及其依赖，跑 `configs/data/data_juicer_sft.yaml`。这个环境不需要
   torch/CUDA（data-juicer 的大部分算子是 CPU 的；如果某个 filter 确实需要模型
   打分，等实际遇到再决定是否需要 GPU，不预先假设）。保持这个环境体积小、独立，
   升级/重装不会影响训练环境。

3. **`cmedalign-train`（服务器/GPU机器上，训练+推理专用）**
   装 torch/vLLM/DeepSpeed/Ray/FlashAttention/OpenRLHF，见 `requirements.lock.txt`
   Section 2（目前是未验证的草案，G0 阶段要在这个环境里实际装一遍并重新锁定版本号）。

## Makefile 里怎么体现

`make data-clean` 之类跑数据清洗的目标，应该在 `cmedalign-clean` 环境下执行；
`make train-sft`/`train-dpo`/`train-grpo`/`smoke-*` 之类跑真实训练的目标，应该在
`cmedalign-train` 环境下执行。这台机器没有 conda 环境需要创建，所以 Makefile 里
先按"假设两个环境已经存在，用 `conda run -n <env>` 调用"的方式写好，真正建环境是
G0 阶段服务器上的工作（见 `spec/tasks.md` Phase 1）。

## 什么时候不需要分开

评测（`eval-core`/`eval-baselines`）、统计（`statistics`）、画图（`figures`）、
论文数值（`paper-values`）、人评打包（`human-pack`）这些步骤本身不跑 data-juicer
也不跑训练框架，用 `.venv`（或未来一个类似的轻量评测环境）即可，不需要
`cmedalign-clean` 或 `cmedalign-train` 里的重依赖。如果评测需要本地 vLLM 推理
（跑 M0-M3 或本地开源 baseline），那部分推理调用应该在 `cmedalign-train` 环境
（已经有 vLLM），评测脚本本身的统计/解析逻辑仍在轻量环境里跑。
