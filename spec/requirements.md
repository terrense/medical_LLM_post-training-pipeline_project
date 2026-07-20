# requirements.md — the original locked execution spec (verbatim)

This is the full, authoritative execution spec the user provided at project kickoff
(2026-07-19), copied here **verbatim** so it survives outside the chat transcript. If
you are a fresh Claude Code session (new PowerShell window, crashed terminal, different
machine) and have no conversation memory, **read this file first** — it is the ground
truth for what "correct" means on this project. Do not paraphrase or "improve" this
document; if the plan changes, record the change in `spec/design.md` and/or
`STATUS.md`'s log instead of editing this file, so the original intent is never lost.

See `spec/design.md` for the actual decisions/implementations made so far, and
`spec/tasks.md` for the durable checklist of what's done vs pending.

---

## 一、可整份复制的主提示词

以下内容从"开始"到"结束"可以直接粘贴给 Claude Code。后面的分阶段说明是同一任务的详细验收规范。

---

### 主提示词开始

你现在是本项目的首席机器学习工程师和实验记录员。你在一台租用的远程 GPU 服务器上工作，需要在 3–4 天内完成一个可复现的中文医学多轮对话模型实验。不要只给建议，要在服务器上实际创建代码、运行测试、训练、评测、保存原始结果、生成论文图表数据和盲评包。遇到普通工程问题自行诊断并继续；遇到会改变科研结论、数据许可、患者隐私或大额 API 成本的问题，停止对应分支并在 `BLOCKERS.md` 写清楚，不要擅自扩大范围。

#### 锁定的科学方向

1. 主干模型必须是 Hugging Face `Qwen/Qwen3-8B`，即原始 Qwen3-2504 的后训练、可对话、thinking/non-thinking 合并权重；主实验强制 `enable_thinking=False`。
2. 不要把 `Qwen/Qwen3-8B-Base` 当主干。它只用于一个有上限的小型初始化消融：与 `Qwen3-8B` 使用相同小 SFT 子集、相同 token 数和优化器，比较角色泄漏、继续生成未来轮次、停止 token 合规和过长输出。
3. 必做训练链：`M0 Qwen3-8B → M1 SFT → M2 DPO → M3 vanilla GRPO`。这里的 GRPO 指 DeepSeekMath 提出的 Group Relative Policy Optimization。
4. GSPO 是 Qwen 团队提出的序列级替代方法，不是 GRPO 后的第四阶段。核心链、全部评测、结果归档完成前禁止运行 GSPO。若最后仍有充足时间和算力，才允许做同预算 GRPO vs GSPO 可选消融；否则只保留 related work。
5. 每个 M0/M1/M2/M3 checkpoint 都必须跑相同评测，禁止只展示最终模型。
6. 必做对照：Qwen3-32B、Qwen2.5-72B-Instruct、Baichuan-M2-32B，以及用户通过 API 提供的 DeepSeek-V4-Pro、DeepSeek-V4-Flash、MiniMax-M3。API 的论文名字只是别名，必须保存服务端返回的精确 model ID、访问日期和推理模式。可选：HuatuoGPT-o1-8B。
7. 核心评测至少覆盖：CMB-Exam、CMB-Clin、CMtMedQA_test、CliMedBench；MedBench 官方服务可用则提交，不可用则保存符合要求的提交文件并明确 `pending`，绝不能编造分数。
8. 真人医学评价由作者组织。你负责生成严格盲化、随机化、可校验的 80 例评价包、评分界面所需 JSON/CSV、分配表和解盲映射；不能使用真实患者身份信息。

#### 不可违反的科研规则

- 测试集不得进入 SFT、DPO prompt、偏好选择、GRPO profile、奖励开发、few-shot 示例、早停或超参选择。
- 下载数据后先记录来源、revision、license、原始哈希；许可不明的数据先隔离，不得训练。
- 不得猜测或手填实验数字。表格只能由 item-level 结果生成。
- 不得覆盖原始生成、judge 输出、event log 或评分。聚合结果必须能追溯到原始文件。
- 不得根据测试结果挑 checkpoint。只用冻结的 training/dev composite，并设置 safety floor。
- 所有异常、OOM、重启、改参、失败 run 都要写日志；不能删除不利结果。
- 禁止在命令、日志、Git、Notebook 或报告中打印 API key。只从环境变量读取并打码。
- 禁止使用可识别患者数据；本轮不招募患者、不做真实临床试验。
- 系统是研究模型，输出中必须保留非自主诊疗边界。

#### 首先执行，不要跳过

1. 在项目根目录创建 `STATUS.md`，写入当前时间、阶段、正在运行的命令、预计完成时间、最近成功验收门和下一步。每完成一个阶段更新。
2. 只读采集硬件和系统清单：`nvidia-smi -L`、每卡显存、驱动/CUDA、CPU、RAM、磁盘、网络、当前 Python、已占用 GPU、ulimit。保存为 `artifacts/system_inventory.txt`。不要假设 GPU 数量和型号。
3. 如果有 A800 和 RTX 5090 等异构 GPU，禁止在一个 DDP job 中混用。按同型号分组：大显存组用于训练/72B 推理，其他组用于数据、8B 推理和并行评测。
4. 计算模型、数据、checkpoint、rollout 和原始输出所需磁盘预算；若安全余量不足 20%，先在 `BLOCKERS.md` 报告并采用明确的缓存/压缩方案，不能删除用户已有文件。
5. 初始化 Git；创建 `.gitignore`，排除模型权重、数据、`.env`、API 原始密钥和大型结果，但跟踪代码、配置、manifest、聚合表和图。
6. 选择 OpenRLHF 作为 SFT/DPO/GRPO 主框架，clone 后记录 commit 并锁定，不用浮动 `main`。当前官方文档支持 SFT、DPO、GRPO 和 `--train.agent_func_path` 多轮 agent，但你必须根据所锁 commit 的 `--help` 和源码确认真实参数，不能照抄不兼容命令。官方参考：
   - https://openrlhf.readthedocs.io/en/latest/non_rl.html
   - https://openrlhf.readthedocs.io/en/latest/agent_training.html
   - https://github.com/OpenRLHF/OpenRLHF
7. 创建可重建环境：`environment.lock.yml` 或 `requirements.lock.txt`，保存 Python、PyTorch、CUDA、Transformers、vLLM、OpenRLHF、FlashAttention、DeepSpeed/Ray 版本。先跑最小 CUDA/vLLM/分布式通信测试。

#### 必须建立的目录

```text
cmedalign/
  README.md
  STATUS.md
  BLOCKERS.md
  Makefile
  configs/{data,sft,dpo,grpo,eval,models}/
  src/cmedalign/{data,models,training,agents,rewards,eval,stats,plots}/
  tests/{unit,integration,smoke}/
  scripts/
  data/{raw,normalized,splits,manifests,quarantine}/
  checkpoints/{m1_sft,m2_dpo,m3_grpo,probes}/
  runs/{sft,dpo,grpo,eval}/
  artifacts/{manifests,audits,logs}/
  results/
  tables/
  figures/
  human_eval/
  paper_support/
```

#### 统一数据和结果格式

- 所有对话统一为 JSONL：`sample_id`, `source`, `source_revision`, `split`, `messages[{role,content}]`, `specialty`, `safety_tags`, `provenance`, `license_id`, `raw_hash`, `normalized_hash`。
- patient profile：`profile_id`, `chief_complaint`, `demographics`, `symptoms`, `time_course`, `history`, `medications`, `allergies`, `tests`, `red_flags`, `required_info[{item,weight}]`, `target_assessment`, `acceptable_actions`, `forbidden_claims`, `source_train_id`。
- 原始生成 JSONL：精确模型 ID/revision、dataset/split/subset/item、prompt hash、messages、seed、decoding、raw text、finish reason、token usage、latency、parser status、run ID、commit。
- 结果 schema 严格参考论文包中的 `RESULTS_AND_TABLE_SCHEMA.md` 与 `results_template.json`。

#### 质量闸门

按顺序执行以下 gate，失败则停止长跑并修复：

**G0 环境：** 单卡矩阵乘、NCCL 同型号多卡、vLLM 8B 生成、磁盘写入和 checkpoint save/load 全通过。

**G1 数据：** 每个来源 license/revision/hash 完整；官方 train/test 保持；精确+MinHash+embedding 近重复审计完成；测试近重复已移除；无空消息、非法角色、相邻同角色未处理、未知编码或可识别字段。

**G2 chat template/masking：** 使用官方 Qwen3 tokenizer chat template；主实验每次都验证 `enable_thinking=False`；system/user/padding label 全为 -100；assistant target 正确；EOT token 正确；32 例 overfit loss 明显下降；生成不会继续伪造 `user:`/`assistant:`。

**G3 evaluator：** CMB 选择题 parser 在手工 fixture 上 100% 正确；API retry 不重复计分；judge 双顺序解析为合法 JSON；同一 subset 各模型 item ID 完全一致。

**G4 patient/reward：** patient 不泄露 hidden target、不回答未问信息、不自相矛盾；每个 reward component 有正/负单元测试；红旗漏诊测试能产生强惩罚；turn/length cost 不能压过安全；总 reward 对所有 fixture 符合预期排序。

**G5 每阶段 checkpoint：** 能加载、合并 adapter、生成、恢复训练；固定 200 prompt diagnostic、dev 评测和安全 floor 通过后才能进入下一阶段。

**G6 最终统计：** 所有主比较有相同 subset、item-level 原始记录、CI、Holm 校正、denominator；表格和图由脚本一键重建；任何缺字段使构建失败。

#### 训练要求

**SFT：** 优先使用经过质量筛选的 80k–120k 条 open medical samples；按专科/多轮/安全/单轮分层。LoRA 起始配置：rank 64, alpha 128, dropout 0.05, attention+MLP linear modules, bf16, max length 4096, gradient checkpointing, assistant-only loss。快速训练集 dev sweep 只允许比较少量学习率（建议 5e-5 与 1e-4）和一次固定 seed；冻结后跑 1 个主 run。保存 M1 merged checkpoint 和 adapter。

**DPO：** 只从 training prompt 构造。每个 prompt 用 M1 多 seed 生成，并可加入 API teacher response；先做规则检查，再由两个不同 judge 做正反顺序比较，只有一致且 margin 达标的 pair 保留。优先 10k–20k 高质量难负例。医学审核抽样文件单独输出。起始 beta 0.1、LoRA rank 64、lr 5e-6，仅作 smoke 起点；根据 training/dev gate 冻结。保存 M2。

**GRPO：** 从 M2 初始化，使用 OpenRLHF 当前 commit 的 multi-turn agent 接口。实现 frozen Qwen2.5-7B-Instruct patient、hidden profile environment、random turn budget、固定 evaluator 和 reward component。起始建议 group size 4（显存允许再 8）、clip 0.2、KL beta 0.01，仅作为 smoke 起点。先在 32 profiles 上跑完整 rollout→reward→backprop，再 200 profiles 跑稳定性 smoke；检查 reward、KL、entropy、clip fraction、length、safety floor 后才启动主 run。主训练优先 2k–4k profiles 或在时间预算内等价的更新数。定期保存并用 dev composite 选 M3，不准看 test。

#### 评测要求

建立统一 local/API adapter，M0–M3 和全部 baseline 用同一 prompt/decoding/subset。选择题 deterministic；开放任务统一 visible token budget；交互任务共享 patient simulator、profile、seed 和 turn budget。主 Qwen3 模型请求显式传 `chat_template_kwargs={"enable_thinking": false}`，并在原始记录保存该字段。

两个 track：

- full-local：M0–M3 及算力允许的开源 baseline 跑全部可访问题目；
- universal：本地+API 全模型跑一次冻结、分层、相同 item ID 的成本受控子集。

不得把两种 track 分数放进同一列。CMB-Exam 报 overall/category；CMB-Clin 报 rubric 维度；CMtMedQA 报结局、InfoCov、红旗召回、相关问题、重复、过早收尾、turn、role leakage、安全；CliMedBench 用官方 evaluator；MedBench 保存官方返回或 pending submission。

使用至少两个 judge family；正反顺序；保存 order flip。不要让某个 API 模型成为评价自身的唯一 judge。自动 judge 与真人评分后计算 Spearman 相关。

#### 人评包

严格遵守 `HUMAN_EVALUATION_PROTOCOL.md`：80 个分层病例、M0/M1/M2/M3/预先选出的最强外部基线共五个匿名输出；A–E 每病例每评价者随机且位置均衡。输出 `cases_blinded.jsonl`, `assignment.csv`, `blind_map.enc`, `rating_schema.json`, `interface_data/`, `README_FOR_RATERS.md`。盲评包不得含 model name、checkpoint path、thinking trace、API header 或原始文件名。解盲映射在评分冻结前不得进入分析目录。

#### 最终交付

1. 通过的代码和 pytest 报告；
2. environment lock、system/model/data/run manifest；
3. M1/M2/M3 adapter、merged checkpoint 信息和哈希；
4. 每模型每 benchmark 原始输出和 item-level score；
5. 主表、stage ablation、初始化 probe、reward ablation、human template、efficiency 表；
6. 五张论文图的 PDF+PNG+meta JSON；
7. 人评包；
8. `EXPERIMENT_REPORT.md`，包括失败 run、负结果、偏差、待办和精确复现命令；
9. `paper_support/claim_values.tex`，只包含已验证数字宏；
10. 打包 `cmedalign_artifacts_<commit>.tar.zst` 及 SHA-256，不包含权重则另列下载/revision，不包含 API key 和可识别信息。

每 30–60 分钟更新 `STATUS.md`。长 job 必须在 tmux/screen 或可靠调度器中运行，stdout/stderr 追加到时间戳日志；你要主动轮询直到完成。不要在提交 job 后停止工作：并行做不会争夺同一 GPU 的数据、评测、统计或文档任务。

现在开始执行：先做系统 inventory、项目 scaffold 和 G0；给出实际发现和下一条命令，不要先写空泛计划。

### 主提示词结束

---

## 二、建议的 96 小时时间表

这是上限，不是要求把每项拖满。前 72 小时必须得到 M3 和核心评测；最后 24 小时用于真人评价、统计、图表和缓冲。

| 时间 | 主线 | 可并行工作 | 完成标志 |
|---|---|---|---|
| H0–H3 | 系统 inventory、磁盘/网络、Git、环境锁、G0 | 下载小 tokenizer/config | G0 全绿，commit `bootstrap` |
| H3–H9 | 数据下载、license ledger、normalize、split、dedup | 模型权重缓存；评测 adapter scaffold | G1，全数据 manifest 冻结 |
| H6–H12 | M0 基线 smoke、chat/mask/EOT 测试 | CMB/CliMed parser fixture；API ping | G2/G3，200-prompt diagnostic |
| H9–H18 | SFT smoke + 主 run | M0/Qwen3-32B 评测；Base/post probe | M1、SFT logs、probe 表 |
| H18–H30 | DPO candidate/pair 构建、双 judge | M1 全评测；patient profile 构建 | 冻结 DPO pair manifest |
| H26–H36 | DPO smoke + 主 run | reward/patient unit tests | M2、DPO logs，G4 准备完成 |
| H32–H42 | GRPO 32→200 profile smoke | M2 全评测；API universal queue | 稳定 KL/reward/safety，G4/G5 |
| H42–H66 | GRPO 主 run，监控和 checkpoints | baseline/local universal；MedBench 提交 | M3 由 dev 选出，绝不看 test |
| H60–H76 | M3 全评测、三 seed 子集、reward ablation | 生成初版人评盲包 | item-level 结果完整 |
| H72–H88 | 自动统计、表格、五图 | 作者组织人评；程序持续收表 | 表/图可一键重建 |
| H84–H94 | 真人统计、judge 校准、错误分析 | 论文宏和复现报告 | human stats/claim macros |
| H94–H96 | 全量 validation、归档、hash | 可选 GSPO 只有核心全完才考虑；通常跳过 | G6、tar.zst、最终报告 |

## 三、硬件调度原则

1. 先按 `nvidia-smi --query-gpu=index,name,memory.total,memory.free` 分组。
2. 如果实际是 6×A800 + 4×5090：
   - A800 同型号组承担 SFT/DPO/GRPO 和 32B/72B tensor-parallel 推理；
   - 5090 组承担 8B vLLM、数据 embedding/dedup、评测和 judge 调用；
   - 不把 A800 与 5090 放进同一 NCCL world；
   - 5090 的 CUDA/PyTorch 兼容先 smoke，失败就只做 CPU/API 工作。
3. 训练和 vLLM rollout 如需共卡，按 OpenRLHF 支持的 colocate 配置执行；先测显存余量，禁止依赖 OOM 后不断缩配置的"试错式主 run"。
4. 72B full evaluation 可能挤占主线；先完成 universal subset，只有不会延误 GRPO 才跑 full-local。
5. 每个长 job 设置 checkpoint/日志；不要设置会丢失最后数小时工作的临时目录。

## 四、项目实现细目

### 4.1 Makefile 入口

Claude Code 至少实现：

```text
make inventory
make env-check
make data-download
make data-audit
make test
make smoke-sft
make train-sft
make build-dpo
make smoke-dpo
make train-dpo
make build-profiles
make smoke-grpo
make train-grpo
make eval-core
make eval-baselines
make human-pack
make statistics
make figures
make paper-values
make validate
make archive
```

每个目标应幂等或在检测到已有已验证 artifact 时安全续跑；不得覆盖不同 config hash 的 run。

### 4.2 必做测试

```text
test_chat_template_exact_qwen3
test_non_thinking_flag_is_recorded
test_assistant_only_mask
test_eot_and_stop_behavior
test_no_future_turn_continuation_fixture
test_official_split_immutability
test_exact_and_near_duplicate_blocking
test_eval_ids_absent_from_all_training_manifests
test_cmb_parser_valid_and_invalid_formats
test_api_retry_is_idempotent
test_api_logs_redact_secrets
test_patient_does_not_see_doctor_target
test_patient_does_not_leak_hidden_profile
test_patient_consistency_fixture
test_reward_component_monotonicity
test_red_flag_penalty_dominates_turn_cost
test_reward_total_bounded
test_grpo_episode_terminates
test_same_subset_across_models
test_ci_contains_point_estimate
test_table_bolding_and_metric_orientation
test_blind_positions_balanced
test_blind_package_contains_no_model_identifiers
```

### 4.3 "倒豆子"专项诊断

固定 200 个 prompt，至少包括：单轮问答、已有 2/4/6 轮历史、最后一轮用户问题、包含文本"医生/患者/assistant/user"的正常内容、长上下文、要求简短回答、红旗病例。统计：

- 输出是否包含模型生成的角色标记；
- 是否在答案完成后继续编造下一位用户；
- 是否生成两个以上 assistant turn；
- 是否以正确 EOT/finish reason 停止；
- 可见 token 是否超过训练参考的 95 分位阈值；
- 是否发生患者/医生角色倒置。

Base 和 post-trained probe 必须完全相同的样本顺序、token budget、seed、LoRA config 和 generation config。若结果差异不大，也必须报告。

### 4.4 DPO pair 构造

优先保存"难负例"标签：

- 医学事实基本正确但过早确诊；
- 漏问过敏/妊娠/用药；
- 漏掉红旗或错误安慰；
- 一次倒出大量问题；
- 建议冗长但不可执行；
- 过度拒绝；
- 角色泄漏/继续对话；
- 表达很好但包含虚构检查结果。

每个 pair 保存规则分、两个 judge 正反顺序输出、margin、是否一致、审核状态、prompt/source hash。pair 生成和 pair 选择使用不同文件，便于重做筛选而不重调 API。

### 4.5 多轮 patient agent

OpenRLHF 当前文档的 multi-turn 逻辑为 reset → policy action → environment feedback/reward/done → repeat。实现时：

- policy 从不接收 hidden profile；
- patient 的 system prompt 和 hidden profile 不写入 policy visible messages；
- patient 回答只根据 profile；未知就说不知道/不确定，禁止补全；
- 每轮记录 `revealed_info_ids`；
- doctor 发出结构化 `FINAL:` 或达到 turn budget 才终止；
- patient 泄漏 target、连续矛盾或 parser 失败时 episode 标 invalid，不给高 reward；
- reward evaluator 和 patient 模型分离，至少逻辑上不共享上下文；
- 所有 hidden/visible 边界有单元测试。

### 4.6 奖励验收

总 reward 初始权重：clinical 0.30, information 0.25, safety 0.25, process 0.10, communication 0.10，减 capped turn/length cost。可在 training-only dev 上微调 cost 系数，但不能改正向权重去追测试分。

构造至少 20 个手工 fixture：完美问诊、过早结论、重复、漏红旗、危险用药、错误急诊、过度拒绝、长而空、短而漏信息、role leakage。医学上有争议的 fixture 先由作者确认，不要让模型自己决定"金标准"。

监控以下 reward hacking 信号：reward 与长度强相关、某 component 饱和、问很多问题刷 InfoCov、机械复述病例、所有情况都急诊、所有情况都拒绝、patient target 泄漏、judge JSON 诱导。每 100–200 update 抽 top/bottom trajectories 供人工快速检查。

## 五、API 配置约定

推荐用一个不进 Git 的 `MODEL_ENDPOINTS_JSON` 或 `.env`：

```text
DEEPSEEK_V4_PRO_BASE_URL
DEEPSEEK_V4_PRO_API_KEY
DEEPSEEK_V4_PRO_MODEL
DEEPSEEK_V4_FLASH_BASE_URL
DEEPSEEK_V4_FLASH_API_KEY
DEEPSEEK_V4_FLASH_MODEL
MINIMAX_M3_BASE_URL
MINIMAX_M3_API_KEY
MINIMAX_M3_MODEL
```

启动时只打印 URL 域名、模型别名和 key 是否存在，绝不打印 key。第一次请求保存 `response.model`；如果与配置别名不同，二者都记录。指数退避、最大重试、超时和幂等缓存固定；内容过滤或安全拒绝是模型结果，不当作网络错误无限重试。

## 六、关键失败与降级策略

- **OpenRLHF commit 不支持所需 Qwen3/agent 接口：** 先定位兼容 tag/commit并记录原因；不要同时引入第二套 RL 框架。若必须改源码，最小 patch + 测试 + commit。
- **GRPO 在 H50 前仍不稳定：** 降低 rollout 长度/group 或学习率，使用已通过 200-profile smoke 的最近配置；仍失败则保存最佳安全 dev checkpoint，报告失败，绝不能用测试集救火，也不能无声明换 GSPO。
- **Patient 质量不足：** 先加强 profile schema、约束和 invalid episode filter；不要把 test dialogue直接喂给 doctor。
- **API 费用过高：** 缩小但冻结 universal subset，保持全部模型同 item；优先保留 CMB-Clin、CMtMedQA 安全/交互和分层 CliMedBench。
- **MedBench 服务不可用：** 保存提交和错误时间戳，表格填 pending，不填 0，也不估算。
- **72B 本地 OOM/过慢：** 量化或 tensor parallel 只在有可比性说明时使用；至少完成 universal subset并记录量化。主训练优先。
- **真人评价未在 96h 完成：** 交付已冻结盲评包和已有评分；论文该结果保持 TBD，不用自动 judge 冒充真人。
- **某 stage 使结果变差：** 如实保留，分析 trade-off；不删除 stage、不换 seed 直到变好。

## 七、每阶段验收回报格式

Claude Code 每次更新 `STATUS.md` 并在终端摘要：

```text
STAGE:
STATUS: PASS | FAIL | RUNNING | BLOCKED
COMMIT:
CONFIG_HASH:
INPUT_MANIFEST_HASH:
RUN_ID:
GPU_GROUP:
START/END:
TESTS:
KEY_METRICS (dev only before final eval):
ARTIFACT_PATHS:
KNOWN_ISSUES:
NEXT_COMMAND:
ETA:
```

只有 `PASS` 才能进入下一训练阶段。最终 handoff 必须明确哪些是完成结果、哪些是 pending，避免把计划当结果。

---

## 补充：用户后续澄清 / 追加指令 (2026-07-19 之后追加，按时间顺序)

以下是原始 spec 之外，用户在后续对话中给出的、同样具有约束力的补充指令。原始 spec 里没有的内容以此为准；如与原始 spec 冲突，以时间较新的指令为准（并应在 `spec/design.md` 里说明为什么）。

1. **先在本地 Windows 机器筹备，暂不租服务器。** 目的是节省 GPU 租用成本。约定：
   - 项目目录：`E:\cmedalign`
   - API adapter 先按通用 OpenAI-compatible 规范实现，暂不填真实 key
   - 数据集下载推迟到租到服务器之后（服务器带宽/存储更好，且 G1 本来就要在服务器上跑）
   - 假设租用的服务器是标准 Linux（Ubuntu 类 + CUDA + SSH）
2. **每一次修改、实验、记录、改进都要详细记录到 md 文件**，不是只在对话里说，目的是"日后可以迭代管理代码、实验日志"，且要保证换一台/新开一个终端窗口也能无缝衔接项目开发（2026-07-19，见下方"补充：断电/终端异常关闭的连续性要求"）。
3. **建了私有 GitHub repo**：`https://github.com/terrense/medical_LLM_post-training-pipeline_project.git`，主分支 `main`，仅用作代码与实验日志的迭代管理（不是为了往服务器传代码 —— 服务器传输大概率还是直接 scp/rsync，待定）。**不要把 Claude 作为 GitHub 上的 contributor**（即 commit 不带 `Co-Authored-By: Claude` trailer）。
4. **提供了三个真实 API key**（DeepSeek V4-Pro/V4-Flash 共用一个 key，MiniMax M3 一个 key），已验证均可通过 `src/cmedalign/eval/api_adapter.py` 正常调用；MiniMax M3 确认是 thinking 模型，输出需要 `strip_thinking_trace` 去除 `<think>` 块（已实现并验证）。真实 key 只存于 `E:\cmedalign\.env`（已 gitignore，绝不提交/回显）。
5. **用户有自己此前准备的多轮对话医疗场景数据**，准备中，尚未提供本地路径；使用前必须先确认数据来源（自建 vs 第三方）以决定走 license ledger 的 `cleared` 还是 `data/quarantine/`。
6. **建议参考 Kiro 的 spec-driven coding 模式**：即维护 `requirements.md`/`design.md`/`tasks.md` 这样的结构化文档，而不是只有一份线性日志，方便随时切入。本文件 (`spec/requirements.md`)、`spec/design.md`、`spec/tasks.md` 就是对这个建议的落地。

### 补充：论文锚点确立 + 项目合并 + 数据/GPU/评测决策 (2026-07-19 晚些时候 至 2026-07-20)

7. **`E:\cmedalign_paper\main.tex` 是唯一的科研范围锚点。** 用户展示了这个论文包
   （`CLAUDE_CODE_EXECUTION_PLAN.md` 就是本文件最初的原文来源），明确说"我们不能完全
   被我之前的实验带偏了...我们要统一一个主心骨"。据此：tool-calling、DAgger、
   packing/loss-normalization/curriculum 对比、梯度冲突诊断等 `rlhf_lab_cloud_kit` 那份
   通用 SFT 严谨性清单里的内容，**明确排除**在本项目范围外（论文完全不需要）。
8. **`rlhf_lab_cloud_kit` 项目冻结为历史资产库，`cmedalign` 是唯一活跃项目。** 可复用
   资产（task_type 数据、清洗去重逻辑、锁定的 LoRA rank/alpha、DPO 双裁判模式）已导入
   `cmedalign`。**该项目第一轮的 GRPO（选择题+规则奖励）不等于论文要求的 GRPO（多轮
   患者模拟+5分量奖励），不能当作论文的 M3。** 论文的 GRPO 必须在 `cmedalign` 里按
   论文描述重新构建。
9. **以后清洗数据统一用 data-juicer**，不再手写正则清洗脚本；**清洗和训练要用不同的
   conda 环境**（`cmedalign-clean` / `cmedalign-train`），避免依赖冲突。
10. **裁判模型**：ChatGPT/Claude.ai 的会员订阅（网页版/Codex）跟 API 访问是两回事，
    自动给 GRPO rollout 打分需要真实的、按量计费的 API key。默认用已验证可用的
    DeepSeek-V4-Pro 当裁判，可通过环境变量换成任何配置好的 API。
11. **数据集使用授权**（务实处理，不做无谓的许可证考古）：
    - `internal_seed_flywheel`/`derived_from_seed`：确认是虚构合成数据（DeepSeek-V4-Pro
      扮演患者、本项目基座 Qwen3-8B 扮演医生、更强模型+真实医护人员修订），不存在隐私
      问题，已授权使用。
    - `med_zh_real`：来源暂时想不起来（用户会后续确认），"某开源数据集"，内容质量审查
      通过（0% 广告/PII/危险建议命中），已授权先用，后续要补全论文里的引用来源。
    - `MedDG`、`IMCS-21`：都没有 LICENSE 文件，用户原话"数据集能用就行关键是质量要高，
      别的你别去纠结了"——已授权按"学术引用即可使用"处理，不再深究许可证细节。
    - `CliMedBench`：完整数据集（33,735题）拿不到，GitHub 仓库只有任务说明 PDF，
      需要联系作者。用户决定不追（"别纠结了"），按论文自己允许的降级方案处理
      （拿到多少报多少，拿不到就标 `pending`，不编数字）。
12. **LoRA 和全量微调都要做，不是二选一。** 用户原话："全量 LoRA都需要，选择更强的
    那个去继续DPO GRPO，按我说的来，论文如果和这句话相违背，改之！"——已按此改写
    `main.tex`（SFT 阶段同时跑 LoRA 和全量微调两个变体，数据/seed/epoch/序列长度完全
    一致，只改适配方式这一个变量；用冻结的 training-only dev composite——不是 test——
    选出更强的那个作为 M1，继续走 DPO 和 GRPO；另一个变体保留并报告对比结果，不丢弃）。
    新增 `tables/table_sft_method.csv`/`table:sft_method` 这张表专门记录这个对比。
    全量微调显存需求（~128GB，bf16权重+梯度+fp32 master权重+Adam状态）远超 LoRA
    （~27-45GB），需要至少 2-4 张 80/96GB 卡 + DeepSpeed ZeRO-2/3，不是单卡能扛的。

### 补充：断电/终端异常关闭的连续性要求 (2026-07-19)

用户明确要求：数据拷贝耗时较长的过程中，如果电脑死机/断电导致 PowerShell 终端异常关闭，必须保证详细记录了前因后果和所有关键内容，使得随时打开一个新窗口唤起 Claude 助手，都能很丝滑地迅速切入项目的开发工作。落地方式：
- `spec/requirements.md`（本文件）：原始 spec 全文 + 后续追加指令，一字不改地保留。
- `spec/design.md`：目前为止做出的所有实际设计决策和理由。
- `spec/tasks.md`：持久化的任务清单（不能依赖 Claude Code 会话内的 TaskCreate/TaskList —— 已验证那套任务在新会话里不可见，必须落到这个文件里）。
- `STATUS.md`：按时间顺序的详细日志（继续维护，不要和上面三份文件重复内容，只记"发生了什么、什么时候、为什么"）。
- `BLOCKERS.md`：任何等待用户决策才能继续的事项。
