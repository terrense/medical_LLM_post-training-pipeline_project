> This file is the chronological detail log. For a fast cold-start (new window, crashed
> terminal, no memory of this conversation), read `spec/requirements.md` →
> `spec/design.md` → `spec/tasks.md` first — this file is the "what happened when"
> supplement, not the primary entry point anymore.

STAGE: 本地准备阶段（尚未租服务器）——数据已导入并核实（rlhf_lab_cloud_kit 部分来源 +
  CMtMedQA 官方 train/test），GRPO agent 骨架已写，data-juicer 清洗配置已写，环境
  分离方案已写；仍缺 MedDG/IMCS-21、结果输出格式对齐、完整污染审计跑完
STATUS: PASS（本地能做的部分；G0 仍需要真实 GPU 才能跑）
COMMIT: see `git log -1` after first commit
CONFIG_HASH: n/a (no training run yet)
INPUT_MANIFEST_HASH: n/a (no real data downloaded yet)
RUN_ID: local-prep-2026-07-19
GPU_GROUP: none (Windows prep machine, no GPU work performed)
START/END: 2026-07-19 / 2026-07-19 (local prep session)
TESTS: pytest tests/unit -q -> 96 passed, 0 failed, 0 skipped
KEY_METRICS (dev only before final eval): n/a
ARTIFACT_PATHS:
  - artifacts/audits/openrlhf_commit_notes.md — OpenRLHF v0.10.4 (ad1796e) pinned, real
    argparse-verified CLI flags for SFT/DPO/GRPO/multi-turn-agent, plus three verified
    Qwen3 chat-template footguns (see below)
  - vendor/OpenRLHF — pinned checkout at tag v0.10.4 (gitignored, re-clone on server)
  - src/cmedalign/{data,eval,rewards,stats,agents}/* — implemented + unit-tested
  - tests/unit/* — 96 tests, all passing locally
  - requirements.lock.txt — Section 1 (CPU-only) verified by actually installing +
    running the full suite; Section 2 (GPU-only: torch/vllm/deepspeed/ray/flash-attn)
    is an unverified best-effort draft, explicitly marked TBD pending G0 on the server
  - configs/{data,sft,dpo,grpo,eval,models}/*.yaml — skeletons with the spec's starting
    hyperparameters and real (grep-verified) OpenRLHF flag names where applicable
KNOWN_ISSUES:
  - No GPU available locally; G0 (hardware inventory + CUDA/vLLM/NCCL smoke) cannot run
    until the rented server is available. Everything built so far was deliberately scoped
    to be GPU-independent.
  - requirements.lock.txt Section 2 (GPU stack) is unverified — G0 must confirm it
    actually installs on the rented hardware before anything else runs.
  - No real datasets downloaded yet (deliberate — user decision: wait for server,
    better bandwidth/storage there and G1 needs to run there anyway).
  - No real API credentials configured yet (deliberate — user decision: build against
    generic OpenAI-compatible spec now, wire real DeepSeek/MiniMax keys later).
  - OpenRLHF's own assistant-only masking mechanism (in its SFT trainer/collator) not
    yet read in full — deferred to the server session (fast to check there); our own
    chat_template.py masking logic is independently correct and tested regardless of
    what OpenRLHF's default collator does, so this doesn't block anything.
  - The `agent_func_path` implementation for the GRPO patient-doctor environment is not
    yet written (needs `examples/python/agent_func_openai_server_executor.py` from the
    vendored OpenRLHF as a reference, and a real LLM to play "patient" — both deferred
    to the server). The boundary logic it must respect (hidden-profile split, leak
    detection, episode termination) is already implemented and unit-tested in
    src/cmedalign/agents/{patient,episode}.py.
NEXT_COMMAND (once SSH to the rented GPU server is available):
  1. `git clone` this repo to the server.
  2. `make inventory` (G0 step 1) — hardware/driver/CUDA/disk/network, no assumptions
     about GPU count or model; if heterogeneous GPUs (e.g. A800 + 5090), group by type
     per the scheduling rules and never mix types in one NCCL job.
  3. `make env-check` (G0 step 2) — actually install requirements.lock.txt Section 2,
     re-pin exact versions once resolved, run the CUDA/NCCL/vLLM smoke test.
  4. Re-verify the OpenRLHF pin still matches (`git -C vendor/OpenRLHF log -1`) and its
     `--help` output still matches artifacts/audits/openrlhf_commit_notes.md before
     writing any training command.
  5. Proceed to G1 (data download + audit) per the Makefile.
ETA: local prep done in a single sitting; the real 96h execution clock starts once GPU
  server SSH is live and G0 passes.

---

## Log

### 2026-07-19 — local prep kickoff
- Confirmed with user: project root = E:\cmedalign; API adapter built against generic
  OpenAI-compatible spec (no real keys yet); dataset downloads wait for the rented server;
  assume standard Linux GPU rental (Ubuntu + CUDA + SSH).
- Ran local system check: Windows box, git 2.53.0, Python 3.13.9, conda 26.1.1, no GPU,
  815GB free on E:.
- git init'd this repo; created `.venv` for local CPU-only dev/test.
- Cloned OpenRLHF, pinned to tag v0.10.4 (commit ad1796e62b56bc9deae95542778336f88d24a3ed).
  Read actual argparse source (not hosted docs) for SFT/DPO/GRPO/multi-turn-agent flags —
  see artifacts/audits/openrlhf_commit_notes.md. Key finding: this pinned commit uses
  dotted namespaced flags (--algo.advantage.estimator group_norm for GRPO, etc.), which
  differs from the flat-flag style shown in OpenRLHF's hosted docs — do not copy doc
  commands verbatim. Also confirmed: OpenRLHF has zero references to `enable_thinking`,
  so the non-thinking flag must be applied in our own data-normalization step, not relied
  upon from OpenRLHF's dataset loader.
- Built and unit-tested (96 tests, all passing) the following GPU-independent modules:
  - `data/schemas.py`, `data/dedup.py` (exact + MinHash + pluggable embedding near-dup),
    `data/license_ledger.py`, `data/audit.py` (G1 CLI)
  - `eval/cmb_parser.py` (CMB multiple-choice answer extraction, 18 hand-built fixtures),
    `eval/judge.py` (dual-order judge JSON parsing + position-bias reconciliation)
  - `eval/api_adapter.py` (generic OpenAI-compatible client: env-var-only credentials,
    exponential backoff for transient errors only, idempotent on-disk cache, secret
    redaction in all logs — tested with a fully mocked HTTP layer, no real network/key)
  - `rewards/components.py` (weighted clinical/information/safety/process/communication
    + capped turn/length costs; verified a missed red flag and role leakage each cost
    more than the maximum possible turn/length cost, by construction and by test)
  - `stats/bootstrap.py`, `stats/tables.py` (percentile bootstrap CI, Holm-Bonferroni
    step-down correction, item-level-only table builder that raises on any missing field)
  - `eval/human_pack.py` (Latin-square-based exact position balancing across A-E labels,
    Fernet-encrypted blind map, forbidden-identifier leak check with a test that proves
    the check actually fires)
  - `agents/patient.py`, `agents/episode.py` (hidden-profile/visible-profile field
    split enforced by assertion, target/forbidden-claim leak detection, profile-ground-
    truth consistency checking, FINAL-marker/turn-budget episode termination)
  - `data/chat_template.py` — downloaded only the Qwen3-8B tokenizer (no weights) and
    hand-verified real chat-template behavior. Found and fixed three real footguns:
    (1) `return_assistant_tokens_mask=True` silently returns an all-zero mask for Qwen3
    (no `{% generation %}` block in its template — transformers only warns, doesn't
    raise); (2) with `enable_thinking=False` the empty `<think></think>` stub is only
    inserted for the conversation's *final* message, not every historical assistant
    turn — a naive per-turn `add_generation_prompt=True` render gets this wrong for
    every non-final turn (caught by an actual mismatched-tokenization test failure,
    then fixed by scanning the one true full-conversation tokenization instead of
    assuming the stub is always present); (3) `apply_chat_template(tokenize=True)`
    returns a `BatchEncoding` dict by default in the installed transformers version, not
    a plain list — `return_dict=False` is required. All three are written up in
    artifacts/audits/openrlhf_commit_notes.md and covered by tests/unit/test_chat_template.py.
- Drafted `requirements.lock.txt` (Section 1 verified by real install+test; Section 2
  GPU stack explicitly marked unverified/TBD), `Makefile` (all required targets; GPU-
  dependent ones are thin wrappers around scripts that don't exist yet — writing those
  scripts is server-session work), and `configs/{data,sft,dpo,grpo,eval,models}/*.yaml`
  skeletons reflecting the spec's starting hyperparameters and the real OpenRLHF flag
  names discovered above.
- Did NOT touch: real datasets, real API keys, any GPU/CUDA work, the actual
  `agent_func_path` implementation (needs a real LLM to play patient), OpenRLHF's own
  SFT collator internals (deferred, fast to check on the server).

### 2026-07-19 (later) — GitHub remote + first two API baselines wired and smoke-tested
- User provided a private GitHub remote (`terrense/medical_LLM_post-training-pipeline_project`)
  for ongoing iterative code + experiment-log management (not for server transfer --
  that'll likely be direct scp/rsync once SSH is available, TBD). Amended the bootstrap
  commit to drop the `Co-Authored-By: Claude` trailer per explicit user request (repo is
  meant to look authored solely by the user), renamed `master` -> `main`, pushed as the
  initial history. Going forward: no co-author trailer on this repo's commits.
- User pasted real DeepSeek and MiniMax API keys directly in chat (acknowledged it wasn't
  ideal, was in a hurry). Stored only in `E:\cmedalign\.env` (gitignored, verified with
  `git check-ignore`), never echoed back in chat or committed anywhere.
- DeepSeek: key matches the official platform's key format (`sk-` + 32 hex chars).
  Queried `GET https://api.deepseek.com/v1/models` (a free metadata call, not a paid
  generation) to get real model IDs rather than guessing -- confirmed
  `deepseek-v4-pro` and `deepseek-v4-flash` are the real server-side model strings (the
  spec's "V4-Pro"/"V4-Flash" naming was closer to the real IDs than DeepSeek's earlier
  models suggested, but still good that we checked rather than assumed). Ran one real
  minimal chat call (`max_tokens=200, temperature=0`) through our actual
  `api_adapter.OpenAICompatibleClient` code path (not just curl) for both -- both
  returned coherent, correct Chinese medical text, no `<think>` tags in either response
  (`had_think_tags: False` via `human_pack.strip_thinking_trace`), Pro hit
  `finish_reason: length` at 200 tokens (response was still complete/sensible; note for
  real eval runs to size `max_tokens` generously), Flash returned `finish_reason: stop`.
  `.env` updated with the confirmed `BASE_URL=https://api.deepseek.com/v1` and both
  model IDs. **DeepSeek V4-Pro and V4-Flash are now confirmed working end-to-end.**
- MiniMax M3: key format (`sk-cp-...`) does NOT match either official DeepSeek-style or
  typical official MiniMax (long JWT-style) key formats -- almost certainly a
  third-party proxy/reseller key, so its `BASE_URL` cannot be safely guessed (still
  `TODO_FILL_IN` in `.env`). **Waiting on user for the MiniMax endpoint (and, ideally,
  the exact model ID string) before it can be smoke-tested.**
- Added `scripts/smoke_test_api.py`: a manual (not part of the automated pytest suite,
  since it costs real money and needs real credentials) one-shot smoke test that loads
  `.env`, calls each configured alias once with a minimal prompt, and reports whether
  `<think>` tags showed up. Re-run this any time a new API alias is wired up.

### 2026-07-19 (later still) — MiniMax M3 confirmed working; all three API baselines live
- User pointed at MiniMax's official docs (`platform.minimaxi.com/docs/api-reference/text-anthropic-api`)
  to resolve the earlier "unknown proxy host" blocker. Fetched it (and the API reference
  index) instead of guessing: MiniMax exposes BOTH an Anthropic-messages-compatible API
  (`https://api.minimaxi.com/anthropic`) and a standard OpenAI-chat-completions-compatible
  one (`https://api.minimaxi.com/v1` domestic, `https://api.minimax.io/v1`
  international), model id `MiniMax-M3` either way. Used the OpenAI-compatible one since
  it matches our existing `api_adapter.py` without needing a second client
  implementation. Set `MINIMAX_M3_BASE_URL=https://api.minimaxi.com/v1`,
  `MINIMAX_M3_MODEL=MiniMax-M3` in `.env`.
- Ran the real smoke test: MiniMax M3 **is** a thinking model as the user warned --
  raw response came back wrapped in `<think>...</think>` (508 chars raw, reasoning
  visible in English even though the question was in Chinese) followed by a clean
  112-char Chinese answer. `human_pack.strip_thinking_trace` correctly removed the
  entire `<think>` block, leaving only the correct final answer. Confirmed by writing
  both raw and stripped text to UTF-8 files and reading them back (the Windows terminal
  itself garbles Chinese text in this session -- that's a display-only codepage issue,
  not a data problem; always verify via file, not raw terminal echo, when eyeballing
  non-ASCII API output here).
- **All three API baselines (DeepSeek V4-Pro, DeepSeek V4-Flash, MiniMax M3) are now
  confirmed working end-to-end through the real adapter code.** `.env` is fully filled
  in (no more `TODO_FILL_IN` placeholders).
- User also sent a WeChat article link claimed to be about DeepSeek V4 Pro/Flash --
  WebFetch got an anti-bot "环境异常" (environment exception) verification wall, not the
  actual article content, so nothing from it could be incorporated. Not blocking:
  Pro/Flash are already independently confirmed working via the official `/v1/models`
  endpoint and real completions above, so this doesn't gate anything.

### 2026-07-19（当天，第三轮）— 用户要求以后用中文沟通/记日志；论文加对照+补数据来源；
  正式落地"锚点决策"：导入数据、写 GRPO 骨架、上 data-juicer、规划双 conda 环境

**沟通约定变更**：用户要求以后全部用中文沟通，日志（本文件、BLOCKERS.md 等）也用中文写。
本条目起遵照执行；之前的英文条目不做翻译，保留原样。

**论文修改**（详见 `E:\cmedalign_paper\CHANGELOG.md`，已 git init 该目录并 commit）：
1. HuatuoGPT-o1-8B 从"可选"提升为"必做"对照——原稿主结果表(Table main)里其实完全没有
   这一行，只在模型登记表标注"optional"，这是原稿的疏漏，已经补上。
2. 新增"种子飞轮合成问诊数据"来源说明——用户澄清 `rlhf_lab_cloud_kit` 里的
   `internal_seed_flywheel`/`derived_from_seed` 数据是完全虚构的：DeepSeek-V4-Pro
   扮演患者、本项目基座 Qwen3-8B 扮演医生作答，再由更强模型+真实医护人员修订，不存在
   任何真实患者/机构隐私。这个数据来源之前完全没有写进论文，现已在 `main.tex` 数据表
   和方法节里正式描述，并在 `SOURCE_AUDIT.md` §6b 记录决策依据。
3. 做了一次网络检索，确认当前(2026-07中)中国大陆可购买的其他前沿级 API 候选：
   阿里 Qwen3-Max/3.7-Max、月之暗面 Kimi K2.5/K2.6、智谱 GLM-5/5.1——DeepSeek-V4-Pro
   本身已经是高性价比第一梯队选择，这只是给未来扩充数据提供候选方向，不是要改写已生成
   数据的溯源。

**PII/隐私问题彻底解决**：`internal_seed_flywheel`/`derived_from_seed` 不是真实数据，
两边项目的 BLOCKERS.md 都已更新为"已解决"。

**数据导入（真实跑过，不是占位符）**：
- 写了 `scripts/import_rlhf_lab_cloud_kit.py`，把 rlhf_lab_cloud_kit 的
  `05_final_v11/train.jsonl` + `dev.jsonl`（150,637 条）转换成 cmedalign 的
  `ConversationRecord` schema，按 source 分文件写入 `data/raw/rlhf_lab_cloud_kit/`。
  Schema 校验：150,635/150,637 通过（2 条 DISC-Med-SFT 的连续同角色问题，跟
  rlhf_lab_cloud_kit 那边审计脚本抓到的是同一批，互相印证）。
- **发现一个新的、独立的许可证问题**（不是隐私问题）：`med_zh_real`
  （58,038 条，单一最大来源）的 `license_id` 是 `unknown`——rlhf_lab_cloud_kit 的文档
  只说它"来自你本地 F 盘"，没有具体许可证记录。按项目规则"许可不明的数据先隔离"，
  已移动到 `data/quarantine/rlhf_lab_cloud_kit/med_zh_real.jsonl`，**没有**进入
  `data/raw/`，已记录到 `BLOCKERS.md`，等用户说清楚这批数据具体是哪个数据集/什么授权。
- 其余 7 个来源（Huatuo26M-Lite/DISC-Med-SFT/shibing624-finetune-zh/
  Chinese-medical-dialogue = 公开许可证；internal_seed_flywheel/derived_from_seed/
  gen_minimax_m3 = 用户自己生成的合成数据）已在 `data/manifests/license_ledger.json`
  标记为 `cleared`。
- 从 HuggingFace 下载了官方 **CMtMedQA**（`Suprit/CMtMedQA`，MIT 许可，68,023 条训练
  + `Suprit/CMtMedQA_test_v1`，apache-2.0，**1,000 条测试**——刚好和 `main.tex`
  Table 1 里写的"1,000"完全对上，交叉验证了官方 test split 没找错）。写了
  `scripts/import_cmtmedqa.py` 转换成同一套 schema，测试集单独存在
  `data/raw/cmtmedqa_test_HELD_OUT/`（目录名故意起得显眼，防止以后不小心拿去训练）。
  两个文件全部通过 schema 校验（69,023/69,023）。
- MedDG、IMCS-21 这两个 `main.tex` 点名要用的数据集在 HuggingFace 上搜不到（可能在
  GitHub 上，需要另外核实），**还没下载**，记入 `spec/tasks.md` 待办。

**GRPO agent 骨架**：写了 `src/cmedalign/agents/openrlhf_agent_func.py`，参照 vendor
里 OpenRLHF 的 `agent_func_openai_server_executor.py` 真实接口（`run_agent`
必须返回 `{"reward", "scores", "extra_logs"}`，`self.client` 是被训练的策略模型，
patient 模拟器必须走独立的 client，不能公用）。实现了医生-患者多轮循环、隐藏档案边界
（复用已验证的 `agents/patient.py`）、reward 里能直接算的部分（info_coverage，来自真实
revealed_info）。**明确标了 TODO 的部分**：clinical/safety/process/communication 四个
分量目前是占位符 0.5，真正需要一个"frozen evaluator with case-specific rubrics"
（论文原话）来打分，这个裁判的 prompt/校准工作留到有真实算力测试的时候再做，不在没有
反馈信号的情况下瞎写。本地验证过这个文件至少能正常 import（用了 try/except 兜底，
没装 OpenRLHF 时用 stub 基类，不会直接报错）。

**data-juicer 迁移**：用户要求以后清洗数据统一用 data-juicer，不再手写正则脚本。写了
`configs/data/data_juicer_sft.yaml`（沿用 rlhf_lab_cloud_kit 里 `dj_med_config.yaml`
验证过的算子命名）+ `scripts/data_juicer/{flatten_for_dj,rejoin_after_dj,dj_run}.py`
（拍平 messages→单字段 text 给 data-juicer 判断留/删，再映射回结构化记录，不会让
data-juicer 的文本级算子破坏 role/turn 结构；`dj_run.py` 里带了 rlhf_lab_cloud_kit
那边验证过的 numpy 兼容垫片）。**还没实际安装/跑过 data-juicer**（本机没必要装，
真正跑清洗建议在服务器上做）。

**环境隔离规划**：写了 `ENVIRONMENTS.md`，规划三个环境：`.venv`(本机开发/测试)、
`cmedalign-clean`(服务器上，data-juicer 专用)、`cmedalign-train`(服务器上，
torch/vLLM/DeepSpeed/OpenRLHF 训练专用)。`Makefile` 里训练相关目标已经加上
`conda run -n cmedalign-train` 前缀，`data-clean` 目标加上 `cmedalign-clean` 前缀。
实际创建这两个 conda 环境要等到了 GPU 机器上再做。

**新增的 schema 字段**（`src/cmedalign/data/schemas.py`）：`ConversationRecord` 加了
`task_type`（区分于 `specialty` 科室字段）、`quality_score`、`synthetic_or_real`、
`synthetic_provenance`（新的 `SyntheticProvenance` 子模型：teacher_model/
doctor_model/generation_date/reviewed_by/audit_status，标记为 synthetic 时强制要求
填写，用 `model_validator` 保证不会漏填）。补了 3 条新单元测试，全部 99 个测试通过。

**跑到一半 / 还在后台跑的**：`make data-audit`（跨源精确去重 + 训练集与 dev 集之间的
MinHash 近似去重污染检查）在导入的 92,599 条数据（train 90,231 + dev 2,368）上启动了，
纯 Python MinHash+LSH 在这个量级下比预期慢，还没跑完，会在跑完后补充结果。

**还没做，诚实列出来**：
- MedDG、IMCS-21 数据集下载
- `RESULTS_AND_TABLE_SCHEMA.md`/`HUMAN_EVALUATION_PROTOCOL.md`/`FIGURE_SPECIFICATIONS.md`
  的精确输出格式跟 `src/cmedalign/stats/`、`eval/human_pack.py` 的对齐（已读完这三份
  文件，但还没动代码去对齐）
- data-juicer 实际安装验证
- GRPO agent 骨架里的裁判打分逻辑（明确留到有真实算力时做，不是遗漏）
