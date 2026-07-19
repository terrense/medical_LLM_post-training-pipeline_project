# tasks.md — durable checklist (do not rely on Claude Code's in-session task list)

**Important for any fresh session:** Claude Code's built-in `TaskCreate`/`TaskList` tool
state does **not** survive a new conversation/window — verified empirically on
2026-07-19 (a brand new `TaskList` call in the same repo returned "No tasks found" after
a prior session had created and completed 11 tasks there). This file is the durable
replacement. **Update it, in the same commit as the actual work, every time a task
changes state.** Check items off with `[x]`, add new tasks as they're discovered, don't
leave it stale.

Read order for a cold start: `spec/requirements.md` (what/why) → `spec/design.md`
(decisions made) → this file (exact current checklist) → `STATUS.md` (chronological
detail + very latest log entries, which may be newer than this file if it wasn't
updated yet — check `STATUS.md`'s top block's date against this file's "Last updated").

**Last updated:** 2026-07-19

---

## Phase 0 — Local prep (this Windows machine, no GPU rented yet)

- [x] Project scaffold created at `E:\cmedalign`, git initialized
- [x] `.gitignore` (weights/data/`.env`/large results excluded; code/config/manifests/tables tracked)
- [x] `README.md`, `STATUS.md`, `BLOCKERS.md` created
- [x] `Makefile` with all required targets (GPU-dependent ones are thin wrappers around
      scripts that don't exist yet — see Phase 1 below)
- [x] OpenRLHF cloned to `vendor/OpenRLHF`, pinned to tag `v0.10.4`
      (`ad1796e62b56bc9deae95542778336f88d24a3ed`), real CLI flags verified against
      source (not hosted docs) — `artifacts/audits/openrlhf_commit_notes.md`
- [x] `requirements.lock.txt` Section 1 (CPU-only) — verified by real install + full
      test suite pass
- [x] `requirements.lock.txt` Section 2 (GPU stack) — draft only, **unverified**, TBD on server
- [x] `configs/{data,sft,dpo,grpo,eval,models}/*.yaml` skeletons written
- [x] `src/cmedalign/data/schemas.py` — `ConversationRecord`, `PatientProfile`,
      `RawGeneration` pydantic schemas + tests
- [x] `src/cmedalign/data/dedup.py` — exact hash, MinHash near-dup, pluggable embedding
      near-dup, cross-split contamination check + tests
- [x] `src/cmedalign/data/license_ledger.py` — license/provenance ledger + trainability gate
- [x] `src/cmedalign/data/audit.py` — G1 CLI (not yet run against real data)
- [x] `src/cmedalign/data/chat_template.py` — Qwen3 assistant-only masking, verified
      against real downloaded tokenizer (no weights), 3 real footguns found+fixed (see
      `spec/design.md`)
- [x] `src/cmedalign/eval/cmb_parser.py` — CMB multiple-choice parser + 18 fixtures
- [x] `src/cmedalign/eval/judge.py` — dual-order judge JSON parsing + reconciliation
- [x] `src/cmedalign/eval/api_adapter.py` — generic OpenAI-compatible client (env-var
      creds, redacted logs, idempotent cache, transient-only retry)
- [x] `src/cmedalign/eval/human_pack.py` — blind packaging, Latin-square position
      balance, Fernet-encrypted blind map, leak check
- [x] `src/cmedalign/rewards/components.py` — weighted reward + dominant red-flag/role-
      leakage penalties
- [x] `src/cmedalign/stats/bootstrap.py`, `stats/tables.py` — bootstrap CI,
      Holm-Bonferroni, item-level-only table builder
- [x] `src/cmedalign/agents/patient.py`, `agents/episode.py` — hidden-profile boundary,
      leak/consistency checks, episode termination
- [x] `tests/unit/*` — 96/96 passing (run: `pytest tests/unit -q`)
- [x] Private GitHub remote wired: `terrense/medical_LLM_post-training-pipeline_project`,
      branch `main`, no Claude co-author trailer
- [x] All 3 API baselines confirmed live end-to-end: `DEEPSEEK_V4_PRO`,
      `DEEPSEEK_V4_FLASH`, `MINIMAX_M3` — `.env` fully filled in (never committed)
- [x] `scripts/smoke_test_api.py` — manual (not pytest) one-shot API smoke test
- [x] `spec/requirements.md`, `spec/design.md`, `spec/tasks.md` (this file) — Kiro-style
      structured docs for cold-start continuity

### Open / pending in Phase 0

- [ ] **Waiting on user:** local path to their pre-existing multi-turn medical dialogue
      data, plus provenance (self-collected vs third-party) — determines
      `data/raw/` (cleared) vs `data/quarantine/` (unclear license). User said (2026-07-19)
      the data is being prepared/copied and will take time.
- [ ] Nothing else is blocked in Phase 0 — everything GPU-independent that could be
      built without real data has been built.

---

## Phase 1 — Once SSH to the rented GPU server exists (not started)

Do these in order; each is a real gate, don't skip ahead if one fails.

- [ ] Get this repo onto the server (method TBD — likely direct scp/rsync from this
      Windows machine, or `git clone` from the GitHub remote; decide when SSH details exist)
- [ ] `make inventory` — G0 step 1: `nvidia-smi -L`, per-GPU VRAM, driver/CUDA, CPU,
      RAM, disk, network, current Python, occupied GPUs, ulimits →
      `artifacts/system_inventory.txt`. **Do not assume GPU count/model.**
- [ ] If heterogeneous GPUs (e.g. A800 + 5090): group by type, never mix types in one
      NCCL job (see `spec/requirements.md` 硬件调度原则)
- [ ] Disk budget check: model+data+checkpoint+rollout space vs available; if <20%
      margin, report in `BLOCKERS.md` with a caching/compression plan before proceeding
- [ ] `make env-check` — G0 step 2: actually install `requirements.lock.txt` Section 2,
      re-pin exact resolved versions, single-GPU matmul + same-type multi-GPU NCCL +
      vLLM 8B generation + disk write + checkpoint save/load, all passing
- [ ] Re-verify OpenRLHF pin still matches (`git -C vendor/OpenRLHF log -1` ==
      `ad1796e62b56bc9deae95542778336f88d24a3ed`) and re-run its own `--help` against
      `artifacts/audits/openrlhf_commit_notes.md` before writing any training command
- [ ] **G0 gate: PASS/FAIL recorded in `STATUS.md` before proceeding**

## Phase 2 — Data (G1)

- [ ] Resolve the user's pre-existing data provenance (see Phase 0 open item) — do this
      FIRST, it may already be sitting in `data/raw/` or `data/quarantine/` by then
- [ ] Download/identify SFT corpora (target 80k-120k, quality-filtered, stratified) —
      see `configs/data/sources.yaml` (currently empty placeholder list)
- [ ] Download eval benchmarks: CMB-Exam, CMB-Clin, CMtMedQA_test, CliMedBench; attempt
      MedBench submission or save conformant pending submission
- [ ] Populate `data/manifests/license_ledger.json` for every source (via
      `src/cmedalign/data/license_ledger.py`) — anything unclear → `data/quarantine/`, never trained on
- [ ] Run `make data-audit` (`src/cmedalign/data/audit.py`) — must PASS: no license
      quarantine issues, no train/test contamination (exact + MinHash), no train/dev contamination
- [ ] **G1 gate: PASS/FAIL recorded in `STATUS.md`**

## Phase 3 — G2/G3 diagnostics before training

- [ ] Build the 200-prompt "倒豆子" diagnostic set (see `spec/requirements.md` §4.3) and
      run it against M0 (Qwen3-8B) — check role-marker leakage, fake continuation,
      multiple assistant turns, EOT/finish_reason correctness, length vs 95th-pct
      threshold, doctor/patient role inversion
- [ ] `test_official_split_immutability`, `test_no_future_turn_continuation_fixture` —
      not yet written (need real data/model to construct meaningfully), add here
- [ ] Base vs post-trained initialization probe (Qwen3-8B-Base ablation, capped small
      SFT subset, same tokens/optimizer) — compare role leakage, future-turn
      continuation, stop-token compliance, overlong outputs
- [ ] CMB/CliMedBench parser fixtures already pass locally (`tests/unit/test_cmb_parser.py`)
      — re-verify against real benchmark data once downloaded
- [ ] **G2/G3 gate: PASS/FAIL recorded in `STATUS.md`**

## Phase 4 — SFT (M1)

- [ ] `configs/sft/smoke.yaml` lr sweep (5e-5 vs 1e-4, fixed seed) on training/dev only
- [ ] Freeze winning lr into `configs/sft/main.yaml`, run `make train-sft`
- [ ] Save merged HF checkpoint + LoRA adapter to `checkpoints/m1_sft/`
- [ ] Run core eval suite against M1 (same as M0)
- [ ] **G5 gate for M1: PASS/FAIL recorded in `STATUS.md`**

## Phase 5 — DPO (M2)

- [ ] `make build-dpo` — construct preference pairs from TRAIN prompts only (M1
      multi-seed + optional API teacher), dual-judge rule+order-flip filtering
      (`src/cmedalign/eval/judge.py`), target 10k-20k hard-negative pairs
- [ ] `configs/dpo/smoke.yaml` gate, then freeze into `configs/dpo/main.yaml`,
      `make train-dpo`
- [ ] Save M2 checkpoint, run core eval suite against M2
- [ ] **G5 gate for M2: PASS/FAIL recorded in `STATUS.md`**

## Phase 6 — GRPO (M3)

- [ ] Write `src/cmedalign/agents/openrlhf_agent_func.py` (the actual
      `--train.agent_func_path` script) — reference
      `vendor/OpenRLHF/examples/python/agent_func_openai_server_executor.py`, wire in
      `src/cmedalign/agents/patient.py` + `episode.py` boundary logic, frozen
      Qwen2.5-7B-Instruct as the patient LLM
- [ ] `make build-profiles` — 2k-4k patient profiles built from TRAIN conversations only
- [ ] `make smoke-grpo` on 32 profiles (full rollout→reward→backprop), then 200 profiles
      stability smoke — check reward/KL/entropy/clip-fraction/length/safety-floor before
      the main run
- [ ] Freeze `configs/grpo/main.yaml` from smoke gate results, `make train-grpo`
- [ ] Select M3 via frozen dev composite ONLY — never look at test before selection
- [ ] Save M3 checkpoint, run core eval suite against M3
- [ ] **G4/G5 gate for M3: PASS/FAIL recorded in `STATUS.md`**

## Phase 7 — Full evaluation (G6)

- [ ] `make eval-core` — M0/M1/M2/M3 × required benchmarks, full-local track
- [ ] `make eval-baselines` — Qwen3-32B, Qwen2.5-72B-Instruct, Baichuan-M2-32B,
      (optional HuatuoGPT-o1-8B), DeepSeek-V4-Pro/Flash, MiniMax-M3 — universal track,
      never mixed into the same column as full-local
- [ ] `make statistics` — bootstrap CI + Holm correction tables from item-level results only
- [ ] `make figures` — 5 paper figures, PDF+PNG+meta JSON
- [ ] **G6 gate: PASS/FAIL recorded in `STATUS.md`**

## Phase 8 — Human eval

- [ ] Pre-select the "strongest external baseline" for the 5-way blind comparison
- [ ] `make human-pack` — 80 stratified cases, blind package (already implemented +
      tested with synthetic data; needs real M0-M3 + baseline outputs)
- [ ] Hand off blind package to the author's rating process (external to this repo's automation)
- [ ] Collect ratings, compute Spearman correlation vs automated judges

## Phase 9 — Final deliverables

- [ ] `paper_support/claim_values.tex` — verified-only number macros
- [ ] `EXPERIMENT_REPORT.md` — failed runs, negative results, deviations, repro commands
- [ ] `make validate` — full deliverable checklist validation
- [ ] `make archive` — `cmedalign_artifacts_<commit>.tar.zst` + SHA-256

---

## Known deferred items (not blockers, just not-yet-done, see `spec/design.md` for why)

- OpenRLHF's own SFT collator masking internals — not read in full yet
- `requirements.lock.txt` Section 2 — unverified draft
- GSPO — explicitly out of scope until core chain + all required evals are archived
