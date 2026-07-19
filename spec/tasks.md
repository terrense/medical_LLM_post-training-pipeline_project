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

## Phase 0.0 — THE ANCHOR: read this before anything else

`E:\cmedalign_paper\main.tex` (+ its `RESULTS_AND_TABLE_SCHEMA.md`,
`HUMAN_EVALUATION_PROTOCOL.md`, `FIGURE_SPECIFICATIONS.md`, `SOURCE_AUDIT.md`) is now the
single source of truth for what this project builds. See `spec/design.md`'s "THE ANCHOR"
section for the full reasoning. Two decisions that change prior assumptions:

- [ ] **GRPO must be the multi-turn patient-simulation environment `main.tex` describes**
      (frozen Qwen2.5-7B-Instruct patient, hidden profile, 5-component reward), **not**
      a static MCQ rule-reward setup. A separate, already-completed MCQ-reward GRPO
      round exists in `rlhf_lab_cloud_kit` — that is a side-result, never to be reported
      as this paper's M3.
- [ ] Tool-calling, DAgger loops, agent action-types, packing/loss-norm/curriculum as
      *reported research comparisons*, and gradient-conflict/PCGrad/multi-adapter are
      explicitly **out of scope** for this paper (removed 2026-07-19). Don't build these
      even though `rlhf_lab_cloud_kit`'s separate SFT-rigor checklist calls for them —
      that checklist is generic SFT hygiene, not this paper's contribution.
- [x] ~~`rlhf_lab_cloud_kit` is now historical/frozen; pull in its reusable assets~~ —
      **done 2026-07-19**: `scripts/import_rlhf_lab_cloud_kit.py` imported 150,637
      records into `data/raw/rlhf_lab_cloud_kit/` (split by source), converted to
      `ConversationRecord` schema (150,635/150,637 pass validation — 2 known
      consecutive-same-role records from `DISC-Med-SFT`, same ones rlhf_lab_cloud_kit's
      own audit found). The `internal_seed_flywheel`/`derived_from_seed` PII question is
      **resolved** (fully synthetic, no real patient data — see `BLOCKERS.md` and
      `cmedalign_paper/SOURCE_AUDIT.md` §6b) and both are imported and license-cleared as
      `proprietary-own-synthetic-data`. **`med_zh_real`** (58,038 records, the single
      largest source) has `license_id: unknown` in the source data — content-quality
      audited 2026-07-19 (0% ad/PII/dangerous-advice regex hits, 0.03% unhedged-
      diagnosis hits, 6 records manually read in full, all clinically sensible; ~2.5%
      show "作为智能助手..." boilerplate suggesting partial AI-generated origin) and
      **authorized by the user to use now** pending exact source confirmation (see
      `BLOCKERS.md` Reminders — not fully resolved, just unblocked; follow up once the
      user confirms which open dataset this actually is, then backfill the license
      ledger + `main.tex` data table with the real citation). Moved from quarantine
      into `data/raw/rlhf_lab_cloud_kit/`.
- [x] Read `FIGURE_SPECIFICATIONS.md` and `SOURCE_AUDIT.md` in full — **done 2026-07-19**.
- [x] **Done 2026-07-19** — Reconciled `src/cmedalign/stats/` and
      `eval/human_pack.py` against `RESULTS_AND_TABLE_SCHEMA.md`'s exact contract:
      - `src/cmedalign/schema/results_layout.py`: canonical `results/`/`tables/`/
        `figures/`/`human_eval/` directory contract (exact filenames from the spec) +
        `ensure_results_layout()` (scaffolds dirs, doesn't fabricate files) +
        `validate_identity_fields()` (checks the required identity-field list, core +
        API-specific).
      - `src/cmedalign/schema/records.py`: pydantic models matching the spec exactly --
        `StatisticsEntry` (the precise `comparison_id/model_a/model_b/.../seed` schema,
        with `require_complete()` enforcing "fails when n_cases/estimate/CI/p_holm/
        subset_id are absent for a primary claim" as actual code, not just prose),
        `TableMainRow` (exact `table_main_universal.csv` columns, validates
        percentages in [0,100] and CI brackets the point estimate),
        `StageAblationRow` (M0-M3 only), `HumanTableRow`, `ErrorAdjudicationRecord`
        (the 9 fixed error categories from FIGURE_SPECIFICATIONS.md Figure 5).
      - `src/cmedalign/stats/build_tables.py`: real builder functions
        (`build_table_main`, `build_stage_ablation`, `build_human_table`) that take
        item-level records and produce schema-conformant rows, including correct
        lower-is-better metric orientation for `mean_rank` (safety_violation). Backs
        the `make statistics` Makefile target. All tested against synthetic
        item-level fixtures (18 new tests across `test_results_schema.py` +
        `test_build_tables.py`), not just imported.
      - Still needed once real results exist: actually wiring `eval-core`/`eval-baselines`
        to write `results/benchmark_item_scores.jsonl` etc. in the first place — the
        builders are ready to consume that format, but nothing produces it yet (no
        training/eval has run).
- [x] **Done 2026-07-19** — `human_pack.py` now has `RATING_DIMENSIONS` (the five
      1-5 dimension anchors copied verbatim from `HUMAN_EVALUATION_PROTOCOL.md` §8,
      not paraphrased) + `build_rating_schema()` (produces `rating_schema.json` incl.
      the §9 extra fields and §15 verbatim opening text) + `build_rater_assignment()`/
      `write_rater_assignment_csv()` (produces `assignment.csv`: every case gets >=2
      independent raters, 25% get a 3rd for reliability, per §7, deterministic,
      load-balanced). Still missing: `ratings_raw/`, `ratings_anonymized.csv` writer,
      `human_statistics.json` writer (the alpha computation itself is now done, see
      below — the writer that assembles the full `human_statistics.json` file per
      RESULTS_AND_TABLE_SCHEMA.md is still separate work).
- [x] **Done 2026-07-20** — `src/cmedalign/stats/reliability.py`: ordinal
      Krippendorff's alpha (`krippendorff_alpha_ordinal`) + case-clustered bootstrap CI
      (`bootstrap_alpha_ci`), hand-implemented (no suitable dependency found), 6 new
      tests (perfect agreement -> exactly 1.0, systematic extreme disagreement -> <0,
      single-rater units correctly ignored not crashed, degenerate all-same-category
      correctly raises rather than returning garbage). 128/128 tests passing overall.
      OLD (superseded) note below, kept for history:
      ~~`human_statistics.json` (needs ordinal Krippendorff's alpha — not yet
      implemented, no library for it is installed; would need either a manual
      implementation or adding a dependency), `protocol_deviations.md` template.
- [x] Data cleaning tooling switched to **data-juicer** per user instruction
      (2026-07-19) instead of hand-written regex scripts. Wrote
      `configs/data/data_juicer_sft.yaml` (adapted from rlhf_lab_cloud_kit's own
      validated `dj_med_config.yaml` operator names) +
      `scripts/data_juicer/{flatten_for_dj,rejoin_after_dj,dj_run}.py` (flatten
      messages->single text field for data-juicer's filters/dedup to score, map its
      keep/drop decision back onto the original structured record — never let
      data-juicer's text-level mappers rewrite message content directly). User confirmed
      (2026-07-19) the imported rlhf_lab_cloud_kit data was already processed with
      data-juicer previously -- this config is for cmedalign's *own* pipeline going
      forward, not a re-clean of already-processed data. **Not yet
      installed/run here** — data-juicer itself isn't installed anywhere yet (see
      `ENVIRONMENTS.md`); user confirmed no rush, install/run on the server, not now.
- [x] **Two-conda-env split planned** (`ENVIRONMENTS.md`, per user instruction that
      cleaning and training likely need separate environments, especially once on a real
      server): `.venv` (this machine, dev/test), `cmedalign-clean` (data-juicer),
      `cmedalign-train` (torch/vLLM/DeepSpeed/OpenRLHF). `Makefile` targets updated to
      `conda run -n <env>` accordingly. Actual env creation happens on the GPU server.
- [x] GRPO `agent_func_path` scaffold written: `src/cmedalign/agents/openrlhf_agent_func.py`,
      matching the real `AgentExecutor.run_agent()` interface from
      `vendor/OpenRLHF/examples/python/agent_func_openai_server_executor.py`. Implements
      the doctor<->patient multi-turn loop and the deterministic reward piece
      (`info_coverage`, from real revealed-required-info tracking).
- [x] **Done 2026-07-19** — reward judge implemented for real (was a placeholder):
      `src/cmedalign/rewards/judge_scorer.py` calls a real LLM judge (default
      `DEEPSEEK_V4_PRO`, swappable via `CMEDALIGN_REWARD_JUDGE_ALIAS`) with a prompt
      built from the exact reward-component rubric table in `main.tex`'s appendix
      (verbatim, not paraphrased), parses clinical/safety/process/communication scores,
      falls back to a flagged neutral score (not a crash) on unparseable judge output.
      3 unit tests with mocked API responses. Wired into `openrlhf_agent_func.py`'s
      `_score_episode`, replacing the old 0.5 placeholders; `red_flag_missed` also now
      computed (heuristic: red-flag text must appear verbatim among revealed
      required_info items -- re-verify against real GRPO profiles once built, not yet
      validated against real data). Important practical note recorded here and in the
      module docstring: a ChatGPT/Claude.ai *subscription* (web UI / Codex access) is
      NOT the same as *API access* -- automated per-rollout scoring needs a real,
      metered API key, which only exists today for DeepSeek/MiniMax.
      `scripts/calibrate_reward_judge.py`: the "抽样检查" cross-judge calibration script
      (re-scores a sample of saved episodes with a second judge, reports per-component
      Spearman correlation) -- matches the paper's own "at least two judge families,
      report consensus/correlation" requirement, without needing every rollout scored
      twice. **Still not run for real** (no real episodes exist yet, needs GPU/vLLM).
- [x] Downloaded official **CMtMedQA** train (`Suprit/CMtMedQA`, MIT, 68,023 records) and
      **held-out test** (`Suprit/CMtMedQA_test_v1`, apache-2.0, exactly 1,000 records —
      matches `main.tex` Table 1's stated count, cross-validated). Converted via
      `scripts/import_cmtmedqa.py`, both pass schema validation (69,023/69,023). Test
      split stored in a deliberately obvious `data/raw/cmtmedqa_test_HELD_OUT/` directory
      name so it can't be accidentally swept into a training glob later.
- [ ] **MedDG and IMCS-21 (both named explicitly in `main.tex` Table 1) not found on
      HuggingFace** via search — likely GitHub-hosted, not yet located/downloaded. Still
      to do.
- [x] `make data-audit` finished: **G1 = FAIL** (this is useful, real information, not a
      failure of the tooling). No exact duplicates in train, no train/test contamination
      (no test set loaded yet), but **732 train<->dev near-duplicate pairs found**, nearly
      all involving `derived_from_seed` "summary" records
      (`rlck_drv_summary_synthetic_diag_pipeline_...`). This is exactly the patient/
      case-level split risk flagged earlier: synthetic case-derived rewrites of the same
      underlying case appear to have been split across train and dev independently
      instead of staying together. **Needs a real fix**: either re-split
      `derived_from_seed` at the case/pipeline-run level before re-importing, or exclude
      it from `dev`/`test` construction entirely until re-split. Not yet fixed — noted
      here rather than silently worked around. Full report: `artifacts/audits/data_audit.json`.
- [ ] MedDG located on GitHub (`github.com/lwgkzl/MedDG`) but **no LICENSE file detected**
      via the GitHub API — same "unclear license" situation as `med_zh_real`, would need
      quarantine treatment unless the paper/repo states usage terms elsewhere (not yet
      checked). IMCS-21's GitHub location not yet located. Neither downloaded.

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

- [x] ~~Waiting on user: local path to pre-existing multi-turn medical dialogue data~~ —
      resolved 2026-07-19: this is `E:\rlhf_lab_cloud_kit` (146,809 records, 10
      task_types, already cleaned/deduped). See Phase 0.0 above and `spec/design.md`'s
      "THE ANCHOR" section for exactly what gets pulled in and what's still blocked
      (PII status of the real-business-data sources).
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

- [ ] Import the reusable, already-cleaned/deduped/task_type-tagged records from
      `E:\rlhf_lab_cloud_kit\data\eval_sets\05_final_v11\train.jsonl` (146,809 records)
      into `data/raw/` — per `spec/design.md`, exclude `internal_seed_flywheel` and
      `derived_from_seed` until the PII blocker there is resolved; the remaining
      open-source-tagged sources (`Huatuo26M-Lite`, `DISC-Med-SFT`,
      `Chinese-medical-dialogue`, `shibing624-finetune-zh`, `med_zh_real`) can be
      imported now
- [ ] Still need, on top of the above: CMtMedQA train/test official split, MedDG,
      IMCS-21 (all named explicitly in `main.tex`'s data table but not present in the
      rlhf_lab_cloud_kit pool) — download/identify these separately
- [ ] Download eval benchmarks: CMB-Exam, CMB-Clin, CMtMedQA_test, CliMedBench; attempt
      MedBench submission or save conformant pending submission
- [ ] Re-run `src/cmedalign/data/dedup.py`'s embedding-similarity gate (the one
      near-dup check rlhf_lab_cloud_kit never implemented) against the imported pool,
      not just exact+MinHash
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
