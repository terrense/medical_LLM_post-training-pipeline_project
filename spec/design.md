# design.md — decisions actually made, organized by topic

This is the "why" companion to `spec/requirements.md` (the "what"). Organized by topic,
not chronologically — for the chronological blow-by-blow, see `STATUS.md`'s Log section.
If you're picking this project back up cold, read `spec/requirements.md` first, this
file second, then `spec/tasks.md` for exactly what's done vs pending.

## THE ANCHOR (2026-07-19): `E:\cmedalign_paper` is the single source of truth for scope

After building out a generic 75-point SFT-rigor checklist (a *different*, unrelated
project's checklist, see below) and cross-referencing an earlier, already-completed
local training round, the user pointed at `E:\cmedalign_paper` — the actual manuscript
package this whole project was scaffolded from (`CLAUDE_CODE_EXECUTION_PLAN.md` in that
folder is verbatim identical to `spec/requirements.md` here). Reading `main.tex` in full
resolved a real scope-creep problem: **cmedalign's job is to fill in `main.tex`'s ~84
`TBD` macros, using exactly the methodology the paper describes — nothing more, nothing
that doesn't serve that.** Concretely, this means:

- **In scope, because the paper explicitly needs it**: M0→M1→M2→M3 stage comparison;
  CMB-Exam/CMB-Clin/CMtMedQA_test/CliMedBench/MedBench; a multi-turn patient-simulation
  GRPO environment with the 5-component reward (`clinical/info/safety/process/comm`,
  see the exact formula in `main.tex` §Reward design); a reward-component ablation
  (remove info reward, remove safety reward, remove turn/length costs); the Base-vs-
  Instruct initialization probe; comparison against Qwen3-32B/Qwen2.5-72B-Instruct/
  Baichuan-M2-32B/DeepSeek-V4-Pro/DeepSeek-V4-Flash/MiniMax-M3/optional HuatuoGPT-o1-8B;
  blinded 80-case human eval with Krippendorff's alpha and Spearman judge-agreement;
  Holm-corrected statistics throughout.
- **Explicitly OUT of scope for this paper** (dropped 2026-07-19 per user instruction
  "tool-calling 似乎没价值...我们要统一一个主心骨"): tool-calling data/evaluation,
  DAgger-style recovery-trajectory loops, explicit agent action-type decomposition
  (ASK/ANSWER/CALL_TOOL/...), sequence-packing as a reported research comparison,
  token-level-vs-sample-level loss normalization as a reported research comparison,
  staged-vs-mixed curriculum as a reported research comparison, gradient-conflict/
  PCGrad/KL-distillation/multi-adapter experiments. None of these appear anywhere in
  `main.tex`. They came from a *separate* engineering-rigor checklist
  (`E:\rlhf_lab_cloud_kit\spec\requirements_sft_rigor.md`) that is good general SFT
  hygiene but is not this paper's contribution — building them would be exactly the
  "led astray by a previous experiment" failure mode the user called out.
- **Authoritative schemas that supersede this project's own generic implementations**:
  `E:\cmedalign_paper\RESULTS_AND_TABLE_SCHEMA.md` defines the exact `results/`,
  `tables/`, `figures/`, `human_eval/` directory contract, required identity fields
  (`run_id, code_commit, config_hash, prompt_hash, model_label, model_id_exact, ...`),
  and the exact `statistics.json` per-comparison schema
  (`comparison_id, model_a, model_b, metric, paired, n_cases, estimate, ci_method,
  ci_low, ci_high, p_raw, p_holm, effect_size, subset_id, seed`). This is stricter and
  more specific than the generic `build_table_from_item_level`/`bootstrap_ci` utilities
  already built in `src/cmedalign/stats/` — those utilities are still useful primitives,
  but the actual output files must conform to this exact schema, not an ad-hoc one.
  Likewise `E:\cmedalign_paper\HUMAN_EVALUATION_PROTOCOL.md` is the authoritative,
  fully-specified human-eval design (exact 5-dimension 1-5 rubric text with anchors,
  role definitions, calibration procedure, severe-disagreement escalation rules,
  deliverable file list) — `src/cmedalign/eval/human_pack.py`'s blinding/Latin-square/
  encryption mechanics are compatible with it (file names already happen to match:
  `cases_blinded.jsonl`, `blind_map.enc`) but the actual rating rubric content must be
  copied verbatim from that protocol document, not invented generically, and several
  deliverables it requires (`assignment.csv` with rater IDs, `ratings_raw/`,
  `ratings_anonymized.csv`, `human_statistics.json`, `protocol_deviations.md`) are not
  yet built. `FIGURE_SPECIFICATIONS.md` and `SOURCE_AUDIT.md` are likewise authoritative
  for figure generation and citation control respectively, and should be read in full
  before that work starts (not yet done as of this writing — flagged in `spec/tasks.md`).

### The one real conflict this surfaced, and how it was resolved

`E:\rlhf_lab_cloud_kit` (a separate, earlier, already-executed local/company-GPU project
using LLaMA-Factory — see its own `spec/design.md` for full detail) had already run a
complete SFT→DPO→GRPO round with real results. But its GRPO stage is **not the same
experiment** as this paper's GRPO: round 1 used a static CMExam multiple-choice
rule-reward (`correct answer = 1`), whereas `main.tex` requires a multi-turn
patient-simulation environment with a frozen `Qwen2.5-7B-Instruct` patient and the
5-component reward. Reusing round 1's GRPO result as M3 would misrepresent what the
paper claims to have measured. **Decision (user, 2026-07-19): rebuild GRPO as the real
multi-turn patient-simulation environment `main.tex` describes; round 1's MCQ-reward
GRPO is a separate, smaller side-result (interesting RLVR data point, not M3).**

**Decision (user, 2026-07-19): consolidate to a single active project.** Going forward,
`cmedalign` (this project) is the one pipeline that actually runs the paper's
experiments. `rlhf_lab_cloud_kit` is now historical/frozen — it does not get new
training runs — but several of its assets are real, reusable inputs to this project's
data pipeline, not to be rebuilt from scratch:

- The 10-category `task_type` taxonomy and its 146,809 real, already-classified,
  already-deduplicated records (`Huatuo26M-Lite`, `DISC-Med-SFT`,
  `Chinese-medical-dialogue`, `shibing624-finetune-zh` = open-source; `med_zh_real` =
  large real-world QA pool; `internal_seed_flywheel`/`derived_from_seed` = real business
  data, **PII status unresolved, see `E:\rlhf_lab_cloud_kit\BLOCKERS.md` #1 — do not pull
  this specific source into cmedalign's training pool until that's answered**).
- The cleaning/dedup pipeline logic (`clean_02.py`'s ad/PII/dosage/definitive-diagnosis/
  danger regex filters, `dedup_04.py`'s exact-hash dedup, `sample_05_v11.py`'s MinHash
  near-dup) — reuse the *logic*, not necessarily the literal scripts, since cmedalign's
  own `src/cmedalign/data/dedup.py` already implements exact+MinHash+embedding dedup
  independently and is unit-tested; the embedding-similarity gate that
  `rlhf_lab_cloud_kit` never implemented is one `cmedalign` already has (untested against
  real data yet).
- The already-measured M0 baseline numbers (CMB 73.8%, CMExam 82.6% on the base model —
  note: paper's M0 is the **Instruct**, non-thinking checkpoint, not the base model that
  earlier baseline was measured on; these numbers are a useful sanity reference, not a
  substitute for re-measuring M0 = Qwen3-8B-Instruct per the paper's own protocol).
- The locked LoRA hyperparameters (`rank=64, alpha=128, dropout=0.05, target=all-linear`)
  — this already matches `main.tex`'s own frozen hyperparameter table exactly, so the
  rank/lr/target-module sweep that produced it does not need to be redone; only the
  *learning rate* is still genuinely open in both places (marked `TBD` in `main.tex`,
  and round 1's sweep picked `lr=2e-4` for a different data mixture/task than this
  paper's — needs its own smoke sweep per `spec/requirements.md`'s SFT section, not
  assumed to transfer).
- The DPO dual-judge pattern (`gen_dpo_pairs.py`, `dpo_judge.py` used MiniMax-M3 +
  DeepSeek as two independent judges) is structurally close to what `main.tex` requires
  ("two independently configured judge models... both presentation orders") — worth
  checking whether it already does order-counterbalancing before assuming it needs to
  be rebuilt from scratch; not yet verified (see `spec/tasks.md`).

## Where things stand (one paragraph)

Local prep phase, no GPU server rented yet. Everything GPU-independent is implemented
and unit-tested (96/96 passing): data schemas/dedup/license-ledger, CMB/judge parsers,
API adapter, reward components, stats utilities, human-eval blind packaging, patient-
agent boundary logic, and Qwen3 chat-template masking (verified against the real
tokenizer, no model weights downloaded). OpenRLHF is pinned and its real CLI surface
documented. All three API baselines (DeepSeek V4-Pro, V4-Flash, MiniMax M3) are wired
into `.env` and confirmed working end-to-end. Nothing has touched a GPU, a real training
dataset, or the actual GRPO agent implementation yet — those need the rented server.

## Repo / process decisions

- **Project root:** `E:\cmedalign` on this Windows machine, git-initialized, pushed to
  a private GitHub remote `terrense/medical_LLM_post-training-pipeline_project` (branch
  `main`). The remote is for iterative code + experiment-log management per the user's
  explicit request — NOT primarily for getting code onto the rented server (that will
  likely be direct scp/rsync once SSH exists, still TBD).
- **No `Co-Authored-By: Claude` trailer on this repo's commits** — user wants commits to
  read as their own authorship on GitHub. (This is a project-specific override of the
  default Claude Code commit convention; don't apply it to other repos unless asked.)
- **Documentation structure follows a Kiro-like spec/design/tasks split** (per user
  suggestion) instead of one big linear log, specifically so a fresh session with zero
  conversation memory can reconstruct full context fast. `STATUS.md`'s log section is
  kept too, but only for chronological "what happened when" — this file and
  `spec/tasks.md` are the ones to read for "what's true right now."
- **Secrets:** real API keys live only in `E:\cmedalign\.env` (gitignored, verified via
  `git check-ignore`). Never printed in chat, logs, or committed files. The user has
  pasted raw keys directly in chat twice, out of time pressure — acknowledged, keys were
  captured to `.env` immediately and never re-echoed.

## OpenRLHF

- Pinned to tag **v0.10.4**, commit `ad1796e62b56bc9deae95542778336f88d24a3ed`, vendored
  at `vendor/OpenRLHF` (gitignored — re-clone and re-check the pin on the server).
- Its hosted docs (`non_rl.html`, `agent_training.html`) show **flat** flag names
  (`--flag_name`); this pinned commit's actual argparse uses **dotted namespaces**
  (`--group.sub.flag`) instead. Always verify against the pinned commit's own source or
  `--help`, never copy a doc command verbatim. Full verified flag list:
  `artifacts/audits/openrlhf_commit_notes.md`.
- Vanilla GRPO in this codebase is NOT a separate script — it's
  `openrlhf/cli/train_ppo_ray.py` with `--algo.advantage.estimator group_norm` (choices
  include `gae, reinforce, rloo, reinforce_baseline, group_norm, dr_grpo`; `group_norm`
  is the DeepSeekMath-style one the spec wants — do not reach for `dr_grpo`).
  `--rollout.n_samples_per_prompt` must be `>1` for this estimator (the group size).
- Multi-turn agent hook is `--train.agent_func_path <script.py>`, confirmed present;
  reference implementation to study before writing our own:
  `vendor/OpenRLHF/examples/python/agent_func_openai_server_executor.py`.
- **OpenRLHF has zero references to `enable_thinking`** — it does not know about Qwen3's
  thinking toggle at all. We must apply `enable_thinking=False` ourselves in our own
  data-normalization step and record it in `RawGeneration.enable_thinking`, not rely on
  OpenRLHF's dataset loader.

## Qwen3 chat template / SFT masking

Implemented in `src/cmedalign/data/chat_template.py`, verified against the real
downloaded Qwen3-8B tokenizer (only tokenizer files, no model weights — a few MB, no GPU
needed). Three real footguns found and designed around, each would have caused a silent
bug if assumed instead of tested:

1. `apply_chat_template(..., return_assistant_tokens_mask=True)` silently returns an
   **all-zero mask** for Qwen3 (its template has no `{% generation %}` block;
   transformers only warns, never raises). Never use this flag for Qwen3.
2. With `enable_thinking=False`, the empty `<think>\n\n</think>\n\n` stub is inserted
   **only for the conversation's final message** — historical assistant turns embedded
   as context get no think wrapper at all (their thinking is stripped when re-serialized,
   matching how the model sees its own past turns at inference time). A naive
   `add_generation_prompt=True` render of `messages[:i]` for every turn `i` (treating each
   turn as if it were "the last one") gets this wrong for every non-final turn — caught by
   an actual failing test, not assumed. The fix scans the one true full-conversation
   tokenization and detects the stub per-occurrence instead of assuming it's always there.
3. `apply_chat_template(..., tokenize=True)` returns a `BatchEncoding` (dict), not a
   plain list, in the installed transformers version — `len()` on it silently returns the
   key count (2), not sequence length. Always pass `return_dict=False`.

Net masking rule: assistant content + that turn's closing `<|im_end|>` (so the model
learns to stop) are trainable; everything else, including any think-stub when present,
is masked with `IGNORE_INDEX = -100`.

**Still to verify on the server** (not assumed, explicitly deferred because it needs
either a GPU or is just faster to check there): OpenRLHF's own SFT
trainer/collator's assistant-only masking mechanism (`openrlhf/trainer/sft_trainer.py`
and whatever collator sits next to `openrlhf/datasets/sft_dataset.py` — a grep for
`-100`/`IGNORE`/`labels` in `sft_dataset.py` itself found nothing, so the real masking
logic likely lives elsewhere). Our own `chat_template.py` masking is independently
correct regardless of what OpenRLHF's default collator does — worst case we bypass
OpenRLHF's own masking and feed pre-masked `labels` directly.

## API adapter (`src/cmedalign/eval/api_adapter.py`)

- Generic OpenAI-chat-completions-compatible client. Credentials read only from
  `{ALIAS}_BASE_URL` / `{ALIAS}_API_KEY` / `{ALIAS}_MODEL` env vars, never hardcoded.
- Retries with exponential backoff only for transient errors (5xx, connection, timeout).
  A 4xx (e.g. content filter) is a real model result, surfaces immediately, never retried.
- Every response cached on disk keyed by a hash of the exact request (model + messages +
  decoding + seed) — a retried/crashed job never double-calls or double-bills the same
  request.
- Every log line is redacted for the configured API key before being emitted.
- Confirmed live for all three configured aliases as of 2026-07-19 (see `spec/tasks.md`
  and `STATUS.md` for the play-by-play):
  - `DEEPSEEK_V4_PRO` / `DEEPSEEK_V4_FLASH`: official DeepSeek platform
    (`https://api.deepseek.com/v1`), key format matched their standard `sk-`+32-hex
    pattern so we trusted the host; real server-side model IDs confirmed via a **free**
    `GET /v1/models` call (not guessed) to be `deepseek-v4-pro` and `deepseek-v4-flash`.
    Neither emits `<think>` tags.
  - `MINIMAX_M3`: the provided key's format (`sk-cp-...`) didn't match either DeepSeek's
    or MiniMax's own typical formats, so the base URL was NOT guessed — resolved once the
    user pointed at MiniMax's official docs. MiniMax exposes both an Anthropic-messages
    endpoint and a standard OpenAI-compatible one; used the OpenAI-compatible one
    (`https://api.minimaxi.com/v1`, model `MiniMax-M3`) to match our existing adapter
    instead of writing a second client. **Is** a thinking model as the user warned — real
    `<think>...</think>` block confirmed in output, correctly stripped by
    `human_pack.strip_thinking_trace`.
- Windows terminal note: this session's Bash/PowerShell console garbles Chinese text on
  print (a display codepage issue only). Always verify non-ASCII API output by writing to
  a UTF-8 file and reading it back, never trust the raw terminal echo.
- Also learned: in this sandboxed Bash tool, `python3` silently fails (exit 49, no
  output at all, even for `print('hello')`) while `python` works fine — use `python`,
  not `python3`, for any script invocation here.

## Reward design (`src/cmedalign/rewards/components.py`)

Weights frozen per spec: clinical 0.30, information 0.25, safety 0.25, process 0.10,
communication 0.10, minus capped turn-cost (max 0.20) and length-cost (max 0.15). Two
flat penalties (`RED_FLAG_MISSED_PENALTY = 0.60`, `ROLE_LEAKAGE_PENALTY = 0.60`) are
each larger than the combined max turn+length cost (0.35) by construction, so a policy
can never "buy back" a missed red flag or role leakage by being fast/terse — verified by
`test_red_flag_penalty_dominates_turn_cost`, not just asserted. Only the cost
*coefficients* are tunable on training-only dev data later; the positive weights and
penalty constants are not to be tuned against any eval signal.

## Human-eval blind packaging (`src/cmedalign/eval/human_pack.py`)

- Exact A–E position balance across the 80 cases achieved via Latin-square blocks (cyclic
  shifts of a random system permutation per block of 5 cases) — balance holds by
  construction, not by chance, and is tested for the exact expected count (16 per
  system per label over 80 cases / 5 systems).
- Blind map (case → label → system, the single most sensitive artifact before scores are
  frozen) is Fernet-encrypted at rest.
- Leak check (`build_blinded_cases`'s `forbidden_identifiers` argument) is proven to
  actually fire by a test that feeds it deliberately-leaky synthetic data and asserts it
  raises — not just present but exercised.
- `<think>` traces stripped via the same `strip_thinking_trace` used for MiniMax M3 output.

## Patient-agent boundary (`src/cmedalign/agents/patient.py`, `episode.py`)

- `build_patient_visible_profile()` strips scoring-only fields
  (`target_assessment`, `acceptable_actions`, `forbidden_claims`, `required_info`,
  `red_flags`, IDs) before anything is handed to whatever plays the patient — asserted in
  code, not just by convention.
- `build_doctor_visible_messages()` returns only prior conversation turns, never the
  profile object.
- Leak detection (`PatientAgent._check_leak`) flags if the patient's own reply contains
  the hidden target diagnosis or a forbidden claim substring.
- Consistency check is a ground-truth heuristic (currently: stated age vs
  `profile.demographics["age"]`) rather than turn-to-turn comparison — simple, testable,
  and catches the concrete "consistency fixture" case the spec asks for. Can be extended
  with more extractable facts later if needed.
- Episode termination: `FINAL:`-prefixed doctor utterance, or turn budget exhausted.
  Marked `invalid` (not scored) on any leak or contradiction.
- The actual `agent_func_path` script OpenRLHF needs is **not yet written** — needs a
  real LLM to play the patient (planned: frozen Qwen2.5-7B-Instruct per spec) and the
  vendored `agent_func_openai_server_executor.py` as a reference. This is GPU/server work.

## Stats (`src/cmedalign/stats/`)

- `bootstrap.py`: percentile bootstrap CI (`bootstrap_ci`), Holm-Bonferroni step-down
  correction (`holm_bonferroni`) — once one hypothesis fails its threshold in ascending-
  p-value order, every later (larger-p) hypothesis is also not rejected, regardless of
  its own p-value (tested explicitly, including a case where a later tiny-p test would
  pass on its own threshold but must still be blocked by the step-down rule).
- `tables.py`: `build_table_from_item_level` raises on any record missing a required
  field (`model`, `metric`, `score`, `item_id`) — no silent skip, no default-filling.
  `mark_best` requires an explicit orientation (`higher_is_better` /
  `lower_is_better`) per metric before it will bold anything, so an error-rate table
  can't accidentally get bolded backwards.

## What's deliberately NOT done yet (and why)

- No GPU/CUDA work at all — this machine has no GPU; G0 needs the rented server.
- No real datasets downloaded — user decision, wait for the server (bandwidth/storage,
  and G1's audit needs to run there anyway).
- `requirements.lock.txt` Section 2 (torch/vllm/deepspeed/ray/flash-attn) is an
  unverified best-effort draft — `make env-check` (G0) must confirm real versions on the
  rented hardware.
- `agent_func_path` GRPO implementation — needs a real patient LLM (server).
- OpenRLHF's own SFT collator internals — fast to check on the server, not blocking.
- User's own pre-existing multi-turn medical dialogue data — not yet transferred in;
  provenance/license status not yet confirmed (self-collected vs third-party), which
  determines whether it lands in `data/raw/` (cleared) or `data/quarantine/` (unclear).
