# design.md — decisions actually made, organized by topic

This is the "why" companion to `spec/requirements.md` (the "what"). Organized by topic,
not chronologically — for the chronological blow-by-blow, see `STATUS.md`'s Log section.
If you're picking this project back up cold, read `spec/requirements.md` first, this
file second, then `spec/tasks.md` for exactly what's done vs pending.

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
