STAGE: local-prep (no GPU server rented yet) — GPU-independent groundwork complete
STATUS: PASS (for everything in scope of local prep; G0 itself cannot start without a GPU)
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
