# OpenRLHF pinned commit — verified CLI surface

Generated locally (Windows prep box) by reading the actual argparse source at the pinned
commit, NOT by copying the hosted docs. The hosted docs
(`non_rl.html`, `agent_training.html`) describe a different/older flag naming convention
(flat `--flag_name`) than this pinned commit, which uses **dotted namespaced flags**
(`--group.sub.flag`). Do not copy example commands from the docs verbatim — translate
flag names against this file or re-grep the pinned commit first.

## Pin

- Repo: https://github.com/OpenRLHF/OpenRLHF
- Tag: `v0.10.4`
- Commit: `ad1796e62b56bc9deae95542778336f88d24a3ed`
- Tagged: 2026-06-08
- Vendored at: `vendor/OpenRLHF` (submodule-free plain clone, checked out detached at this tag)
- Action item once on the GPU server: re-run `git -C vendor/OpenRLHF log -1` to confirm
  the working copy still matches this hash before any training run, and re-run
  `python openrlhf/cli/train_sft.py --help` / `train_dpo.py --help` / `train_ppo_ray.py --help`
  (these do work as real `--help` since it's a real argparse parser) to catch any drift if
  the server's OpenRLHF checkout differs from this one.

## SFT (`openrlhf/cli/train_sft.py`)

Key verified flags (dotted namespace, see full list via grep):
- `--model.model_name_or_path`
- `--data.dataset`, `--data.dataset_split` (default `train`), `--eval.dataset`
- `--train.batch_size` (global), `--train.micro_batch_size` (per GPU), `--train.max_epochs`, `--train.seed`
- `--model.gradient_checkpointing_enable`
- `--ds.zero_stage`, `--ds.adam_offload`, `--ds.lora.rank` (0 = disabled), `--ds.lora.alpha`,
  `--ds.lora.target_modules` (default `all-linear`), `--ds.lora.dropout`
- `--optim {adam,muon}`, `--adam.lr` (default 5e-6), `--adam.betas`, `--adam.weight_decay`
- `--lr_scheduler` (default `cosine_with_min_lr`), `--lr_warmup_ratio`, `--min_lr_ratio`
- `--max_norm` (grad clip)
- `--ckpt.output_dir`, `--ckpt.path`, `--ckpt.save_hf`, `--ckpt.save_steps`
- `--model.pretrain_mode_enable` (switches to non-chat pretrain loss — NOT what we want, leave off)

Spec's suggested LoRA start config (rank 64, alpha 128, dropout 0.05, attn+MLP linear,
bf16, max_len 4096, gradient checkpointing, assistant-only loss) maps to:
`--ds.lora.rank 64 --ds.lora.alpha 128 --ds.lora.dropout 0.05 --ds.lora.target_modules all-linear
--model.gradient_checkpointing_enable`. Assistant-only loss masking is controlled by the
dataset/collator (see `openrlhf/datasets/sft_dataset.py`, uses `apply_chat_template`) —
confirm the exact masking flag name once reading that file in full on the server; grep
`multiple_turns`/`input_template` there before assuming.

## DPO (`openrlhf/cli/train_dpo.py`)

- `--model.beta` (default 0.1) — matches spec's smoke-start beta 0.1
- `--model.label_smoothing`, `--model.nll_loss_coef` (Llama-3.1-style NLL regularization, optional)
- `--adam.lr` (default 1e-5; spec wants 5e-6 for smoke start — pass explicitly)
- `--ds.lora.rank/alpha/target_modules/dropout` — same shape as SFT
- `--data.dataset`, `--data.dataset_split`, `--eval.dataset`

## GRPO — via `openrlhf/cli/train_ppo_ray.py`, NOT a separate script

There is no standalone "GRPO" CLI. Vanilla GRPO is PPO-family code with the advantage
estimator switched:

- `--algo.advantage.estimator group_norm` — this is what makes it (vanilla) GRPO in this
  codebase. Choices are `gae, reinforce, rloo, reinforce_baseline, group_norm, dr_grpo`.
  **`group_norm` is the one to use for vanilla GRPO** per the spec (DeepSeekMath-style group-
  relative advantage). Do NOT reach for `dr_grpo` unless explicitly asked — that's a
  different published variant.
- `--rollout.n_samples_per_prompt` must be `> 1` when `estimator != gae` (asserted in code,
  line ~610). Spec's smoke-start group size 4 (scale to 8 if VRAM allows) → set this flag.
- `--algo.kl.use_loss` (bool) — "whether to use KL loss from GRPO" per the code's own help text.
- `--algo.kl.init_coef` (default 0.01) — matches spec's smoke-start KL beta 0.01.
- `--actor.eps_clip` (default 0.2) — matches spec's smoke-start clip 0.2.
- `--actor.eps_clip_low_high` — optional asymmetric clip, leave default (mirrors eps_clip) unless tuning.
- When `algo.advantage.estimator != "gae"`, the critic model is disabled automatically
  (`args.critic.model_name_or_path = None` — see code around line 601) — expected for GRPO,
  do not fight this by trying to force a critic in.
- Multi-turn agent: `--train.agent_func_path <path.py>` — points at a Python file
  implementing the env step/reward interface. Confirmed present in this pinned commit
  (also referenced in `openrlhf/trainer/ray/vllm_engine.py` and two example scripts:
  `examples/python/agent_func_openai_server_executor.py` and
  `examples/python/vlm_multiturn_agent.py`). When this flag is set, the code auto-sets
  `args.reward.remote_url = "agent"` — i.e. reward computation is expected to come from
  the agent function, not a separate reward model. Read
  `examples/python/agent_func_openai_server_executor.py` in full on the server before
  writing our patient-agent `agent_func_path` implementation — it's the closest existing
  reference for the interface shape (reset/step/reward/done).
- vLLM rollout/train colocate-relevant flags: `--vllm.sync_backend` (default nccl),
  `--vllm.sync_with_ray`, `--vllm.enable_prefix_caching`, `--vllm.enforce_eager`.
- Reward shaping extras available if useful later: `--reward.overlong_buffer_len` /
  `--reward.overlong_penalty_factor` (length penalty), `--reward.stop_properly_penalty_coef`,
  `--reward.clip_range` (default -10,10).

## Qwen3 chat template — verified directly against the real tokenizer (not assumed)

Downloaded only the Qwen3-8B tokenizer files (no weights) and probed `apply_chat_template`
directly. Full implementation + tests: `src/cmedalign/data/chat_template.py`,
`tests/unit/test_chat_template.py` (6 tests, all passing against the real tokenizer,
offline via HF cache). Three findings that would have caused silent bugs if assumed
instead of verified:

1. `return_assistant_tokens_mask=True` silently returns an all-zero mask for Qwen3 (no
   `{% generation %}` block in its template; transformers only warns, doesn't raise).
   Do not use it for Qwen3 loss masking.
2. With `enable_thinking=False`, the empty `<think>\n\n</think>\n\n` stub is inserted
   **only for the conversation's final message**, not for every historical assistant
   turn. A naive per-turn `add_generation_prompt=True` on `messages[:i]` gets this wrong
   for every non-final turn (verified by the mismatch, then fixed) — the working
   implementation scans the one true full-conversation tokenization and detects the stub
   per-occurrence instead of assuming it's always there.
3. `apply_chat_template(..., tokenize=True)` returns a `BatchEncoding` (dict), not a
   plain list, in the installed transformers version — pass `return_dict=False`.

Net effect for SFT masking: assistant content + that turn's closing `<|im_end|>` are
trainable; everything else (including any think-stub, when present) is masked. This is
implemented and unit-tested now; only the OpenRLHF-side collator wiring (item 2 in the
"still to verify" list below) remains to confirm on the server.

## Things to verify for real once on the GPU server (do not assume from this file alone)

1. Whether `enable_thinking` is threaded through anywhere in OpenRLHF's chat-template
   application — a repo-wide grep here found **zero** references to `enable_thinking` in
   this pinned commit. This means OpenRLHF's dataset/tokenizer code does NOT know about
   Qwen3's thinking toggle — we must apply
   `tokenizer.apply_chat_template(..., enable_thinking=False)` ourselves in our own
   data-normalization step *before* handing text to OpenRLHF's dataset loader, and record
   that flag in the normalized-record provenance (per spec G2). Do not rely on OpenRLHF
   to do this for us.
2. The exact assistant-only masking mechanism in `openrlhf/datasets/sft_dataset.py` —
   only skimmed via grep for `apply_chat_template`, not read in full yet (deferred to
   server session since it's fast to do there and this note is already actionable).
   Note: grepping that file for `-100`/`IGNORE`/`labels` found nothing, so masking
   (if any beyond prompt/response split) likely lives in the collator or trainer step,
   not the dataset class — check `openrlhf/trainer/sft_trainer.py` and the collator
   next to `sft_dataset.py` first, not just the dataset file itself.
3. Whether `pip install` of this pinned commit's `requirements.txt` actually resolves
   cleanly against the CUDA/driver on whatever GPUs get rented — this is exactly what
   `make env-check` / G0 is for; do not assume the version pins in our
   `requirements.lock.txt` draft are correct until that gate passes.
