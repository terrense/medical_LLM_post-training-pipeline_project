# cmedalign

Reproducible Chinese medical multi-turn dialogue alignment experiment.

Training chain: `M0 Qwen3-8B (enable_thinking=False) → M1 SFT → M2 DPO → M3 vanilla GRPO`
(GRPO = DeepSeekMath Group Relative Policy Optimization, via OpenRLHF's
`--algo.advantage.estimator group_norm`). GSPO is related-work only unless core results
are fully archived and there is spare budget for an optional same-budget ablation.

See the full execution spec this project was scaffolded from for all rules, gates, and
deliverables (kept outside this repo — ask the project owner if you need the source doc).

## Status

This repo was bootstrapped on a **local Windows prep machine**, before a GPU server was
rented, specifically to do everything that does not require a GPU: schemas, parsers,
reward-function logic, API adapters, stats utilities, human-eval packaging, and pinning
OpenRLHF — all with unit tests against synthetic/mocked data. See `STATUS.md` for the
live state and `BLOCKERS.md` for anything stopped pending a decision.

**Nothing in this repo has touched a real GPU, a real dataset, or a real API key yet.**
`requirements.lock.txt` is a best-effort draft and MUST be re-verified with `make
env-check` (G0) on the actual rented hardware before anything else runs.

## Layout

```
configs/        data/sft/dpo/grpo/eval/model configs
src/cmedalign/  package: data, models, training, agents, rewards, eval, stats, plots
tests/          unit/integration/smoke pytest suites
scripts/        one-off / glue scripts invoked by the Makefile
data/           raw/normalized/splits/manifests/quarantine (gitignored, not this machine)
checkpoints/    m1_sft/m2_dpo/m3_grpo/probes (gitignored)
runs/           sft/dpo/grpo/eval run logs (gitignored)
artifacts/      manifests/audits/logs (audits tracked, logs gitignored)
results/        item-level raw results (tracked once real)
tables/ figures/ human_eval/ paper_support/
vendor/         OpenRLHF pinned checkout (gitignored, see artifacts/audits/openrlhf_commit_notes.md)
```

## Setup (local prep machine)

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install -r requirements.lock.txt   # CPU-only subset works here; GPU stack needs the server
pytest tests/unit -q
```

## Setup (rented GPU server, once SSH is available)

```bash
git clone <this repo> && cd cmedalign
make inventory      # G0 step 1: hardware/driver/CUDA inventory, no assumptions about GPU count/model
make env-check      # G0 step 2: verify requirements.lock.txt actually installs + CUDA/NCCL/vLLM smoke
make test           # full pytest incl. GPU-dependent suites
```

Then follow `Makefile` targets in order; see `STATUS.md` for where we left off.
