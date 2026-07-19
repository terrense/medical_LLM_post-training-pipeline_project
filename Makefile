SHELL := /bin/bash
PY := python

.PHONY: inventory env-check data-download data-audit test smoke-sft train-sft \
        build-dpo smoke-dpo train-dpo build-profiles smoke-grpo train-grpo \
        eval-core eval-baselines human-pack statistics figures paper-values \
        validate archive

## --- G0: hardware / environment (GPU server only) ---

inventory:
	@mkdir -p artifacts
	bash scripts/system_inventory.sh | tee artifacts/system_inventory.txt

env-check:
	$(PY) scripts/env_check.py

## --- G1: data ---

data-download:
	$(PY) scripts/data_download.py --manifest configs/data/sources.yaml

data-audit:
	$(PY) -m cmedalign.data.audit --data-dir data/normalized --out artifacts/audits/data_audit.json

## --- tests (run everywhere; GPU-dependent suites are marked and skip without a GPU) ---

test:
	pytest tests/unit tests/integration -q
	@echo "NOTE: tests/smoke requires a GPU + model weights; run 'pytest tests/smoke -q' on the server."

## --- SFT (M1) ---

smoke-sft:
	bash scripts/train_sft.sh --config configs/sft/smoke.yaml

train-sft:
	bash scripts/train_sft.sh --config configs/sft/main.yaml

## --- DPO (M2) ---

build-dpo:
	$(PY) scripts/build_dpo_pairs.py --config configs/dpo/pairs.yaml

smoke-dpo:
	bash scripts/train_dpo.sh --config configs/dpo/smoke.yaml

train-dpo:
	bash scripts/train_dpo.sh --config configs/dpo/main.yaml

## --- GRPO (M3) ---

build-profiles:
	$(PY) scripts/build_patient_profiles.py --config configs/grpo/profiles.yaml

smoke-grpo:
	bash scripts/train_grpo.sh --config configs/grpo/smoke_32.yaml
	bash scripts/train_grpo.sh --config configs/grpo/smoke_200.yaml

train-grpo:
	bash scripts/train_grpo.sh --config configs/grpo/main.yaml

## --- Evaluation ---

eval-core:
	$(PY) -m cmedalign.eval.run --config configs/eval/core.yaml

eval-baselines:
	$(PY) -m cmedalign.eval.run --config configs/eval/baselines.yaml

## --- Human eval / stats / figures / paper (CPU-only, runs anywhere) ---

human-pack:
	$(PY) -m cmedalign.eval.human_pack --config configs/eval/human_pack.yaml

statistics:
	$(PY) -m cmedalign.stats.build_tables --results-dir results --out-dir tables

figures:
	$(PY) -m cmedalign.plots.build_figures --tables-dir tables --out-dir figures

paper-values:
	$(PY) -m cmedalign.stats.build_claim_macros --tables-dir tables --out paper_support/claim_values.tex

validate:
	$(PY) scripts/validate_deliverables.py

archive:
	bash scripts/archive.sh
