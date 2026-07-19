# Blockers

Anything here stops the corresponding branch of work until a human decision is made.
Nothing is silently worked around. Empty section = nothing currently blocked.

## Open

1. **PII/de-identification status of `internal_seed_flywheel` and `derived_from_seed`
   (real business data, 14,718 + 14,433 records, in `E:\rlhf_lab_cloud_kit`) is
   unknown.** Full detail in `E:\rlhf_lab_cloud_kit\BLOCKERS.md` #1 — the existing
   cleaning pipeline there only regex-flags (doesn't remove) phone/QQ/email hits for
   this source, and doesn't check names/ID numbers/addresses/record numbers at all.
   **This blocks importing those two specific sources into `data/raw/` here** (see
   `spec/tasks.md` Phase 2). The other 6 sources in that pool (`Huatuo26M-Lite`,
   `DISC-Med-SFT`, `Chinese-medical-dialogue`, `shibing624-finetune-zh`, `med_zh_real`,
   `gen_minimax_m3`) are open/synthetic and not affected by this blocker.

## Resolved

- **2026-07-19 — project path / API credentials / dataset download timing / remote OS
  assumption.** Not really a blocker (no science/licensing/privacy/cost risk), but flagged
  to the user before proceeding to avoid rework. Resolved via direct user confirmation:
  project at `E:\cmedalign`; API adapter built generically now, real DeepSeek/MiniMax
  credentials wired later; dataset downloads deferred to the rented server; assume
  standard Linux GPU rental. See STATUS.md log for 2026-07-19.

## Reminders for future blockers (do not delete)

Per the execution spec, these categories MUST stop and get written here, never silently
resolved:
- Anything that would change a scientific conclusion.
- Data licensing that is unclear or ambiguous.
- Anything touching patient privacy / identifiable data.
- Any action with large/uncontrolled API cost.
- Test-set contamination risk of any kind.
- Overwriting or deleting raw generations, judge outputs, event logs, or scores.
- Picking a checkpoint based on test results instead of the frozen train/dev composite.
