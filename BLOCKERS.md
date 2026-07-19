# Blockers

Anything here stops the corresponding branch of work until a human decision is made.
Nothing is silently worked around. Empty section = nothing currently blocked.

## Open

(none yet — local prep has not hit a scientific/licensing/privacy/cost decision point)

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
