# Blockers

Anything here stops the corresponding branch of work until a human decision is made.
Nothing is silently worked around. Empty section = nothing currently blocked.

## Open

1. **`med_zh_real`（导入自 rlhf_lab_cloud_kit 的最大单一数据源，58,038 条）许可证不明，
   已隔离到 `data/quarantine/rlhf_lab_cloud_kit/med_zh_real.jsonl`，未进入
   `data/raw/`。** 这是**许可证**问题，不是隐私问题——跟已解决的
   `internal_seed_flywheel`/`derived_from_seed` 隐私澄清是两码事，不能混为一谈。
   `rlhf_lab_cloud_kit` 的文档只记录它"来自你本地 F 盘"，没有记录具体许可证/来源
   （license_id 字段值是 `unknown`）。按项目规则"许可不明的数据先隔离，不得训练"，
   在你说清楚这批数据的真实来源和授权情况之前，不会把它移入 `data/raw/` 参与训练。
   **需要你回答**：这 58,038 条数据具体是从哪来的（自己整理的公开数据集镜像？第三方
   授权？还是别的）？如果是某个具体开源数据集，麻烦告诉我数据集名字/链接，我可以去核实
   许可证。

## Resolved

- **2026-07-19（最新）— `internal_seed_flywheel`/`derived_from_seed` 隐私问题已解决。**
  用户澄清：这两个数据源是完全虚构合成的（DeepSeek-V4-Pro 扮演患者、本项目基座模型
  Qwen3-8B 扮演医生作答，再由更强模型+真实医护人员修订），不是真实病历，不存在任何
  个人/机构隐私。详细决策记录见 `E:\cmedalign_paper\SOURCE_AUDIT.md` §6b（论文的
  可信来源）和 `E:\rlhf_lab_cloud_kit\BLOCKERS.md`。**现在可以正常导入这两个数据源到
  `data/raw/`**，但要在 license_ledger 里如实记录 `synthetic_or_real=synthetic`、
  `teacher_model=DeepSeek-V4-Pro`、`reviewed_by=stronger_model+clinical_staff` 这些
  溯源字段（这是论文自己的报告规则要求，不是隐私要求）。原来的记录（假设是真实数据时
  的分析）见下方历史记录，不再适用。
- ~~PII/de-identification status of `internal_seed_flywheel` and `derived_from_seed`
  (real business data) is unknown~~ — **已解决，见上一条**。原记录：Full detail in
  `E:\rlhf_lab_cloud_kit\BLOCKERS.md` #1 — the existing cleaning pipeline there only
  regex-flags (doesn't remove) phone/QQ/email hits for this source, and doesn't check
  names/ID numbers/addresses/record numbers at all.

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
