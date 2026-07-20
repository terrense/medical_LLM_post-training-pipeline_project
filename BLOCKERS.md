# Blockers

Anything here stops the corresponding branch of work until a human decision is made.
Nothing is silently worked around. Empty section = nothing currently blocked.

## Open

（当前无阻塞——下方两项均已由用户 2026-07-20 决定解除。）

## Reminders（不是阻塞，是别忘了要跟进的事）

2. **CliMedBench 完整数据集（33,735题）确认拿不到，按用户指示不再纠结，直接按降级
   方案走。** 2026-07-20：用户说"数据集能用就行关键是质量要高，别的你别去纠结了"——
   决定：不联系作者、不再花时间找替代下载渠道。执行方式：按 `main.tex` 自己允许的
   "拿不到完整数据集就报告能拿到的子集"处理；如果最终真的一条 CliMedBench 数据都拿
   不到，这一项评测在论文里如实标 `pending`，不编数字、不强行凑。
3. **MedDG/IMCS-21 已解禁**（见下方 Resolved）——授权理由是"学术引用够了，别纠结许可证
   细节"，不是重新审查后认定许可证清晰，写论文引用这两个数据集时要如实说明许可状态
   （沿用它们各自论文的引用要求），不要写成"已确认开源许可"这种不准确的表述。

1. **`med_zh_real` 的确切来源还没定，用户已授权先用，等确认后要回来补全溯源。**
   2026-07-19：用户说这是"某开源数据集"，一时记不清具体名字，会在 1-2 天内去另一台
   电脑翻下载记录确认，让我"现在正常用"（已从 quarantine 移回 `data/raw/`，
   license_ledger 标记为 `unknown_but_authorized`）。**待办**：用户下次提到这批数据
   的具体来源时，回来把 `license_ledger.json` 和论文 `main.tex`/`SOURCE_AUDIT.md`
   里的溯源信息补全（现在论文数据表里还没有这个来源的行，因为具体是哪个数据集还不
   知道，等确认后要新增）。
   已做的内容质量审查（用户要求"你要审查"，不是走过场）：58,038 条，正则扫描
   广告/PII/危险用药建议/无保留断言诊断这几类问题，命中率分别为 0%/0%/0%/0.03%；
   人工抽读 6 条完整对话，内容合理、有恰当保留（建议就医、不给绝对诊断）。**一个
   值得记住的发现**：约 2.5%（1,459/58,038）的回答里出现"作为智能助手，我不能提供
   具体医疗建议"这类样板话术，说明这批数据里至少有一部分回答本身是 AI 生成的，
   不全是人类医生真实作答——等确认具体来源后，这一点需要在论文里如实反映，不能
   笼统写成"人类医生历史问诊记录"。

## Resolved

- **2026-07-20 — MedDG、IMCS-21 解禁。** 用户："数据集能用就行关键是质量要高，别的你
  别去纠结了"——授权按"学术引用即可使用"处理，不再等 Tianchi 平台条款/MedDG 作者
  回复。已从 `data/quarantine/` 移到 `data/raw/{meddg,imcs21}/`，license_ledger 标记
  `unknown_but_authorized`（跟 `med_zh_real` 同样的处理方式）。
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
