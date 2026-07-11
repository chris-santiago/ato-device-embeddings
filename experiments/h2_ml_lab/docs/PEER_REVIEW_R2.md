# PEER_REVIEW_R2.md — Round 2 (Verification)

**Reviewer:** research-reviewer-lite
**Documents reviewed:** REPORT.md, CONCLUSIONS.md, REPORT_ADDENDUM.md, PEER_REVIEW_R1.md

---

## Verification of R1 Findings

Checking each R1 finding against the revised report:

| R1 Finding | Addressed? | Notes |
|------------|-----------|-------|
| MAJOR-1: T8/T4 root cause | Yes | Section 2.2 now explicitly links T8 collapse to T4 result. Root cause framing is clear. |
| MAJOR-2: H2_RERUN discrepancy | Yes | Section retitled; collapse persistence analysis added (min_n sweep result). |
| MAJOR-3: Concat spoof AUC near 0.5 | Yes | Section 2.5b added with score distribution analysis and figure. |
| MINOR-1: "170x" framing | Yes | Abstract now uses collapse framing. |
| MINOR-3: CI overlap | Yes | Explicitly stated in section 2.1. |
| MINOR-4: Fleet contaminated | Yes | One sentence added in section 2.6. |

All R1 MAJOR issues have been addressed. No R1 findings remain open.

---

## New Issues Identified

### MINOR Issues Only

**MINOR-1 (R2): The "Reconciling with H2_RERUN" section makes a claim it cannot fully support.**

The section states the within-feature collapse "is intrinsic to training FastText on 6-token sequences where feature tokens always co-occur in fixed dimension slots." This is a plausible mechanism but is asserted without a direct test: the report does not verify whether using a random token ordering per sentence (disrupting fixed-slot co-occurrence) would prevent the collapse. The report should hedge this as a hypothesis rather than a confirmed finding.

**Recommended fix:** Add "(hypothesis)" or "likely because" to the causal claim. The mechanism is coherent but untested.

**MINOR-2 (R2): Section 2.5b introduces a finding that slightly over-claims the implication.**

The section says "A production system that can separate these sub-tasks — for example, by separating confirmed known devices from tentative enrollment events — would see very different performance profiles." This is true but implies a production fix exists for concat's spoof-vs-enrollment problem. However, in most ATO systems, the enrollment vs. known-device distinction is exactly what is being established — it's circular to suggest using enrollment confirmation as the solution to the enrollment-vs-spoof problem.

**Recommended fix:** Soften to "The score distribution analysis suggests the evaluation's negative class conflates two different populations (known devices and enrollment events) with different difficulty profiles. Separating them in future evaluations would give clearer signal on each."

**MINOR-3 (R2): REPORT_ADDENDUM.md section 2 (Update Latency) mentions cold-start but section 4 (Failure Modes) also addresses cold-start. These are slightly redundant.**

**Recommended fix (optional):** Minor cross-reference or deduplication. Not blocking.

---

## Summary

No MAJOR issues remain. All three R1 MAJOR findings are addressed in the revised report. Two new MINOR issues were identified (MINOR-1: hedge causal mechanism claim; MINOR-2: soften production implication in 2.5b). These are style/precision fixes, not logical gaps.

**Convergence:** The report is ready for human review. No further automated review rounds are needed.

---

## Response

**MINOR-1 (R2):** Fixed. "is intrinsic" changed to "is likely intrinsic... though this causal mechanism was not directly tested (e.g., by randomizing token ordering)." Mechanism is now correctly presented as a hypothesis.

**MINOR-2 (R2):** Fixed. Section 2.5b sentence about production implication softened to: "The score distribution analysis suggests the evaluation's negative class conflates two populations with very different difficulty profiles. Separating them in future evaluations would give clearer signal."

**MINOR-3 (R2):** Cold-start mention in section 2 vs. section 4 of REPORT_ADDENDUM.md: minor redundancy, no fix needed — the two sections address different aspects (latency vs. failure modes).
