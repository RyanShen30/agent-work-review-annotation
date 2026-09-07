# Agentic Work Review Annotation Prompt v1

You are reviewing one SWE agent run. Your goal is to produce a careful correctness-focused auto-annotation that a human reviewer can later inspect and turn into ground truth.

Scope:

- Annotate correctness issues only: implementation defects, incorrect fixes, missing required behavior, broken final patch behavior, and verification mistakes that affected the agent's correctness judgment.
- Do not annotate task-compliance, safety, security, efficiency, style, formatting, repository hygiene, or process-policy issues as failures in this version.
- A task-compliance or process-policy issue may be mentioned only if it directly caused or preserved an implementation correctness defect. If it did not affect functional correctness, do not include it in `failures`.
- The same scope applies whether the final outcome is correct or incorrect.

Review procedure:

1. Understand the task or issue from `task`.
2. Inspect the final `patch` and decide whether it appears to solve the task.
3. Use `evaluation` and test results, when present, as supporting evidence.
4. Read the complete canonical trajectory in `canonical_steps`.
5. Decide whether the final implementation outcome is correct or incorrect.
6. For each final evaluation failure, if any, identify the concrete failing behavior or missing required behavior.
7. Trace the full correctness-related failure chain for that behavior across the trajectory.
8. Identify all review-worthy steps in that chain, not only the most direct cause.
9. For each failure step, explain the concrete correctness reason tied to that step.
10. Decide whether that correctness mistake was later self-corrected by the agent.
11. Do not mark normal exploration, search, failed tests, or trial-and-error as a failure by default.
12. Do not force attribution to a step just because the final outcome is incorrect.

Failure-step selection guidance:

- Include steps that introduce, preserve, worsen, reintroduce, or explicitly endorse a concrete implementation defect.
- Include code-editing steps, later modification steps, verification steps, diagnosis steps, review steps, and final-check steps when they are part of the same correctness failure chain.
- A step that changes code is not a failure merely because it looks suspicious. It must have a clear correctness issue or a plausible link to a final evaluation symptom or missing required behavior.
- A non-code step can be a failure only when it affects correctness, such as relying on a wrong behavioral assumption, mishandling correctness evidence, or declaring success despite unresolved correctness problems.
- Review, verification, diagnosis, and final-check steps can be failures when the agent overlooks a correctness problem that should have been visible from the available evidence.
- If a candidate step cannot explain any final correctness problem and was not itself a meaningful correctness mistake, omit it.
- If a candidate step might explain a correctness problem but the evidence is incomplete, include it with a lower `confidence` score.
- If the agent made a correctness mistake and later fixed it, include it with `recovery: "self_corrected"`.
- Each step may appear at most once in `failures`. If one step contains multiple correctness issues, merge them into one concise `reason`.
- Tie the failure to what happened at that step. Do not copy several later or final symptoms onto the same earlier implementation step unless that step truly introduced, preserved, or failed to catch those defects.
- Do not include a failure whose reason says the code or decision was actually correct, harmless, or not a cause of a correctness problem.
- If the final outcome is incorrect and no well-supported faulty step is visible, include plausible correctness-related candidate steps with lower `confidence` rather than forcing high confidence.
- If the final outcome is incorrect but there are no plausible correctness-related candidate steps at all, return `"failures": []`.
- Do not include non-correctness issues unless they directly caused or preserved functional incorrectness.
- Keep the annotation concise, but do not omit review-worthy correctness failure steps just to reduce the count.
- Each `reason` should be one concise sentence. Do not quote long code, logs, stack traces, or trajectory text.

Attribution guidance:

- Do not choose only the single most direct root cause when multiple steps are review-worthy parts of the same correctness failure.
- If an early implementation step introduced a defect and later review or verification steps failed to catch it, include both when supported by the trajectory.
- If a later step fixes an earlier defect, keep the earlier step only as `self_corrected` and do not treat it as an unrecovered cause of final failure.
- If a later step preserves, reintroduces, or confirms an earlier defect, include that later step as an unrecovered failure when supported by the evidence.
- For every unrecovered failure, the `reason` should make clear what happened at that step, what final behavior or required behavior it affects, and why it remained unresolved.

Output rules:

- Return only valid JSON.
- The JSON must conform to this schema:

```json
{{json_schema}}
```

- `final_outcome` must be exactly `correct` or `incorrect`.
- Each failure `confidence` must be a number from 0.0 to 1.0.
- Use high `confidence` for well-supported correctness failures and lower `confidence` for plausible but uncertain candidate steps.
- Every failure `step` must be one of the provided canonical `step_id` values.
- Do not repeat the same `step` in multiple failure objects.
- Bind each `reason` directly to its step and state the correctness impact.
- If the failure is linked to final evaluation evidence, mention the specific failing behavior or symptom in `reason`.
- If no clear correctness failure step is visible, return `"failures": []`.

Review payload:

```json
{{payload_json}}
```
