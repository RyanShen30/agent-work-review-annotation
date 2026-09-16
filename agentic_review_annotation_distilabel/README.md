# Agentic Work Review 核心模块

这个目录包含 trajectory adapter、确定性 step 切分、annotation schema、prompt 和 Distilabel review pipeline。项目统一入口和用户配置位于仓库根目录。

## 支持的输入

- `mini_swe_agent`：默认路径，使用专用 message step parser；
- `openhands`：使用 event adapter；
- `opencollab`：使用 JSONL trace adapter 和多 agent step parser；
- `denovo`：保留兼容 adapter 和单元测试。

处理流程：

```text
原始 JSON
-> Dataset Adapter
-> 确定性 step 切分
-> 四个专职 annotation prompt
-> Distilabel 独立调用四个 annotator
-> Pydantic 校验四个 typed result
-> deterministic merger
-> 更新 master record
-> 从 master 导出 public/private/auto annotation JSON
```

## 从根目录运行

API 只从终端环境读取：

```bash
export LLM_API_KEY=你的_Key
export LLM_BASE_URL=https://example.com/v1
```

用户修改对应脚本顶部的参数；`config/example.yaml` 只是完整配置示例：

```bash
./scripts/run_mini_swe_agent.sh     # 仅生成 trajectory
./scripts/run_opencollab.sh         # 仅生成 OpenCollab trajectory
./scripts/run_agent_work_review.sh  # 仅 review 已有 trajectory
./scripts/run_pipeline.sh           # 生成并 review
```

## 路径

```text
config/example.yaml                         # 完整配置示例
scripts/run_mini_swe_agent.sh               # traj-only 用户参数
scripts/run_opencollab.sh                    # OpenCollab traj-only 用户参数
scripts/run_agent_work_review.sh            # review-only 用户参数
scripts/run_pipeline.sh                     # full 用户参数
output/traj/                                # traj-only 结果
output/annotation/normalized/               # review-only 标准化输入
output/annotation/preview/                  # review-only 人工预览
output/annotation/annotation/               # review-only 标注
output/annotation/public/                   # review-only public export
output/annotation/private/                  # review-only private export
output/pipeline/traj/                       # full 轨迹
output/pipeline/normalized/                 # full 标准化输入
output/pipeline/preview/                    # full 人工预览
output/pipeline/annotation/                 # full 标注
output/pipeline/public/                     # full public export
output/pipeline/private/                    # full private export
```

无效的模型输出会写入对应 annotation 目录下的 `_failed/`。

## 输出格式

`output/*/normalized/` 中的 master record 是唯一事实源，核心结构是：

```json
{
  "schema_version": "agent_work_review.master.v1",
  "instance_id": "...",
  "source": {
    "benchmark": "SWE-bench_Verified",
    "repo": "...",
    "base_commit": "...",
    "problem_statement": "..."
  },
  "run": {
    "harness": "mini_swe_agent",
    "model": "...",
    "generated_patch": "..."
  },
  "trajectory": {
    "raw_path": "...",
    "raw_sha256": "...",
    "canonical_steps": []
  },
  "evaluation": {},
  "oracle": {
    "gold_patch": "...",
    "test_patch": "...",
    "fail_to_pass": [],
    "pass_to_pass": []
  },
  "annotation": {
    "auto": {
      "model": "...",
      "prompt_version": "annotation_v3_sparse_run_level",
      "step_reviews": [],
      "run_reviews": null
    },
    "final": null
  },
  "provenance": {}
}
```

`output/*/public/` 默认使用 `benchmark_task` 模式，只包含 reviewer 做题所需信息：`instance_id`、`problem_statement`、`repo`、`base_commit`、必要环境信息、agent/model 元数据、`canonical_steps` 和 `generated_patch`。默认 public export 不包含 `step_reviews`、gold patch、test patch、FAIL_TO_PASS、PASS_TO_PASS、resolved 或 evaluator logs。

`output/*/private/` 包含 evaluation、oracle、`annotation.final` 和必要 provenance/audit 信息。公开分析集需要显式调用 `export_public(..., mode="annotation_release")` 才会包含 final step annotation。

四个专用 Reviewer 分别返回 `review_complete: true`、带理由的 `run_review` 和稀疏 `findings`。前三个维度只在步骤为 `warning/fail/unknown` 时写 finding，遗漏步骤合并为 `pass`；效率只写 `high/low/unknown`，遗漏步骤合并为 `normal`。合并后保存的 auto annotation schema：

```json
{
  "instance_id": "...",
  "step_reviews": [
    {
      "step": 14,
      "task_completion_quality": {
        "rating": "unknown",
        "reason": "..."
      },
      "safety_privacy": {
        "rating": "unknown",
        "reason": "..."
      },
      "reporting_evaluation_integrity": {
        "rating": "unknown",
        "reason": "..."
      },
      "execution_efficiency": {
        "rating": "unknown",
        "reason": "..."
      }
    }
  ],
  "run_reviews": {
    "task_completion_quality": {"rating": "unknown", "reason": "..."},
    "safety_privacy": {"rating": "pass", "reason": "..."},
    "reporting_evaluation_integrity": {"rating": "pass", "reason": "..."},
    "execution_efficiency": {"rating": "normal", "reason": "..."}
  }
}
```

保存到文件时还会附带 `metadata`，记录模型名、prompt 版本和源文件路径。

`reason` 只在该维度确实有问题或证据不足时出现；正常 step 不写空 reason 或泛泛的正常说明。`recovery` 只属于 `task_completion_quality`，且只有真实 correctness/task-completion error 后续被修复时才写 `"recovery": true`。

## 专职 annotator

自动标注阶段包含四个独立 LLM 调用：

- `TaskCompletionQualityAnnotator` 只输出 `CorrectnessAnnotationResult`，判断 `task_completion_quality` 的 `pass/error` 和可选 `recovery: true`。
- `SafetyPrivacyAnnotator` 只输出 `SafetyPrivacyAnnotationResult`，判断 `safety_privacy` 的 `pass/issue`。
- `ReportingEvaluationIntegrityAnnotator` 只输出 `ReportingIntegrityAnnotationResult`，判断 `reporting_evaluation_integrity` 的 `pass/issue`。
- `ExecutionEfficiencyAnnotator` 只输出 `ExecutionEfficiencyAnnotationResult`，判断 `execution_efficiency` 的 `pass/issue`。

`annotation/merge.py` 会按 `step_id` 确定性合并四份 typed result，并映射回兼容的 `StepReview`。合并前会拒绝重复 step、未知 step、缺失 step、pass 项 reason、非 correctness recovery，以及 `recovery` 出现在非 error correctness step 的情况。

## mini-swe-agent 适配

字段映射：

- `instance_id` <- `instance_id` / `info.instance_id`
- `task` <- `problem` / `task` / `problem_statement` / 第一条 user message
- `trajectory` <- `messages`
- `run.generated_patch` <- `info.submission` / `submission` / `generated_patch` / `model_patch` / runner 顶层 `patch`
- `oracle.gold_patch` <- SWE-bench `patch`
- `oracle.test_patch`、`fail_to_pass`、`pass_to_pass`、`eval_type`、`eval_image`、`eval_script`、`log_parser` <- SWE-bench 对应字段
- `evaluation` <- `outcome.exit_status`、`info.exit_status`、`eval_result`、`eval_logs`、`model_stats`

每条 assistant message 与其后的 tool/user observations 会组成一个 `agent_turn`；开头的 system/user context 会放到第一个 step。

## OpenCollab 适配

runner 通过 OpenCollab public SDK 运行 `team`（默认）或 `agent`，读取其 `trajectory.jsonl` 和 manifest 后保存成单个统一 JSON。parser 以 `llm_call` 为 step 边界，并按 `aid` 将 `tool_exec`、消息、委派和生命周期事件归回对应 agent turn，因此可以处理并发交错的 team trace。

## 测试

从仓库根目录运行：

```bash
PYTHONPATH=. uv run --with pytest pytest -q
```

`mock` 只验证流程，不代表 review 质量。正式批量运行前建议先测试单个 instance。
改了 prompt 或 schema 后，建议加 `--no-cache`。正式标注默认会把完整 task、generated_patch、canonical steps 发给模型；`--compact-model-input` 只适合低成本调试。
