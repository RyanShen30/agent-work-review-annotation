# Agentic Work Review 自动标注框架

这个项目用于对于不同的coding agent问题使用不同的agent框架生成 的工作轨迹做自动预标注，产出结构化 JSON，之后供人工检查、修正，并沉淀成 GT。

当前版本基于 Distilabel，默认处理 mini-swe-agent 生成的 trajectory，并可用 mini-swe-agent、OpenHands 或 OpenCollab 在 repository-level 任务上生成轨迹。

## 当前支持范围

当前支持四种输入：

- mini-swe-agent `messages`，当前默认路径；
- OpenHands `events`，暂保留；
- OpenCollab `trajectory.jsonl`，使用专用多 agent step parser；
- DeNovoSWE `trajectory`，仅保留兼容 adapter 和单元测试，不再提交原始样例数据。

各 adapter 将原始结果转换为统一 `Sample`，再进入同一条标注 pipeline。

## 当前流程

```text
原始 JSON
-> 对应 Adapter
-> 确定性 step 切分
-> 拼 annotation prompt
-> Distilabel 调用模型
-> Pydantic 校验结构化 JSON
-> 每条样本保存一个 annotation JSON
```

Step 切分不由模型完成。DeNovoSWE 保留原 step 编号；mini-swe-agent 按 assistant turn 切分；OpenHands 暂按事件切分；OpenCollab 以完成的 `llm_call` 为 step，并按 `aid` 归入工具结果和协作事件。

## 目录结构

```text
agentic_review_annotation_distilabel/
├── adapters/                 # 原始数据 -> 统一 Sample
│   ├── base.py
│   ├── denovo.py
│   ├── mini_swe_agent.py
│   ├── opencollab.py
│   └── openhands.py
├── agents/                   # 运行 mini-swe-agent / OpenHands / OpenCollab
├── steps/                    # trajectory -> canonical steps
│   ├── base.py
│   ├── agent.py
│   ├── denovo.py
│   ├── mini_swe_agent.py
│   └── opencollab.py
├── annotation/               # 输出 schema、prompt builder、专职 annotator 和 merger
│   ├── annotators.py
│   ├── merge.py
│   ├── prompt_builder.py
│   └── schema.py
├── pipelines/                # Distilabel pipeline
│   └── distilabel_pipeline.py
├── prompts/
│   └── annotation_v1.md
└── run.py

config/
├── example.yaml              # 完整配置示例
└── opencollab_team.yaml      # repository-level coding team 示例
scripts/
├── run_mini_swe_agent.sh     # traj-only 参数与入口
├── run_opencollab.sh         # OpenCollab team/agent traj-only 入口
├── run_agent_work_review.sh  # review-only 参数与入口
└── run_pipeline.sh           # full 参数与入口
output/
├── traj/                     # traj-only 轨迹
├── annotation/               # review-only 产物
└── pipeline/                 # full 模式轨迹及标注产物
tests/                        # 基础单元测试
```

## 安装依赖

克隆仓库后先下载全部 submodule：

```bash
git submodule update --init --recursive
```

建议使用虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r agentic_review_annotation_distilabel/requirements.txt
```

之后运行命令时，推荐一直用 `.venv/bin/python`，这样不需要手动 `activate`。

生成轨迹前，在选用的 `thirdparty` 子仓库中创建其 `.venv`。OpenCollab 使用官方 public SDK，当前 submodule 固定到 `v0.7.0` 对应提交。

OpenCollab 默认运行 `team` 模式。将 `generation.harness` 与 `review.dataset` 都设为 `opencollab`，并按需配置：

```yaml
generation:
  harness: opencollab
  harness_kwargs:
    mode: team
    provider: openai
    budget: 1000000
    team_config: config/opencollab_team.yaml
    use_worktrees: true

review:
  dataset: opencollab
```

若要做单 agent 对照，把 `mode` 改成 `agent`。runner 会把 OpenCollab 的 JSONL trace 和 team/agent manifest 嵌入统一的 trajectory JSON，再交给相同 review pipeline。

## 统一配置与运行

`config/example.yaml` 是完整示例且无需修改；用户直接修改对应 shell 脚本顶部的变量。API 凭证只从当前终端环境读取，不读取 `.env`：

```bash
export LLM_API_KEY=你的_Key
export LLM_BASE_URL=https://example.com/v1
```

生成与 review 阶段直接读取同一组变量；config 不保存凭证或 base URL，也不修改 thirdparty 代码。模型名称及其他运行参数在对应脚本顶部设置。

顶层 `mode` 支持 `traj-only`、`review-only`、`full`。三个脚本都显式使用 `config/example.yaml`，并用脚本中的变量覆盖全部相关参数：

```bash
./scripts/run_mini_swe_agent.sh     # traj-only
./scripts/run_opencollab.sh         # OpenCollab traj-only
./scripts/run_agent_work_review.sh  # review-only
./scripts/run_pipeline.sh           # full
```

统一入口也可以直接运行：

```bash
.venv/bin/python main.py --config config/example.yaml
```

通常直接修改脚本顶部变量即可。`INSTANCE` 支持 benchmark 行号、instance ID，或 `-1`（依次运行整个 benchmark）。如需临时覆盖，也可以直接调用入口：

```bash
.venv/bin/python main.py --config config/example.yaml --mode traj-only --instance 10
.venv/bin/python main.py --config config/example.yaml --mode review-only --input output/traj --runner mock
.venv/bin/python main.py --config config/example.yaml --mode full --instance astropy__astropy-14365
```

输出分别位于 `output/traj/`、`output/annotation/`、`output/pipeline/`。

## 快速自测

把 `scripts/run_agent_work_review.sh` 中的 `RUNNER` 改为 `mock`，并将 `INPUT` 指向一个已有 trajectory 文件或目录，然后运行：

```bash
./scripts/run_agent_work_review.sh
```

Mock 只检查 adapter、step parser、Distilabel pipeline 和输出保存流程，不代表真实标注质量。真实运行时将 `RUNNER` 改回 `llm`。

## 自动标注方式

自动标注阶段现在由四个独立的专职 annotator 完成，而不是一次 LLM 调用同时判断四个维度：

```text
canonical sample + canonical steps
-> TaskCompletionQualityAnnotator
-> SafetyPrivacyAnnotator
-> ReportingEvaluationIntegrityAnnotator
-> ExecutionEfficiencyAnnotator
-> deterministic merger
-> annotation.auto.step_reviews
-> human review / edit
-> annotation.final
```

四个 annotator 接收相同的 `canonical_steps` 和相同 step 编号。`TaskCompletionQualityAnnotator` 可以接收 `evaluation` 等私有评测证据辅助 GT 构造；其他三个 annotator 默认只接收 task、repo/environment 元数据、generated patch 和 canonical steps。

每个 annotator 只输出自己的 typed result：

- `CorrectnessAnnotationResult`：`label` 为 `pass` 或 `error`，只有 correctness error 可以带 `recovery: true`。
- `SafetyPrivacyAnnotationResult`：`label` 为 `pass` 或 `issue`。
- `ReportingIntegrityAnnotationResult`：`label` 为 `pass` 或 `issue`。
- `ExecutionEfficiencyAnnotationResult`：`label` 为 `pass` 或 `issue`。

随后 `annotation/merge.py` 用纯 Python 按 `step_id` 合并，并映射回兼容的 `StepReview`：correctness `error` -> `task_completion_quality.rating=fail`，安全/报告 `issue` -> `rating=fail`，效率 `issue` -> `execution_efficiency.rating=low`，正常效率为 `normal`。合并前会检查 step id 合法性、重复、缺失、reason/recovery 约束。

## 产物在哪里

按运行模式保存：

```text
output/traj/*.json                         # traj-only 原始轨迹
output/annotation/normalized/*.json        # review-only 标准化输入
output/annotation/preview/*.json           # review-only 人工预览
output/annotation/annotation/*.json        # review-only 最终标注
output/annotation/annotation/_failed/      # review-only 失败输出
output/annotation/public/*.json            # review-only public export
output/annotation/private/*.json           # review-only private export
output/pipeline/traj/*.json                # full 原始轨迹
output/pipeline/normalized/*.json          # full 标准化输入
output/pipeline/preview/*.json             # full 人工预览
output/pipeline/annotation/*.json          # full 最终标注
output/pipeline/annotation/_failed/        # full 失败输出
output/pipeline/public/*.json              # full public export
output/pipeline/private/*.json             # full private export
```

`normalized` 保存 master record，是 benchmark instance 的内部事实源；`preview` 仅供人工快速检查；`annotation` 保存兼容自动标注 JSON；`public` 默认不包含 step_reviews、oracle、resolved 或 evaluator logs；`private` 保存 grader/maintainer 需要的 evaluation、oracle、annotation.final 和 provenance。无效模型输出会写入对应 annotation 目录下的 `_failed/`。

## 输出 JSON 格式

`output/*/normalized/` 中的 master record 核心结构是：

```json
{
  "schema_version": "agent_work_review.master.v1",
  "instance_id": "example_id",
  "source": {
    "benchmark": "SWE-bench_Verified",
    "repo": "repo/name",
    "base_commit": "...",
    "problem_statement": "..."
  },
  "run": {
    "harness": "mini_swe_agent",
    "model": "model-name",
    "generated_patch": "diff --git ..."
  },
  "trajectory": {
    "raw_path": "path/to/raw.json",
    "raw_sha256": "...",
    "canonical_steps": []
  },
  "evaluation": {
    "resolved": null,
    "per_test_results": [],
    "eval_logs": null
  },
  "oracle": {
    "gold_patch": "diff --git ...",
    "test_patch": "diff --git ...",
    "fail_to_pass": [],
    "pass_to_pass": []
  },
  "annotation": {
    "auto": {
      "model": "model-name",
      "prompt_version": "annotation_v2_specialized",
      "step_reviews": []
    },
    "final": null
  },
  "provenance": {
    "source_path": "annotation/samples/mini_swe_agent_sample.json"
  }
}
```

`output/*/annotation/` 仍保存兼容的自动标注文件：

```json
{
  "instance_id": "example_id",
  "step_reviews": [
    {
      "step": 7,
      "task_completion_quality": {
        "rating": "warning",
        "reason": "The step made a plausible change but did not verify the relevant behavior.",
        "recovery": true
      },
      "safety_privacy": {
        "rating": "pass"
      },
      "reporting_evaluation_integrity": {
        "rating": "pass"
      },
      "execution_efficiency": {
        "rating": "normal"
      }
    }
  ],
  "metadata": {
    "model": "model-name",
    "prompt_version": "annotation_v2_specialized",
    "source_path": "output/traj/mini_swe_agent__example.json"
  }
}
```

字段约束：

- `step_reviews` 必须刚好覆盖每个真实存在的 `step_id`，不能漏也不能重复；
- `task_completion_quality`、`safety_privacy`、`reporting_evaluation_integrity` 的 `rating` 使用 `pass`、`warning`、`fail`、`unknown`；
- `execution_efficiency.rating` 使用 `high`、`normal`、`low`、`unknown`；
- `reason` 只在该维度确实有问题或证据不足时出现，正常 step 不写空 reason 或泛泛的正常说明；
- 只有 `task_completion_quality` 有 `recovery`，且只有真实 correctness/task-completion error 后续被修复时才写 `"recovery": true`。

## 当前 mini-swe-agent 字段适配方式

当前 adapter 根据 mini-swe-agent 保存的结果字段做映射：

- `instance_id`：优先来自 `instance_id`、`info.instance_id` 等字段；
- `task`：优先来自 `problem`、`task`、`problem_statement`，否则取第一条 user message；
- `trajectory`：来自 `messages`，完整保留；
- `run.generated_patch`：来自 `info.submission`、`submission`、`generated_patch`、`model_patch` 或 mini-swe-agent runner 顶层 `patch`；
- `oracle.gold_patch`：来自 SWE-bench `patch`，不要和 generated patch 混用；
- `oracle.test_patch`、`fail_to_pass`、`pass_to_pass`、`eval_type`、`eval_image`、`eval_script`、`log_parser`：来自 SWE-bench 对应字段，缺失时为空或 null；
- `evaluation`：来自 `outcome.exit_status`、`info.exit_status`、`eval_result`、`eval_logs`、`model_stats` 等。

MiniSWEAgent step parser 会把一条 assistant message 和其后的 tool/user observations 组成一个 `agent_turn`：

```json
{
  "step_id": 1,
  "raw_message_indices": [2, 3],
  "action_ids": ["call_1"],
  "observation_indices": [3],
  "content": {
    "type": "agent_turn",
    "agent_message": {},
    "actions": [],
    "observations": [],
    "context_messages": []
  }
}
```

这样 mini-swe-agent 的“跑轨迹”输出可以直接进入后续自动标注。

## OpenCollab step 切分

OpenCollab parser 把每条完成的 `llm_call` 作为一个 `opencollab_agent_turn`。同一 `aid` 后续产生的 `tool_exec` 会成为该 step 的 observation；`spawn*`、`message*`、`agent_*`、`worktree*` 等记录放入 `orchestration_events`；context shaping、terminal 和 retry 等放入 `runtime_events`。全局 topology 记录附在首个 step 的 `context_events`。这种切分允许不同 agent 并发交错，同时保留可审查的行动归属。

## 常用参数

- `--input`：输入 trajectory JSON 文件或目录；脚本默认使用 `output/traj`。
- `--limit N`：最多处理 N 条。
- `--start-index N`：从排序后的第 N 条开始跑。
- `--runner mock|llm`：mock 不调模型，llm 调真实模型。
- `--overwrite`：覆盖已有成功结果。
- `--no-cache`：不复用 Distilabel cache，改 prompt/schema 后建议加上。
- `--model-max-new-tokens N`：设置模型最大输出长度。
- `--compact-model-input`：只用于便宜调试，会截断发给模型的输入；正式标注不要用。

## 跑测试

```bash
PYTHONPATH=. uv run --with pytest pytest tests -q
```

## 已知边界

`INSTANCE=-1` 会顺序处理整个 SWE-bench。大样本运行和真实模型标注会产生费用，批量运行前建议先单条测试。
