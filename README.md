# Agentic Work Review 自动标注框架

这个项目用于对于不同的coding agent问题使用不同的agent框架生成 的工作轨迹做自动预标注，产出结构化 JSON，之后供人工检查、修正，并沉淀成 GT。

当前版本默认处理 mini-swe-agent 生成的 trajectory，并可用 mini-swe-agent、OpenHands 或 OpenCollab 在 repository-level 任务上生成轨迹。Local review 使用 Distilabel；Docker review 可在 coding agent 的仓库副本中运行命令。

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
-> Distilabel（local）或仓库工具调用（docker）调用模型
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
├── pipelines/                # Local / Docker review pipeline
│   ├── distilabel_pipeline.py
│   └── review_docker.py
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

顶层 `mode` 支持 `traj-only`、`review-only`、`full`。四个脚本都显式使用 `config/example.yaml`，并用脚本中的变量覆盖相关参数。运行 review 的 `run_agent_work_review.sh` 和 `run_pipeline.sh` 可通过顶部的 `REVIEW_RUNTIME=local` 或 `REVIEW_RUNTIME=docker`进行控制。

`full` 模式会先检查 `output/pipeline/traj/` 中是否已有当前 harness、模型和 benchmark 实例的有效轨迹；存在时直接复用该文件并重新运行 review。需要重新生成轨迹时，删除对应的 JSON 文件后再运行脚本。

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

Mock 只检查 adapter、step parser、标注合并和输出保存流程，不会启动 Docker 容器，也不代表真实标注质量。真实运行时将 `RUNNER` 改回 `llm`。

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

### Review 运行环境

配置未指定 `review.runtime` 时默认为 `local`，沿用现有的 Distilabel 纯文本标注；两个 review 脚本当前都显式选用 `docker`。mini-swe-agent 的 Docker 运行结束后会先通过 `docker commit` 保存最终仓库快照，再删除 coding 容器。每个 annotator 从同一快照分别启动独立临时容器，并可通过 `run_command` 检索代码、编写临时测试和运行命令。Reviewer 容器在该 annotator 完成后删除；宿主仓库不会挂载，容器网络默认关闭。

```yaml
review:
  runtime: docker
  command_timeout: 120
  max_tool_calls: 12
  # docker_image: my-image:tag  # 旧轨迹缺少镜像信息时填写
  # docker_cwd: /testbed       # 旧轨迹工作目录不正确时填写
  # docker_platform: linux/amd64  # 其他单架构镜像可显式指定
```

新生成的 mini-swe-agent Docker 轨迹会在 `review_workspace` 中保存基础镜像、工作目录、base commit、最终快照标签和不可变 image ID。最终快照包含未跟踪文件、运行时安装的依赖和其他未进入 patch 的容器状态。Reviewer 优先使用该快照；快照不存在的旧轨迹会回退到“基础镜像 + `generated_patch`”重建。无法确定任何镜像时需设置 `review.docker_image`。

镜像名包含 `.x86_64.` 时，review 自动以 `linux/amd64` 启动；其他跨平台镜像可用 `review.docker_platform` 指定平台。

使用回退重建时，如果基础镜像包含晚于任务 `base_commit` 的初始化提交，review 会在临时容器中先回到 `base_commit`，再应用轨迹中的最终 diff；这不会修改镜像或宿主仓库。

`full` 模式默认使用 `cleanup_policy: on_success`：四个 Reviewer 全部完成后，按“最终快照在前、基础镜像在后”的顺序删除本次轨迹声明的受管镜像；review 失败时保留镜像以便排查。可选值为 `always`、`on_success`、`never`。`traj-only` 和 `review-only` 不自动删除镜像，便于分阶段运行和重复标注。

```yaml
cleanup_policy: on_success
generation:
  environment_kwargs:
    keep_image: true
    save_final_snapshot: true
```

Docker 模式不使用 Distilabel 的模型缓存。`review-only` 默认跳过已有 annotation，使用 `--overwrite` 可重跑；`full` 每次都会重跑 review，即使复用了已有轨迹。

测试已有轨迹时可运行：`.venv/bin/python main.py --config config/example.yaml --mode review-only --input output/traj/你的轨迹.json --set review.runtime=docker --overwrite`。

每个 annotator 只输出自己的 typed result：

- `CorrectnessAnnotationResult`：finding 使用 `warning`、`fail` 或 `unknown`，已恢复的问题可以带 `recovery: true`。
- `SafetyPrivacyAnnotationResult`：finding 使用 `warning`、`fail` 或 `unknown`。
- `ReportingIntegrityAnnotationResult`：finding 使用 `warning`、`fail` 或 `unknown`。
- `ExecutionEfficiencyAnnotationResult`：finding 使用 `high`、`low` 或 `unknown`。

每个专用 reviewer 返回 `review_complete: true`、一个整条轨迹的 `run_review`，以及只包含非默认步骤的稀疏 `findings`。前三个维度直接使用 `warning/fail/unknown`，遗漏步骤由 `annotation/merge.py` 补为 `pass`；效率直接使用 `high/low/unknown`，遗漏步骤补为 `normal`。合并器还会检查 instance id、非法或重复 step id、reason/recovery 约束，并输出兼容的完整 `StepReview` 和四个 run-level 结果。

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
      "prompt_version": "annotation_v3_sparse_run_level",
      "step_reviews": [],
      "run_reviews": null
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
  "run_reviews": {
    "task_completion_quality": {
      "rating": "warning",
      "reason": "The run recovered from a localized correctness problem."
    },
    "safety_privacy": {"rating": "pass", "reason": "No safety issue was found."},
    "reporting_evaluation_integrity": {"rating": "pass", "reason": "The report matches the available evidence."},
    "execution_efficiency": {"rating": "normal", "reason": "The run used a reasonable amount of work."}
  },
  "metadata": {
    "model": "model-name",
    "prompt_version": "annotation_v3_sparse_run_level",
    "source_path": "output/traj/mini_swe_agent__example.json"
  }
}
```

专用 Reviewer 的原始输出使用稀疏格式：

```json
{
  "instance_id": "example_id",
  "review_complete": true,
  "run_review": {
    "rating": "warning",
    "reason": "The run recovered from a localized correctness problem."
  },
  "findings": [
    {
      "step_id": 7,
      "rating": "warning",
      "reason": "The step made an incorrect assumption that was fixed later.",
      "recovery": true
    }
  ]
}
```

字段约束：

- 专用 Reviewer 的 `findings` 只能引用真实 `step_id`，不能重复；遗漏表示默认 `pass`，效率维度遗漏表示 `normal`；
- 合并后的 `step_reviews` 仍会完整覆盖每个真实 `step_id`；
- 每个专用 Reviewer 必须返回 `review_complete: true` 和带非空理由的 `run_review`；
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
