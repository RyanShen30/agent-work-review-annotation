# Agentic Work Review 自动标注框架

这个项目用于对于不同的coding agent问题使用不同的agent框架生成 的工作轨迹做自动预标注，产出结构化 JSON，之后供人工检查、修正，并沉淀成 GT。

当前版本基于 Distilabel，默认处理 mini-swe-agent 生成的 trajectory，并可用 mini-swe-agent/OpenHands 在 SWE-bench Docker 环境中生成轨迹。OpenHands 的 runner 和 adapter 先保留在仓库里，这一轮主要维护 mini-swe-agent 到自动标注的链路。

## 当前支持范围

当前支持三种输入：

- mini-swe-agent `messages`，当前默认路径；
- OpenHands `events`，暂保留；
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

Step 切分不由模型完成。DeNovoSWE 保留原 step 编号；mini-swe-agent 和 OpenHands 按消息或事件顺序生成 canonical steps。

## 目录结构

```text
agentic_review_annotation_distilabel/
├── adapters/                 # 原始数据 -> 统一 Sample
│   ├── base.py
│   ├── denovo.py
│   ├── mini_swe_agent.py
│   └── openhands.py
├── agents/                   # 运行 mini-swe-agent / OpenHands
├── steps/                    # trajectory -> canonical steps
│   ├── base.py
│   ├── agent.py
│   ├── denovo.py
│   └── mini_swe_agent.py
├── annotation/               # 输出 schema、prompt builder
│   ├── prompt_builder.py
│   └── schema.py
├── pipelines/                # Distilabel pipeline
│   └── distilabel_pipeline.py
├── prompts/
│   └── annotation_v1.md
├── config/
│   └── config.example.yaml
├── data/
│   ├── normalized/           # 运行后生成：完整 normalized 输入
│   ├── normalized_preview/   # 运行后生成：便于人工快速查看的 preview
│   └── auto_annotations/     # 运行后生成：最终自动标注 JSON
├── requirements.txt
└── run.py
```

项目根目录还有：

```text
annotation/samples/           # 小型 mini-swe-agent mock 样例
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

并且需要同时配置mini-swe-agent和openhand两个库对应的环境，分别在thirdparty的仓库目录下

## 配置 API Key

模型调用走 OpenAI-compatible 接口。支持这些环境变量：

```bash
AGENTIC_REVIEW_API_KEY
AGENTIC_REVIEW_MODEL
AGENTIC_REVIEW_BASE_URL
```

也兼容：

```bash
OPENAI_API_KEY
OPENAI_BASE_URL
```

如果项目根目录已有 `.env`，当前代码不会自动读取它，需要在同一个终端里加载一次：

```bash
set -a
source .env
set +a
```

同一个终端窗口里加载一次即可，后续多次运行不需要重复加载。关闭终端后需要重新加载。

OpenRouter 示例：

```bash
AGENTIC_REVIEW_BASE_URL=https://openrouter.ai/api/v1
AGENTIC_REVIEW_MODEL=minimax/minimax-m3:free
AGENTIC_REVIEW_API_KEY=你的_OpenRouter_Key
```

DeepSeek 示例：

```bash
AGENTIC_REVIEW_BASE_URL=https://api.deepseek.com
AGENTIC_REVIEW_MODEL=deepseek-chat
AGENTIC_REVIEW_API_KEY=你的_DeepSeek_Key
```

DeepSeek 默认会通过 `extra_body` 关闭 thinking，减少空输出和无效 token 消耗。确实想开启 thinking 时再加 `--enable-thinking`。

## 生成 SWE-bench trajectory

在根目录 `.env` 中设置 `LLM_API_KEY`，其余参数位于 `config/config.*.yaml`：

```bash
./scripts/run_mini_swe_agent_swe_bench.sh
./scripts/run_openhand_swe_bench.sh
```

结果保存到 `output/`。`environment_kwargs.keep_image` 控制任务结束后是否保留 Docker 镜像。

## 快速自测

不调用真实模型，用 mock 跑完整流程：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner mock \
  --limit 3 \
  --overwrite
```

这个命令默认读取 `annotation/samples/mini_swe_agent_sample.json`，会验证：

- 能读取原始 JSON；
- 能适配 mini-swe-agent 字段；
- 能用 MiniSWEAgent 专用 step parser 切出 canonical steps；
- 能走 Distilabel pipeline；
- 能生成并保存结构合法的 annotation JSON。

Mock 结果只用于检查流程，不代表真实标注质量。

## 使用真实模型运行

加载 `.env` 后运行：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner llm \
  --limit 3 \
  --overwrite \
  --no-cache
```

如果只想跑某一个样例，可以指定单个文件：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner llm \
  --input annotation/samples/mini_swe_agent_sample.json \
  --overwrite \
  --no-cache
```

如果模型经常输出被截断，可以适当增大输出 token：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner llm \
  --input annotation/samples/mini_swe_agent_sample.json \
  --model-max-new-tokens 8192 \
  --overwrite \
  --no-cache
```

## 产物在哪里

运行后会生成三类文件：

```text
agentic_review_annotation_distilabel/data/normalized/
```

完整 normalized 输入。这里会保留原始 `trajectory`，并额外包含 `canonical_steps`。

```text
agentic_review_annotation_distilabel/data/normalized_preview/
```

便于人工快速检查的精简 preview。它不用于模型标注，只是给人看结构。

```text
agentic_review_annotation_distilabel/data/auto_annotations/
```

最终自动标注 JSON。每条样本一个文件，文件名是 `{instance_id}.json`。

如果模型返回空内容或结构不合法，原始生成内容会保存到：

```text
agentic_review_annotation_distilabel/data/auto_annotations/_failed/
```

## 输出 JSON 格式

最终 annotation JSON 的核心结构是：

```json
{
  "instance_id": "example_id",
  "step_reviews": [
    {
      "step": 7,
      "task_completion_quality": {
        "rating": "warning",
        "reason": "The step made a plausible change but did not verify the relevant behavior.",
        "recovery": "self_corrected"
      },
      "safety_privacy": {
        "rating": "pass",
        "reason": "The step stayed within the provided repository context."
      },
      "reporting_evaluation_integrity": {
        "rating": "pass",
        "reason": "The step did not claim an unsupported result."
      },
      "execution_efficiency": {
        "rating": "normal",
        "reason": "The amount of exploration was reasonable for the task."
      }
    }
  ],
  "metadata": {
    "model": "model-name",
    "prompt_version": "annotation_v1",
    "source_path": "annotation/samples/mini_swe_agent_sample.json"
  }
}
```

字段约束：

- `step_reviews` 必须刚好覆盖每个真实存在的 `step_id`，不能漏也不能重复；
- `task_completion_quality`、`safety_privacy`、`reporting_evaluation_integrity` 的 `rating` 使用 `pass`、`warning`、`fail`、`unknown`；
- `execution_efficiency.rating` 使用 `high`、`normal`、`low`、`unknown`；
- 只有 `task_completion_quality` 有 `recovery`，使用 `not_applicable`、`unrecovered`、`self_corrected`、`unknown`。

## 当前 mini-swe-agent 字段适配方式

当前 adapter 根据 mini-swe-agent 保存的结果字段做映射：

- `instance_id`：优先来自 `instance_id`、`info.instance_id` 等字段；
- `task`：优先来自 `problem`、`task`、`problem_statement`，否则取第一条 user message；
- `trajectory`：来自 `messages`，完整保留；
- `patch`：来自 `info.submission`、`submission`、`generated_patch`、`model_patch` 或 `patch`；
- `evaluation`：来自 `outcome.exit_status`、`info.exit_status`、`eval_result`、`eval_logs`、`model_stats` 等。

MiniSWEAgent step parser 会把一条 assistant message 和其后的 tool/user observations 组成一个 `agent_turn`：

```json
{
  "step_id": 0,
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

## 常用参数

- `--input`：输入 JSON 文件或目录，默认 `annotation/samples`。
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

当前 runner 一次处理一个 SWE-bench 实例。大样本运行和真实模型标注会产生费用，批量运行前建议先单条测试。
