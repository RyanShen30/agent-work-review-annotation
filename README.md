# Agentic Work Review 自动标注框架

这个项目用于对于不同的coding agent问题使用不同的agent框架生成 的工作轨迹做自动预标注，产出结构化 JSON，之后供人工检查、修正，并沉淀成 GT。

当前可运行版本基于 Distilabel。第一版已经接入项目内现有的 DeNovoSWE 样例，默认样例路径是 `annotation/samples/*.json`。在9.9日组会之后，由予童搭建使用不同的agent框架结合由这不同docker环境的完整项目生成一条Long-Horizon的、完整的working traj。在这一个版本中，我们初步需要支持mini-swe-agents和openhands这两种高影响力并且简洁的coding-agent框架。

## 当前支持范围

当前代码只支持 **DeNovoSWE raw JSON**，也就是项目里 `annotation/samples/*.json` 这种原始结构。

还没有实现：

- 配置驱动的通用字段映射 adapter；
- 多数据集 adapter 注册；
- 按 trajectory 结构选择的多 parser；
- 可复用的通用 step parser；
- SWE-agent、OpenHands 等其他数据集的直接接入。

也就是说，现在不能把任意 SWE-agent/OpenHands JSON 直接丢进来跑。要支持新的数据源，仍然需要新增对应 adapter 和 step parser，然后在 `run.py` 里注册。

## 当前流程

```text
原始 JSON
-> DeNovoSWE Adapter
-> 确定性 step 切分
-> 拼 annotation prompt
-> Distilabel 调用模型
-> Pydantic 校验结构化 JSON
-> 每条样本保存一个 annotation JSON
```

Step 切分不由模型完成。DeNovoSWE 原始 `trajectory` 里已经有明确 step，因此当前 parser 会直接按原 step 一一映射成 canonical steps。

## 目录结构

```text
agentic_review_annotation_distilabel/
├── adapters/                 # 原始数据 -> 统一 Sample
│   ├── base.py
│   └── denovo.py
├── steps/                    # trajectory -> canonical steps
│   ├── base.py
│   └── denovo.py
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
annotation/samples/           # 当前 3 条 DeNovoSWE 原始样例
tests/                        # 基础单元测试
```

## 安装依赖

建议使用虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r agentic_review_annotation_distilabel/requirements.txt
```

之后运行命令时，推荐一直用 `.venv/bin/python`，这样不需要手动 `activate`。

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

## 快速自测

不调用真实模型，用 mock 跑完整流程：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner mock \
  --limit 3 \
  --overwrite
```

这个命令会验证：

- 能读取原始 JSON；
- 能适配 DeNovoSWE 字段；
- 能切出 canonical steps；
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
  --input annotation/samples/sample_002.json \
  --overwrite \
  --no-cache
```

如果模型经常输出被截断，可以适当增大输出 token：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner llm \
  --input annotation/samples/sample_002.json \
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
  "final_outcome": "correct",
  "failures": [
    {
      "step": 7,
      "reason": "The step introduced a correctness-relevant bug that was not fixed later.",
      "confidence": 0.9,
      "recovery": "unrecovered"
    }
  ],
  "metadata": {
    "model": "model-name",
    "prompt_version": "annotation_v1",
    "source_path": "annotation/samples/sample_001.json"
  }
}
```

字段约束：

- `final_outcome` 只能是 `correct` 或 `incorrect`；
- `failures[].step` 必须是真实存在的 `step_id`；
- `confidence` 是 0 到 1 的模型内部置信度；
- `recovery` 只能是 `unrecovered`、`self_corrected` 或 `unknown`。

## 当前 DeNovoSWE 字段适配方式

当前 adapter 根据真实样例字段做映射：

- `instance_id`：来自原始 `instance_id`；
- `task`：来自原始 `initial_messages`，保留 system/user message；
- `trajectory`：来自原始 `trajectory`，完整保留；
- `patch`：来自原始 `patch`，完整 diff 字符串；
- `evaluation`：来自 `success`、`score`、`finish_reason`、`error`、`difficulty`、`eval_result`。

Canonical step 当前只包含：

```json
{
  "step_id": 0,
  "content": {}
}
```

其中 `content` 是原始 step 对象的完整拷贝，不会强行拆成 reasoning/action/observation 的统一格式。

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

当前测试不依赖 pytest，可以直接用标准库 unittest：

```bash
.venv/bin/python -m unittest tests.test_distilabel_denovo_pipeline -v
```

## 已知边界

当前版本只正式支持 DeNovoSWE raw JSON。SWE-agent、OpenHands 等其他数据集还没有接入；后续需要先实现统一接口、多 parser 或通用 parser，再复用同一条 Distilabel annotation pipeline。

大样本真实模型标注会产生费用。正式批量跑之前，建议先用 `--runner mock` 检查流程，再用 `--input` 单条样例试跑真实模型。

## 9.9之后的计划
1. 能在调用api和本地部署模型的两种情况下使用mini-swe-agents和openhands两个开源的coding agents框架对于给定的问题进行产出模型的traj。
2. 将不同的agent框架产出的结果归一化为agent traj protocal的形式方便忆安后面的处理。
