# Agentic Work Review 自动标注框架

这个目录里是当前可运行的 Distilabel 版本，用来对 SWE Agent trajectory 做自动预标注。

## 流程

```text
原始 JSON
-> Dataset Adapter
-> 确定性 step 切分
-> 拼 annotation prompt
-> Distilabel 调用模型
-> Pydantic 校验结构化 JSON
-> 保存 annotation JSON
```

默认支持的数据集是 mini-swe-agent，输入样例在项目根目录的 `annotation/samples/mini_swe_agent_sample.json`。

## 当前支持范围

当前代码支持三种输入注册：

- `mini_swe_agent`：默认路径，包含数据 adapter 和专用 step parser；
- `openhands`：保留 runner 和通用事件 adapter，这一轮不展开；
- `denovo`：保留兼容 adapter 和单元测试，不再提交 DeNovo 原始样例数据。

mini-swe-agent 的 runner 负责生成轨迹，`adapters/mini_swe_agent.py` 和 `steps/mini_swe_agent.py` 负责把轨迹接入自动标注。

## 快速开始

在项目根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r agentic_review_annotation_distilabel/requirements.txt
```

先用 mock 跑通流程：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner mock \
  --limit 3 \
  --overwrite
```

配置好 API key 后，用真实模型跑：

```bash
set -a
source .env
set +a

.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner llm \
  --limit 3 \
  --overwrite \
  --no-cache
```

## API 配置

支持 OpenAI-compatible 模型服务。优先读取环境变量：

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

`.env` 不会被 Python 自动加载，需要在终端里 `source .env`。同一个终端加载一次即可，关闭终端后需要重新加载。

也可以复制配置模板：

```bash
cp agentic_review_annotation_distilabel/config/config.example.yaml \
  agentic_review_annotation_distilabel/config/config.yaml
```

然后运行：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --config agentic_review_annotation_distilabel/config/config.yaml
```

## 运行产物

完整 normalized 输入：

```text
agentic_review_annotation_distilabel/data/normalized/
```

人工快速查看用 preview：

```text
agentic_review_annotation_distilabel/data/normalized_preview/
```

最终模型自动标注：

```text
agentic_review_annotation_distilabel/data/auto_annotations/
```

失败的原始模型输出：

```text
agentic_review_annotation_distilabel/data/auto_annotations/_failed/
```

## 输出格式

核心 annotation schema：

```json
{
  "instance_id": "...",
  "step_reviews": [
    {
      "step": 14,
      "task_completion_quality": {
        "rating": "unknown",
        "reason": "...",
        "recovery": "unknown"
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
  ]
}
```

保存到文件时还会附带 `metadata`，记录模型名、prompt 版本和源文件路径。

## mini-swe-agent 适配

当前 adapter 使用 mini-swe-agent 保存结果字段：

- `instance_id` <- `instance_id` / `info.instance_id`
- `task` <- `problem` / `task` / `problem_statement` / 第一条 user message
- `trajectory` <- `messages`
- `patch` <- `info.submission` / `submission` / `generated_patch` / `model_patch` / `patch`
- `evaluation` <- `outcome.exit_status`、`info.exit_status`、`eval_result`、`eval_logs`、`model_stats`

当前 step parser 将每条 assistant message 及其后的 tool/user observations 合成一个 `agent_turn`，并把开头的 system/user context 挂到第一个 step 上。

## 常用命令

跑单条：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner llm \
  --input annotation/samples/mini_swe_agent_sample.json \
  --overwrite \
  --no-cache
```

增大输出 token：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner llm \
  --input annotation/samples/mini_swe_agent_sample.json \
  --model-max-new-tokens 8192 \
  --overwrite \
  --no-cache
```

跑测试：

```bash
.venv/bin/python -m unittest tests.test_distilabel_denovo_pipeline -v
```

## 注意事项

- `mock` 只检查流程，不代表真实 review 质量。
- 改了 prompt 或 schema 后，建议加 `--no-cache`。
- 正式标注默认会把完整 task、patch、canonical steps 发给模型。
- `--compact-model-input` 只适合低成本调试，不适合正式标注。
- 大模型调用会产生费用，批量跑之前先单条试跑。
