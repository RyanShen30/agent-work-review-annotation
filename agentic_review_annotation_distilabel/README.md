# Agentic Work Review 自动标注框架

这个目录里是当前可运行的 Distilabel 版本，用来对 SWE Agent trajectory 做自动预标注。

## 流程

```text
原始 JSON
-> DeNovoSWE Adapter
-> 确定性 step 切分
-> 拼 annotation prompt
-> Distilabel 调用模型
-> Pydantic 校验结构化 JSON
-> 保存 annotation JSON
```

当前支持的数据集是 DeNovoSWE，输入样例在项目根目录的 `annotation/samples/*.json`。

## 当前支持范围

当前代码只支持 **DeNovoSWE raw JSON**，也就是项目里 `annotation/samples/*.json` 这种结构。

还没有实现：

- 配置驱动的通用字段映射 adapter；
- 多数据集 adapter 注册；
- 按 trajectory 结构选择的多 parser；
- 可复用的通用 step parser；
- SWE-agent、OpenHands 等其他数据集的直接接入。

因此现在不能把任意 SWE-agent/OpenHands JSON 直接放进来跑。要支持新的数据源，仍然需要新增对应 adapter 和 step parser，并在 `run.py` 里注册。

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
  "final_outcome": "correct",
  "failures": [
    {
      "step": 14,
      "reason": "...",
      "confidence": 0.9,
      "recovery": "unrecovered"
    }
  ]
}
```

保存到文件时还会附带 `metadata`，记录模型名、prompt 版本和源文件路径。

## DeNovoSWE 适配

当前 adapter 使用真实样例字段：

- `instance_id` <- `instance_id`
- `task` <- `initial_messages`
- `trajectory` <- `trajectory`
- `patch` <- `patch`
- `evaluation` <- `success`、`score`、`finish_reason`、`error`、`difficulty`、`eval_result`

当前 step parser 直接使用 DeNovoSWE 原始 trajectory 中的 `step` 编号。`content` 保留完整原始 step，不额外统一内部字段。

## 常用命令

跑单条：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner llm \
  --input annotation/samples/sample_002.json \
  --overwrite \
  --no-cache
```

增大输出 token：

```bash
.venv/bin/python -m agentic_review_annotation_distilabel.run \
  --runner llm \
  --input annotation/samples/sample_002.json \
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
