# Agentic Work Review 核心模块

这个目录包含 trajectory adapter、确定性 step 切分、annotation schema、prompt 和 Distilabel review pipeline。项目统一入口和用户配置位于仓库根目录。

## 支持的输入

- `mini_swe_agent`：默认路径，使用专用 message step parser；
- `openhands`：使用 event adapter；
- `denovo`：保留兼容 adapter 和单元测试。

处理流程：

```text
trajectory JSON -> adapter -> canonical steps -> prompt -> Distilabel -> schema 校验 -> annotation JSON
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
./scripts/run_agent_work_review.sh  # 仅 review 已有 trajectory
./scripts/run_pipeline.sh           # 生成并 review
```

## 路径

```text
config/example.yaml                         # 完整配置示例
scripts/run_mini_swe_agent.sh               # traj-only 用户参数
scripts/run_agent_work_review.sh            # review-only 用户参数
scripts/run_pipeline.sh                     # full 用户参数
output/traj/                                # traj-only 结果
output/annotation/normalized/               # review-only 标准化输入
output/annotation/preview/                  # review-only 人工预览
output/annotation/annotation/               # review-only 标注
output/pipeline/traj/                       # full 轨迹
output/pipeline/normalized/                 # full 标准化输入
output/pipeline/preview/                    # full 人工预览
output/pipeline/annotation/                 # full 标注
```

无效的模型输出会写入对应 annotation 目录下的 `_failed/`。

## mini-swe-agent 适配

字段映射：

- `instance_id` <- `instance_id` / `info.instance_id`
- `task` <- `problem` / `task` / `problem_statement` / 第一条 user message
- `trajectory` <- `messages`
- `patch` <- `info.submission` / `submission` / `generated_patch` / `model_patch` / `patch`
- `evaluation` <- `outcome.exit_status`、`info.exit_status`、`eval_result`、`eval_logs`、`model_stats`

每条 assistant message 与其后的 tool/user observations 会组成一个 `agent_turn`；开头的 system/user context 会放到第一个 step。

## 测试

从仓库根目录运行：

```bash
PYTHONPATH=. uv run --with pytest pytest -q
```

`mock` 只验证流程，不代表 review 质量。正式批量运行前建议先测试单个 instance。
