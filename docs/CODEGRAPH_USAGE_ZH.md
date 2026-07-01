# CodeGraph 使用文档

本文记录如何在本仓库使用 [colbymchenry/codegraph](https://github.com/colbymchenry/codegraph)，以及已经完成的本地初始化结果。

## 当前安装状态

本机已安装 CodeGraph CLI：

```bash
codegraph --version
# 1.0.1
```

当前仓库已经完成初始化：

```bash
cd /data/hdt/newtrackvla
codegraph init .
codegraph status .
```

当前索引结果：

- 项目路径：`/data/hdt/newtrackvla`
- 已索引文件：`86`
- 图节点：`2,499`
- 图边：`5,444`
- 数据库大小：`7.04 MB`
- 语言分布：`python 86`
- 索引目录：`.codegraph/`

`.codegraph/` 是本地生成的 SQLite 索引目录，已经加入 `.gitignore`，不建议提交到 Git。

## CodeGraph 做什么

CodeGraph 会把代码解析成一个本地知识图谱，包含文件、类、函数、方法、导入关系、调用关系等信息。它适合用来回答这些问题：

- 某个类/函数在哪里定义？
- 一个入口函数会调用哪些模块？
- 改某个符号可能影响哪里？
- 当前项目有哪些主要文件和符号？
- 让 Codex/Claude/Cursor 等 agent 少做重复的 `rg`、`find`、读文件探索。

本仓库是 Python 代码为主，CodeGraph 当前主要覆盖顶层训练/评估脚本、`tools/`、`multi_agent/` 和 `unrealzoo-gym/` 中的 Python 文件。

## 常用命令

查看索引状态：

```bash
codegraph status .
```

重新全量索引：

```bash
codegraph index . --force
```

增量同步：

```bash
codegraph sync .
```

查看项目文件树：

```bash
codegraph files --max-depth 2 --no-metadata
```

搜索符号：

```bash
codegraph query UnrealZoo --limit 20
codegraph query SingleAgent --limit 30
codegraph query AnchorDiffusion --limit 30
```

查看某个类/函数节点：

```bash
codegraph node UnrealZooSingleAgentPlanner
codegraph node AnchorDiffusionActionModel
```

探索一片功能区域：

```bash
codegraph explore --max-files 5 "drone single agent pipeline eval_unrealzoo_single_agent UnrealZooSingleAgentPlanner"
```

分析调用方：

```bash
codegraph callers UnrealZooSingleAgentPlanner
```

分析被调用方：

```bash
codegraph callees AnchorDiffusionActionModel
```

分析改动影响范围：

```bash
codegraph impact AnchorDiffusionActionModel
```

## 应用到当前仓库

### 1. 快速看仓库结构

```bash
codegraph files --max-depth 2 --no-metadata
```

当前 CodeGraph 看到的主要结构：

```text
eval.py
eval_unrealzoo_single_agent.py
eval_unrealzoo_multi_agent.py
model.py
model_unrealzoo_anchor_diffusion.py
train.py
train_unrealzoo_anchor_diffusion.py
tools/
multi_agent/
evt_bench/
unrealzoo-gym/
```

这适合在接手代码或定位入口时先跑一遍。

### 2. 分析 drone single-agent 训练评估流水线

入口脚本：

```bash
sh/run_drone_single_agent_pipeline.sh
```

该脚本按环境变量控制阶段：

- `RUN_ORGANIZE`：整理 drone 原始数据，调用 `tools/organize_drone_data.py`
- `RUN_SPLIT`：切分 train/test，调用 `tools/split_unrealzoo_single_agent_data.py`
- `RUN_PROCESS`：生成训练 JSONL，调用 `tools.make_tracking_data`
- `RUN_CACHE`：预缓存视觉特征，调用 `tools.precache_frames`
- `RUN_TRAIN`：训练，调用 `sh/train_with_estimate.sh`
- `RUN_EVAL`：评估，调用 `eval_unrealzoo_single_agent.py`
- 评估结束后调用 `tools.calculate_unrealzoo_single_agent_metrics`

推荐用下面的 CodeGraph 命令追踪相关符号：

```bash
codegraph query SingleAgent --limit 30
codegraph node UnrealZooSingleAgentPlanner
codegraph explore --max-files 5 "eval_unrealzoo_single_agent main UnrealZooSingleAgentPlanner predict action_from_waypoints"
```

当前索引显示 `UnrealZooSingleAgentPlanner` 位于：

```text
eval_unrealzoo_single_agent.py:122
```

主要成员：

- `__init__(self, args)`
- `reset(self)`
- `_encode(self, frame_bgr)`
- `predict(self, frame_bgr, instruction)`
- `action_from_waypoints(self, waypoints)`

调用关系提示：`UnrealZooSingleAgentPlanner` 由 `eval_unrealzoo_single_agent.py` 中的 `main` 调用。

### 3. 分析 Anchor Diffusion 模型

核心符号：

```bash
codegraph query AnchorDiffusion --limit 30
codegraph node AnchorDiffusionActionModel
codegraph explore --max-files 5 "AnchorDiffusionActionModel forward trajectory anchors"
```

当前索引显示 `AnchorDiffusionActionModel` 位于：

```text
model_unrealzoo_anchor_diffusion.py:497
```

主要成员：

- `__init__`
- `_normalize`
- `_denormalize`
- `_decode`
- `_gather_top_candidate`
- `forward`

这适合修改 anchor diffusion 训练、推理步数、候选轨迹选择或 loss 时先看影响面。

### 4. 查询要收窄

`codegraph explore` 的自然语言查询太宽时，会把相邻模块也带出来。例如查询 `drone single agent pipeline` 可能同时召回 `multi_agent/` 的训练数据类。建议按以下顺序逐步收窄：

```bash
codegraph query 关键词 --limit 30
codegraph node 精确类名或函数名
codegraph explore --max-files 3 "精确类名 函数名 文件名"
```

对本仓库尤其推荐把文件名、类名、函数名放在同一个查询里，例如：

```bash
codegraph explore --max-files 3 "eval_unrealzoo_single_agent UnrealZooSingleAgentPlanner predict"
codegraph explore --max-files 3 "tools.make_tracking_data drone action_field"
codegraph explore --max-files 3 "tools.precache_frames vision_cache cache_root"
```

## Codex MCP 接入

当前已经完成 CLI 安装和仓库索引，但没有自动修改 Codex 配置。如果希望 Codex 在之后的会话中直接拥有 CodeGraph MCP 工具，可以执行：

```bash
codegraph install --target=codex --location=global --yes
```

或者手动把下面片段加入 `/home/hdt/.codex/config.toml`：

```toml
[mcp_servers.codegraph]
command = "codegraph"
args = ["serve", "--mcp"]
```

修改 MCP 配置后需要重启 Codex 会话。若没有启用 MCP，仍然可以通过本文的 CLI 命令使用 CodeGraph。

## 日常维护

如果只是当前 shell 里手动查询，改代码后建议运行：

```bash
codegraph sync .
```

如果怀疑索引不准，运行：

```bash
codegraph index . --force
codegraph status .
```

如果以后不想在这个仓库继续使用 CodeGraph：

```bash
codegraph uninit .
```

这会删除 `.codegraph/` 本地索引目录，不会卸载全局 CLI。

如果要卸载全局 agent 配置：

```bash
codegraph uninstall
```
