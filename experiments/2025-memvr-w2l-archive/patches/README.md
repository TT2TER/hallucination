# 补丁对照表

这些补丁从不同基线生成，适用于分别核对各项改动。

| 文件 | 对照基线 | 内容与用途 |
|---|---|---|
| `server-llava-tracked.patch` | LLaVA `c121f0432da27facab705978f83c4ada465e46fd` | 服务器已跟踪模型文件的 `cache_position` 参数修改；新增文件由 `server/LLaVA-overlay/` 保存 |
| `server-memvr-layout.patch` | MemVR `3cfbc6b5a97c926c287b7af5c8ef0d39a8a4f6c0` | 21 个工作区删除项，记录目录布局状态 |
| `server-mllms-know.patch` | mllms_know `a64c5d5a0ece793a2f7dc96e926d283a199033c1` | Notebook 代码、输出和元数据的原始差异 |
| `server-vs-memvr.patch` | 上述 MemVR 提交的三个原文件 | 与 LLaVA 中对应版本比较：注释和模型路径；路径前缀用于阅读对照 |
| `github-main-vs-llava.patch` | 上述 LLaVA 提交 | 本仓库 `a7f1fd8f…` 的 Python、Shell 和 `.gitignore` 完整差异，可用 `git apply` 在相应基线恢复 |
| `github-main-vs-memvr.patch` | 上述 MemVR 提交的 `memvr.py` | 本仓库核心回溯实现与所选 MemVR 基线的完整差异；路径前缀用于阅读对照 |
| `github-w2l-adaptation.patch` | 上述 mllms_know 提交的 `run.py`、`llava_methods.py`、`utils.py` | 本仓库对应三个文件的适配差异；路径前缀用于阅读对照 |
| `entropy-diff-trigger.patch` | 本仓库 `a7f1fd8f66fe1fef5184bdfdbfff8e8ea9dfa1ad` | 切换为 `d2aec1e9…` 的熵差触发逻辑 |
| `reverse-mask.patch` | 同上 | 切换为 `41a12256…` 的反向百分位筛选 |

完整提交哈希、快照文件哈希和来源映射在 `../evidence/` 与 `../CHECKSUMS.sha256`。保存的源码和补丁反映历史状态；恢复后应先按 README 调整模型、数据和本地代理路径，再验证执行环境。
