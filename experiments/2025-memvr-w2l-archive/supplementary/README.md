# 补充的历史研究资料

2026-10-21 最后一次检查将范围扩展到三个项目目录以外，发现并补存以下资料。

## 新增的权重转换脚本

`standalone-transformers/` 保存两份原本位于独立 Transformers 4.40.0 目录下的新增脚本：

- `convert_auto.py`：遍历输入目录的子目录并安排并行转换。
- `convert_llava_weights_to_hf_local.py`：加载本地 safetensors 分片，转换参数名称，建立 HF 格式模型并保存。

这两份文件均不在服务器保留的原始 Transformers 4.40.0 ZIP 中。它们记录了本地权重转换方案，当前材料没有与之配套的成功执行日志。代码中“所有任务完成”“Conversion completed”等文字是打印语句。

脚本保留历史实现：例如本地加载器要求 `model.safetensors.index.json`，而盘点中的原始 LLaVA 权重为 `.bin` 分片。再次使用时需要核对输入格式、依赖版本和路径，然后另行验证转换。

## VS Code 的 27 个代码快照

`editor-history/` 保留编辑器保存的历史文件，并通过 `source-manifest.json` 记录其原始项目路径、时间戳及校验值。文件夹名和快照名沿用编辑器编号。

这些快照包含当时的路径调整、参数和模型实现尝试。`LLaVA/memvr.py` 的 `hk6Y.py`、`bYng.py` 尝试编码 `local_image` 并拼接视觉特征，`KV3L.py` 加入相应参数；它们属于编辑历史，当前工作区的最终文件是主 README 记录的另一份状态。

`editor-history/-29f932b5/E8Tr.py` 第 51 行保留了语法不完整的编辑状态。它与其他历史快照一起用于追溯过程，运行验证应基于已选定的完整版本。

## IPython 输入记录

`ipython-history.json` 保存 27 条输入，字段为会话号、行号和输入源码。它可用于回顾交互式调试过程。该文件的证据范围是保存过的输入，程序执行输出和运行状态需由其他材料补充。

## 来源与公开副本

`source-manifest.json` 为 29 份代码文件分别记录原始 SHA-256 与公开副本 SHA-256，另记录交互输入文件的两份校验值。公开副本把第三方个人目录示例的前缀统一为 `/cpfs/user/UPSTREAM_USER`；其他代码内容按导出快照保存。

这些补充资料与当前源码分目录保存，便于分别查看可恢复的工作区和过程中的草稿。
