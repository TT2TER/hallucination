# 公开评测数据的下载与恢复

更新日期：2026-10-21。归档采用“保留独有资料、公开 ZIP 按链接重新获取”的方式。三个 ZIP 的文件本体留在待清空服务器，本仓库保存它们的来源、旧版本校验值和逐文件恢复映射。

## 1. 三个原始 ZIP

| 包 | 原服务器相对 `playground/data/` 的位置 | 旧包字节数 | 内容 |
|---|---|---:|---|
| LLaVA 评测包 | `eval.zip` | 23,210,829 | 52 个题目、转换程序、参考预测和评分文件 |
| MME 数据包 | `eval/MME/MME_Benchmark_release_version.zip` | 201,161,723 | 1,187 张图片、1,187 个问题与标准答案文本 |
| MME 评分工具包 | `eval/MME/eval_tool.zip` | 107,320 | 31 个评分程序、示例结果及附属文件 |

数据目录中还有 22 个文件来自 LLaVA 的 Git 基线，以及 15 个额外 MemVR/MME 结果文件。上述五组来源覆盖了检查到的 2,494 个工作文件；三个 ZIP 自身另计。macOS 附带资源文件和字节码缓存按可丢弃附属文件处理。

### LLaVA 的 eval.zip

- [LLaVA 固定版本的评测说明](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md#scripts)
- [官方说明指向的 Google Drive 文件](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view)
- [直接下载入口](https://drive.google.com/uc?export=download&id=1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy)

官方说明指出，这个包自带评测标注、程序和 LLaVA 1.5 的预测。因此服务器里的许多旧答案、GPT 评分记录来自下载包。恢复原包即可获得它们；重新调用模型或外部评分服务，得到的是一次新的运行记录。

### MME 图片与标注

- [MME 官方项目及下载入口](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
- [官方项目链接的数据文件页面](https://huggingface.co/datasets/darkyarding/MME/blob/7056f44dac19de35e510b62d734bb2f7a6f64739/MME_Benchmark_release_version.zip)
- [固定 revision 下载地址](https://huggingface.co/datasets/darkyarding/MME/resolve/7056f44dac19de35e510b62d734bb2f7a6f64739/MME_Benchmark_release_version.zip)

当前公开文件为 199,848,448 字节，SHA-256 为 `1c4b79b494eae354df8fc0728c64fe4b89e46b4c609d36003be15dace6d8e3d7`。它与服务器旧 ZIP 的整体字节不同。归档保留了每个工作文件的内容指纹，恢复工具会检查下载包能否提供相同内容；打包结构变化可通过内容匹配处理，实际内容变化则会停止恢复并报告缺失文件。

本轮核验了公开文件元数据，完整下载及在线包内部内容比对留待实际恢复时完成。这个版本差异意味着：未来可重新获取公开 MME 数据，但原服务器旧版的逐文件重现仍以内容检查结果为准。

### MME 评分工具

- [官方工具文件页面](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/dd2950902889cd614d4edf606827240166d29381/tools/eval_tool.zip)
- [固定提交下载地址](https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/dd2950902889cd614d4edf606827240166d29381/tools/eval_tool.zip)

其公开 Git blob 为 `e02681d44e5a5e7f81023e50d3ce4cfce34c07d0`，与服务器旧工具 ZIP 的 Git blob 相同。

## 2. 旧 ZIP 校验值

这些是清空前旧文件的 SHA-256，用于识别当时版本。下载包可以因版本或打包方式不同而具有不同校验值；具体工作文件是否一致由恢复工具继续核验。

```text
800fe5af9213166aa31007ef74e3b99159fdc82afd26fa727ebe3314049abd00  eval.zip
36f837596c5b60255b23e420bed78935b59efb603eb8b63c1a1c078047e3a7de  MME_Benchmark_release_version.zip
b8125e2a7c3418e5761c12b3cfe4f1624b3c53ba44009e75b7d3f797d3d8acee  eval_tool.zip
```

逐文件指纹、旧包信息与内容来源见 [`evidence/data-recovery.json`](evidence/data-recovery.json)。

## 3. 在新机器上恢复

需要 Python 3.10 或更新版本、Git 和 curl。先克隆本仓库并进入其根目录；已有检出可直接使用。下面的命令只准备评测资料，模型权重和实验环境按主 README 另行准备。

在 Bash 中执行；所有下载、恢复和核验命令均保存日志：

```bash
set -euo pipefail
LLAVA_ROOT="$PWD"
ARCHIVE="$LLAVA_ROOT/experiments/2025-memvr-w2l-archive"
DOWNLOADS="$LLAVA_ROOT/playground/data/eval/recovery-downloads"
mkdir -p "$DOWNLOADS"

curl -fL --retry 3 \
  'https://drive.google.com/uc?export=download&id=1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy' \
  -o "$DOWNLOADS/eval.zip" 2>&1 | tee "$DOWNLOADS/download-eval.log"

curl -fL --retry 3 \
  'https://huggingface.co/datasets/darkyarding/MME/resolve/7056f44dac19de35e510b62d734bb2f7a6f64739/MME_Benchmark_release_version.zip' \
  -o "$DOWNLOADS/MME_Benchmark_release_version.zip" 2>&1 | tee "$DOWNLOADS/download-mme.log"

curl -fL --retry 3 \
  'https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/dd2950902889cd614d4edf606827240166d29381/tools/eval_tool.zip' \
  -o "$DOWNLOADS/eval_tool.zip" 2>&1 | tee "$DOWNLOADS/download-eval-tool.log"

python3 "$ARCHIVE/tools/restore_data.py" \
  --llava-root "$LLAVA_ROOT" --downloads "$DOWNLOADS" --check-inputs \
  2>&1 | tee "$DOWNLOADS/check-inputs.log"

python3 "$ARCHIVE/tools/restore_data.py" \
  --llava-root "$LLAVA_ROOT" --downloads "$DOWNLOADS" --restore \
  2>&1 | tee "$DOWNLOADS/restore.log"

python3 "$ARCHIVE/tools/restore_data.py" \
  --llava-root "$LLAVA_ROOT" --verify-existing \
  2>&1 | tee "$DOWNLOADS/verify.log"
```

如果 Google Drive 返回确认页面，使用上述官方文件页面手动下载到同一个 `DOWNLOADS` 目录。恢复工具会检查 ZIP 格式和文件内容；HTML 页面无法通过这一步。

### 工具如何恢复

1. 从三个 ZIP 建立文件内容索引，按指纹恢复到历史工作路径。因此 MME 包中额外的一层 `MME_Benchmark/` 目录能够正确展开到运行代码所需的位置。
2. 从 LLaVA `c121f043…` 的 Git 对象恢复 22 个原始提示词/示例文件。本仓库历史已包含该提交，使用完整克隆即可。
3. 从本归档的 `results/MME/` 恢复 15 个额外结果文件：2,374 条 JSONL 答案和 14 个分类文本。
4. 写入前核对所有输入；已有同名文件内容不同时停止，便于先保留另一份工作区。
5. 最后验证全部 2,494 个工作文件的 Git blob 内容指纹。

`--check-inputs` 和 `--verify-existing` 均为只读模式；`--restore` 才会写入 `playground/data/`。最后的 `--verify-existing` 输出 `verified_files: 2494`，表示历史工作文件的内容一致。三个 ZIP 保存在下载目录，macOS 附带资源和缓存按第 1 节的范围处理。

## 4. 恢复后的使用边界

当前保存的是当时的目录内容，并非每个基准的完整原始数据集。MME 有图片和标注；其他基准目录多为任务题目及参考预测，实际重跑还应按 [LLaVA 对应版本的评测说明](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/docs/Evaluation.md) 准备各自图片、标注或评测服务。

额外 MemVR/MME 结果已作为历史记录保存。恢复这些文件可以继续检查和评分；重新推理属于新的实验执行，命令、模型版本和输出应另行记录。
