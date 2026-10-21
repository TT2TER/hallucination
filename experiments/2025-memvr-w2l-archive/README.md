# MemVR 与 Where to Look 实验归档

归档日期：2026-10-21。实验代码的已知提交主要发生在 2025 年 8—9 月。

本目录保存服务器上的修改、现存实验输出，以及本仓库已有研究代码的固定版本和差异。归档依据是 Git 提交、文件内容比对、Notebook 和结果文件；每类材料都保留了来源和校验记录。

## 1. 实验基于什么

可以确认两组工作：

1. **服务器复现与调试**：在 LLaVA 中接入 MemVR，调整模型路径和接口；另行尝试运行 Where to Look 的 Notebook。
2. **本仓库的组合方法**：在 MemVR 中加入 Where to Look 的注意力计算和视觉 token 筛选，并尝试不同保留比例、熵差触发、反向筛选。

三份服务器项目的 Git 检出基线如下。它们也是本目录生成差异时使用的准确版本。

| 项目 | 上游仓库 | 服务器 HEAD |
|---|---|---|
| LLaVA | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) | [`c121f0432da27facab705978f83c4ada465e46fd`](https://github.com/haotian-liu/LLaVA/commit/c121f0432da27facab705978f83c4ada465e46fd) |
| MemVR | [1zhou-Wang/MemVR](https://github.com/1zhou-Wang/MemVR) | [`3cfbc6b5a97c926c287b7af5c8ef0d39a8a4f6c0`](https://github.com/1zhou-Wang/MemVR/commit/3cfbc6b5a97c926c287b7af5c8ef0d39a8a4f6c0) |
| Where to Look | [saccharomycetes/mllms_know](https://github.com/saccharomycetes/mllms_know) | [`a64c5d5a0ece793a2f7dc96e926d283a199033c1`](https://github.com/saccharomycetes/mllms_know/commit/a64c5d5a0ece793a2f7dc96e926d283a199033c1) |

本仓库代码以归档前的 `main` 提交 [`a7f1fd8f66fe1fef5184bdfdbfff8e8ea9dfa1ad`](https://github.com/TT2TER/hallucination/commit/a7f1fd8f66fe1fef5184bdfdbfff8e8ea9dfa1ad) 为固定参照。Git 历史确认其继承自上述 LLaVA 提交。MemVR、Where to Look 的版本是服务器当前可核对的基线；当年每次复制代码的确切时刻和检出版本，需要当时的操作记录才能进一步确定。

## 2. 已有 Git 历史记录的实验过程

时间使用各提交的作者时间，均为北京时间。提交证明代码曾被保存；实际运行情况需结合输出文件判断。

| 日期 | 提交/分支 | 源码所记录的工作 |
|---|---|---|
| 2025-08-08 | [`731fef86`](https://github.com/TT2TER/hallucination/commit/731fef86af4170c729f5431f95cb39f67040995b)，`twice+w2l` | 接入 MemVR 及 Where to Look 相关函数；新增 `our_inference_llava.py`、`llava_model_vqa_loader_our.py` 等组合方法入口 |
| 2025-08-25 | [`ae03b9a0`](https://github.com/TT2TER/hallucination/commit/ae03b9a08d253588a8da3361663097aa180fcc41) | 将视觉 token mask 从均值阈值改为 `look_rate` 百分位筛选；增加不同参数的 MME、POPE 等调用脚本 |
| 2025-08-25 | [`d2aec1e9`](https://github.com/TT2TER/hallucination/commit/d2aec1e938cf3d1acb3c6da52424da416f3a2681)，`entropy-diff-trigger` | 触发条件从当前层熵超过阈值，改为当前层与上一层的熵差超过阈值 |
| 2025-08-25 | [`a7f1fd8f`](https://github.com/TT2TER/hallucination/commit/a7f1fd8f66fe1fef5184bdfdbfff8e8ea9dfa1ad)，归档前 `main` | 通过一个 revert 提交恢复使用当前层熵的阈值判断；熵差版本保存在独立分支 |
| 2025-09-15 | [`41a12256`](https://github.com/TT2TER/hallucination/commit/41a1225666adbfcfca29d238f2b0a64888d7c861)，`test/reverse` | `utils.py` 将 mask 改为选取低于或等于百分位阈值的 token，保存反向筛选尝试 |
| 2026-10-21 | 本次归档 | 从服务器只读导出源码差异、Notebook 和既存结果，补齐基线、来源和恢复说明 |

### 组合方法实际改了哪些部分

- `llava_methods.py`、`run.py`：将 Where to Look 相关注意力计算接入本地 LLaVA 模型接口；`run.py` 中的 `vicrop_qa()` 生成视觉注意力与 mask。
- `utils.py`：`att_from_att_image_adaptive()` 根据 `look_rate` 生成视觉 token mask。
- `memvr.py`：新增启停和 mask 设置接口；在 MemVR 触发回溯时，用 mask 选择参与回溯的视觉 token。`github-main-vs-memvr.patch` 保存了对上游 MemVR 实现的完整文本差异。
- `our_inference_llava.py`、`llava/eval/llava_model_vqa_loader_our.py`：组合注意力计算、视觉 mask 和 MemVR 推理；后者在计算注意力时关闭 MemVR，在回答时重新开启并设置 mask。
- `eval_scripts/llava/mme_our.sh`、`pope_our.sh` 等：保存了参数组合及预期结果路径。脚本存在说明调用方案已被写下，运行完成情况需要相应结果和日志确认。

例如，主版本的 `look_rate=25` 使用第 75 百分位阈值，并保留大于等于阈值的元素。对注意力值 `[1, 2, 3, 4]`，阈值为 `3.25`，只保留 `4`。并列值会影响保留数量；`look_rate=0` 仍会保留最大值，不能解释为零个 token。

熵差分支则改变触发语义：假设当前层熵为 `0.8`、上一层为 `0.7`、阈值为 `0.75`，主版本满足 `0.8 > 0.75`，熵差版本判断的是 `0.1 > 0.75`。

## 3. 服务器上新增或修改的部分

### LLaVA 目录：45 个文件形成恢复覆盖层

相对服务器的 LLaVA 基线，归档保存以下四类文件。完整逐文件分类在 [`evidence/server-file-origins.tsv`](evidence/server-file-origins.tsv)。

| 类别 | 文件数 | 具体内容 |
|---|---:|---|
| 从 MemVR 原样复制 | 37 | 两处评测目录各 10 个文件、14 个评测 Shell 脚本、3 个推理脚本 |
| 从现有评测包原样复制 | 4 | MME 转换/评分、MM-Vet 转换、SEED 帧提取工具 |
| 在 MemVR 文件上的本地调整 | 3 | `memvr.py` 的注释；根目录 `inference_llava.py` 和 MME 脚本的模型路径 |
| LLaVA 已跟踪文件修改 | 1 | `llava_llama.py` 增加 `cache_position=None` 参数及说明 |

`server/LLaVA-overlay/` 按服务器项目内路径保存这 45 个文件，便于覆盖到准确的 LLaVA 基线中。

**改动性质：**服务器 `memvr.py` 相对所选 MemVR 基线只有四处注释差异，Python AST 相同；模型路径调整为 `/root/autodl-tmp/llava-v1.5-7b`。`cache_position` 只被加入函数签名，函数体没有继续使用或转发它，因此这里记录为接口兼容尝试，运行兼容性需另行验证。

这与 GitHub 上已有的组合方法是两份不同状态。特别是服务器和 GitHub 的 `llava_llama.py` 都包含这个兼容参数，其文本差异还包含空行。应通过基线和具体内容理解差异，文件哈希不同本身不能说明算法变了。

### MemVR 目录：工作区布局记录

该仓库相对自身 HEAD 记录了 21 个文件删除，当前导出的既有源码内容与基线一致。LLaVA 目录中存在相关代码的复制文件。`patches/server-memvr-layout.patch` 保存删除状态；该文件用于描述快照中的目录布局，代码来源由上游提交和 LLaVA 覆盖层共同保留。

### Where to Look Notebook：加载尝试与失败记录

`server/mllms_know/quick_start.ipynb` 保存服务器上的完整 Notebook，`patches/server-mllms-know.patch` 保存相对上游基线的原始差异。

- 单元数从 11 个变为 13 个。
- 修改模型加载单元：增加本地代理配置、`revision='a272c74'` 等加载选项和本地模型路径注释。
- 新增另一种 `LlavaProcessor` 加载尝试，以及 Transformers 版本检查单元。
- 两个加载单元均保存异常：`data did not match any variant of untagged enum ModelWrapper at line 277156 column 3`。
- 版本检查单元保存输出 `4.40.0`。Notebook 元数据的 kernel 显示名由 `vicrop` 改为 `memvr`，Python 版本记录由 `3.10.16` 改为 `3.10.0`。
- 后续四个有输出的示例单元与上游 Notebook 的完整单元对象相同。示例答案、图片和注意力图归为上游继承输出；这次本地加载尝试的状态由新保存的两个异常体现。

单元级证据见 [`evidence/notebook-cells.json`](evidence/notebook-cells.json)，单元编号从 0 开始。

## 4. 找到的实验输出与证据范围

`results/MME/` 保存服务器已有的 15 个额外文件，它们均不在服务器现有的原始 `eval.zip` 结果内容中：

- `answers/llava-v1.5-7b/memvr.jsonl`：2,374 条答案，`model_id` 均为 `llava-v1.5-7b`，`metadata` 均为空对象。
- `eval_tool/answers/llava-v1.5-7b/memvr/`：14 个类别的转换结果，合计 2,374 行。

JSONL 中有 1,187 个 `question_id`，每个对应两题；`(question_id, prompt)` 组合共 2,374 个。按随附转换器的文件名和提问规范化规则核对，14 类文本中的答案与 JSONL 一致。

服务器 MME 脚本声明：`retracing_ratio=0.12`、`entropy_threshold=0.75`、`starting_layer=5`、`ending_layer=16`、`temperature=0`、`max_new_tokens=1`。这组参数来自脚本快照，结果文件的元数据不足以证明实际使用了该启动命令。

现有材料足以保存预测和检查转换一致性。实际运行命令、权重校验值、耗时和对照结果仍需运行日志补充；本次归档的证据范围是代码、既存输出和文件一致性。组合方法、不同 `look_rate`、熵差及反向筛选的性能结论需要相应的对照输出支持。

## 5. 文件组织与恢复方式

```text
server/LLaVA-overlay/          服务器相对 LLaVA 基线的 45 个新增/修改文件
server/mllms_know/             修改后的完整 Notebook
github-main-overlay/          归档前 main 相对 LLaVA 基线的 38 个 Python/Shell 文件
patches/                      分别对照 LLaVA、MemVR、Where to Look 和各分支的差异
results/MME/                  15 个既存结果文件
evidence/                     来源、版本、单元比较、结果行数及环境包记录
licenses/                     上游许可文本及 Where to Look 的原始说明
CHECKSUMS.sha256               本目录文件的 SHA-256 校验表
```

恢复时，先选择需要恢复的状态，再检出对应的准确基线：

- **服务器 LLaVA 状态**：检出 LLaVA `c121f043…`，将 `server/LLaVA-overlay/` 的内容按原路径复制进去。
- **服务器 Notebook 状态**：检出 `mllms_know` 的 `a64c5d5a…`，使用归档的 `quick_start.ipynb`。
- **GitHub 组合方法**：直接检出本仓库 `a7f1fd8f…`；或检出 LLaVA 基线后使用 `github-main-vs-llava.patch`。`github-main-overlay/` 也保存了对应 Python/Shell 文件的固定副本。
- **历史变体**：基于 `a7f1fd8f…` 分别使用熵差或反向 mask 补丁，也可直接检出第 2 节链接的准确提交。

补丁采用不同对照基线，见 [`patches/README.md`](patches/README.md)。服务器的模型绝对路径、本地代理设置和数据路径需要按恢复机器调整。

## 6. 环境、归档校验与后续状态

检查时包元数据如下；完整包清单在 [`evidence/environment-packages.json`](evidence/environment-packages.json)。

| 环境 | Python | PyTorch | Transformers |
|---|---|---|---|
| base | 3.10.8 | 2.1.2+cu121 | 4.40.0 |
| memvr | 3.10.0 | 2.1.2 | 4.40.0 |
| mllms_know | 3.10.18 | 2.6.0 | 4.40.0 |

环境表来自安装元数据，Notebook 的 kernel 和版本输出属于其保存时的历史状态。归档过程只读取服务器文件，在本地检查传输哈希、上游基线、补丁恢复和结果结构；本轮使用原有实验代码，运行验证范围详见 [`evidence/verification.json`](evidence/verification.json)。

本目录保存代码差异、Notebook 和上述结果。模型权重、完整 Conda 目录、评测图片和其他大体积数据的保存范围仍应按清空前盘点清单另行核对。恢复实验时需要准备对应基础模型和原始评测数据。

## 7. 来源与署名

LLaVA 和 MemVR 的上游许可文本保存在 `licenses/`。Where to Look 的原始 README 声明 MIT 许可；本次服务器检出中的根目录未含其所指的 `LICENSE` 文件，因此一并保存原始 README 及仓库版本链接。逐文件复制来源和本地差异分开登记，便于后续追溯。
