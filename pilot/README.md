# EvidenceGap-V pilot（3 条孪生样本）

对应方案 `实验记录/PROPOSAL_evidence_acquisition_260904.md` §3–§4 的"来源 B：受控编辑对"。目的不是当训练数据，而是先造出 3 条**在粗观测下客观不可辨、只有拿到 E* 才可判**的样本，用 DeepEyes-7B 走一遍 C1–C4 认证，把认证流程和四种证据条件的评测代码跑通。

## 文件

| 路径 | 内容 |
|---|---|
| `make_samples.py` | 生成器。全部可调参数在 `PARAMS`；每次迭代只改参数并升 `--version` |
| `samples.json` | 3 条样本的题面 / 选项 / 孪生答案 / E* 时间窗与空间框 / decoy 定义 / 粗采样帧索引 |
| `videos/<id>_{A,B}.mp4` | 孪生视频对（在 VideoNet 真实视频上做最小合成编辑） |
| `eval_deepeyes.py` | 认证评测（ms_swift 环境，`CUDA_VISIBLE_DEVICES=1 python eval_deepeyes.py --tag vN`） |
| `eval/results_vN.json`, `eval/log_vN.txt` | 每版结果；`results_final.json` = 各样本定版的合并 |
| `实验记录/REPORT_evidencegap_pilot_260904.pdf`（+ 同名 .html） | 从已执行的 notebook 抽图/表/轨迹排成的中文 PDF 报告（设计 → 迭代 → 认证 → 六条件信念 → 逐轮推理轨迹）。一次性生成脚本已删 |
| `evidencegap_pilot.ipynb` | **已执行、带输出的 notebook**：孪生对长什么样 → 六个证据条件的信念表/柱状图 → DeepEyes 逐轮工具调用轨迹 → 认证汇总（`build_notebook.py` 生成骨架，`pilot_viz.py` 是展示层） |

## 样本（5 条，全部通过 DeepEyes-7B 认证）

| id | 机制 | 基底视频 | 孪生差异 | E* | 定版 |
|---|---|---|---|---|---|
| `s1_occlusion_color` | 遮挡解除（时间为主） | f103252e 咖啡俯拍 18s | 卡片下圆片 红 / 蓝 | 帧 262–292（两个粗采样点之间）× 卡片区域 | v1 |
| `s2_tiny_tag` | 局部小字（空间为主） | 905ce3c1 调酒 4s | 标签数字 358 / 386，字高 9px | 全程 × 标签框 44×30 px | v4 |
| `s2b_tiny_tag_1080p` | 同上，1080p 底片 | f103252e | 字高 12px | 全程 × 56×38 px | v4 |
| `s3_roll_direction` | 短时窗动作 + 时序（时空） | 1c47ea6a 咖啡俯拍 9s | 小球 向右 / 向左 | 帧 160–187 × 桌面底部横条 | v2 |
| `s3b_flash_symbol` | 短时窗一闪而过的物体 | 1c47ea6a | 卡片符号 三角 / 圆 | 帧 160–187 × 卡片区域 | v3 |

粗观测 = 8 帧均匀采样、每帧 ≤256·28² 像素（与 07-23 探针和 VideoNet 评测协议一致）。S1/S3/S3b 的证据窗口严格落在两个粗采样点之间，所以 8 帧下**根本采不到**；S2/S2b 全程可见但标签在粗分辨率下只有 ~4 px 高。

## 最终认证结果（`eval/results_final.json`，检查图 `eval/check_final.jpg`）

| 样本 | C1 | C2 | C3 | C4 | decoy | none_k8 配对 p(y*) | true p(y*) A/B | twin 翻转 A/B | agentic A/B |
|---|---|---|---|---|---|---|---|---|---|
| s1_occlusion_color | ✓ | ✓ | ✓ | ✓ | ✓ | 0.35 | 1.00/0.99 | 1.00/1.00 | ✓/✗ |
| s2_tiny_tag | ✓ | ✓ | ✓ | ✓ | ✓ | 0.29 | 1.00/1.00 | 1.00/1.00 | ✗/✗ |
| s2b_tiny_tag_1080p | ✓ | ✓ | ✓ | ✓ | ✓ | 0.21 | 1.00/1.00 | 1.00/1.00 | ✓/✗ |
| s3_roll_direction | ✓ | ✓ | ✓ | ✓ | ✓ | 0.47 | 0.95/0.72 | 0.72/0.95 | ✓/✗ |
| s3b_flash_symbol | ✓ | ✓ | ✓ | ✓ | ✓ | 0.33 | 1.00/1.00 | 1.00/1.00 | ✗/✓ |

读法：`none_k8` 下 A/B 信念几乎相同（TV ≤ 0.04），配对平均 p(y*) 都在 chance 附近 → 粗观测客观分不开；`true` 下 p(y*) 全部 ≥ 0.72、`twin` 下全部翻到孪生答案 → 答案由证据**内容**决定；`decoy`/`irr` 不产生错误确信。`none_k16/32` 在 S1/S3/S3b 上能解（窗口被采到），在 S2/S2b 上仍不能（分辨率问题加帧没用）——这就是方案里说的 necessity 随观测预算变化的曲线。

**DeepEyes 自己用工具（agentic，zoom+seek）：10 次里对 4 次，每对孪生里最多只对一边** —— 不比纯先验猜好。看轨迹：S1-B 三次 zoom 全在卡片盖着的时刻，然后按先验答 "red"；S3-B seek 到了窗口但仍答 "right"（语言先验 0.79）；S2 两边各 zoom 一次都没框到标签。这就是方案 §1 的三层失败在 5 条样本上的具体样子：Acquisition 时间定位不到窗口、空间定位不到标签，Utilization 拿到了也被先验压住。

## 中文版（2026-09-04 下午加）

`samples.json` 每条样本带 `question_zh` / `options_zh`（视频画面里的数字、图形本身语言无关，不用重渲染）。`eval_deepeyes.py --lang zh` / `ev.set_lang('zh')` 把题面、粗采样开头语、工具返回说明、agentic 的 system / instruction / continue 提示词全部换成中文，并要求 DeepEyes **用中文思考**；工具名和 JSON 调用格式不变。中文下语言先验会变（"朝右侧/朝左侧"和 "toward the right/left side" 的先验不一定一样），所以认证在中文下重跑了一遍（`eval/results_notebook.json`，由 notebook 产生），notebook 默认就是中文版；英文已执行版留在 `evidencegap_pilot_en_executed.ipynb`。

| 样本（中文） | C1–C4 | none_k8 配对 p(y*) | true p(y*) A/B | twin 翻转 A/B | decoy 无错误确信 | 文本先验偏置 | agentic A/B |
|---|---|---|---|---|---|---|---|
| s1_occlusion_color | ✓ | 0.32 | 0.99/0.99 | 0.99/0.99 | ✗（红色 0.44→0.59） | 0.06 | ✓/✗ |
| s2_tiny_tag | ✓ | 0.25 | 1.00/1.00 | 1.00/1.00 | ✓ | 0.12 | ✗/✗ |
| s2b_tiny_tag_1080p | ✓ | 0.24 | 1.00/1.00 | 1.00/1.00 | ✓ | 0.13 | ✗/✗ |
| s3b_flash_symbol | ✓ | 0.24 | 1.00/0.99 | 0.99/1.00 | ✓ | 0.09 | ✗/✓ |
| s3_roll_direction | ✓ | 0.47 | 0.83/0.65 | 0.64/0.84 | ✓ | 0.67（"朝右侧"） | ✓/✗ |

中文下 agentic 3/10（英文 4/10），`</tool_call>` 截停后没有再出现退化输出；"朝右侧"的语言先验（0.80）和英文 "toward the right side"（0.79）一样重。

## 迭代记录

| 版本 | 改动 | 结果 |
|---|---|---|
| v1 | 初版三条 | S1 ✓；S2 ✗C1（24px 字在粗采样下 ~11px，p=0.99 直接读出）；S3 全挂（"from left to right" 文本先验 0.82，真证据也推不出方向） |
| v2 | S2 字 15px；S3 球 r=30→48、证据框加高、选项改 "toward the right/left side"；新增 S3b 符号卡 | S2 仍 ✗C1（~7px 仍可读）；S3 C2/C4 过了但 C1/C3 因先验单边 >0.5 挂；S3b 只挂 C3（"cross" 先验 0.52） |
| v3 | 认证判据改为孪生对平均（见 `certify()` docstring）；S2 字 12px；新增 S2b（1080p 底片，15px）；S3b 换 circle | S3 ✓、S3b ✓；S2 ✗（~5.5px 仍 0.99）；S2b ✗（~4.7px 半可读：A 0.87，B 误读） |
| v4 | S2 字 9px（~4.1px）；S2b 字 12px（~3.7px） | 两条都 ✓ |

经验：DeepEyes-7B 在 588×336 的粗采样帧上能可靠读出 ~5.5px 高的粗体数字，要到 ~4px 才真正读不出；而原生 9–12px 的裁剪它读得 100%。这个 4px↔9px 的窗口就是"空间 necessity"能落脚的地方，比预想的窄。

## 四种证据条件（同一粗观测前缀，只换追加的"工具返回帧"）

- `true`：本视频 E* 窗口内 4 帧，裁到 E* 框，高分辨率
- `twin`：孪生视频同窗口同框的 4 帧 —— 忠实用证据的模型应翻到孪生答案
- `decoy`：同框同尺寸但不含判别信息（S1/S3 取窗口外的帧，S2 取同高度另一块台面）—— 不应收敛
- `irr`：同尺寸灰图 —— 不应变化
- 另有 `text`（无帧）、`none_k8/16/32`（不同预算的均匀采样）、`agentic`（DeepEyes 自己用 zoom+seek）

信念 $b$ = 四个选项字母首 token 的 softmax，对选项顺序做 4 个循环置换取平均（去字母偏置），judge-free。

## 认证标准（`certify()`）

| 条件 | 判据 |
|---|---|
| C1 粗观测不可辨 | A/B 在 `none_k8` 下信念的 TV 距离 < 0.15，且两者 $p(y^*)$ < 0.5 |
| C2 证据充分 | `true` 下 A/B 都答对且 $p(y^*)\ge 0.6$ |
| C3 无文本捷径 | `text` 下 $p(y^*)$ < 0.5 |
| C4 证据可交换 | `twin` 下 A/B 都翻到孪生答案 |
| 附加（对模型的测量，不进 PASS） | `decoy` 不产生错误确信（置信度不升 >0.1 且指向错误答案）；`irr` 相对 `none_k8` 的 TV < 0.25 |

PASS 只看 C1–C4，即样本本身是否"粗观测不可辨、证据充分、无捷径、证据可交换"。模型在 decoy 下变得更自信、在 irr 下乱动，是 False Resolution / 不稳定这类**要测量的模型行为**，不是样本缺陷，所以单独报。

## 迭代记录

见文末（每版一行：改了什么 → 哪条过/没过 → 下一步）。
