# evidencegap_v_natural — 自然视频上的"证据必要性 → 获取 → 使用"流水线

## 在验证什么

论文主张（`实验记录/PROPOSAL_evidence_acquisition_260904.md`）：现有视频工具模型的"调用率高、acc 高"证明不了工具真的参与了推理。要把问题拆成三层，每层单独用反事实测：

| 层 | 问题 | 怎么测 |
|---|---|---|
| Necessity（样本性质） | 这道题在 8 帧粗观测下**客观上**答不了、给了某段证据就能答？ | 信念随观测预算的曲线 + 滑窗搜索 E* |
| Acquisition（模型行为） | 模型自己调工具时，找没找到 E*？ | agentic 轨迹里 seek/zoom 的时间窗与 E* 求交 |
| Utilization（模型行为） | 把 E* 喂给它，它用没用？ | 同一前缀换四种"工具返回"：true / decoy / irr / 等预算密采 → EEG |

合成 pilot（`../evidencegap_v_pilot/`）已经证明这套测法在受控样本上能拉开差距。这里是把它搬到 VideoNet 真实视频上，回答：自然视频里有多少题是 evidence-required？DeepEyes 在这些题上三层各断在哪？

## 流水线（每步对应的代码 / 输入 / 输出）

```
items.py ──► frames.py ──► belief.py ──► pipeline.py ──► run_screen.py ──► run_audit.py ──► analyze.py / traj_viz.py
 选题+元数据   帧缓存(JPEG)   信念读出        三阶段逻辑      阶段一+二        阶段三+agentic     表/图/精简JSON
```

### 0. 数据与基础设施

| 步骤 | 代码 | 做什么 | 产物 |
|---|---|---|---|
| 选题 | `items.py` | 从 `VideoNet/benchmarks/mcq_{test,val}.jsonl` 选出本地有视频的 339 条，join `benchmark_video_metadata.parquet` 拿 domain/name/definition；`sibling_map()` 找同域兄弟视频（twin 条件用，222/339 有） | `cache/video_probe.json`（fps/帧数/分辨率） |
| 帧缓存 | `frames.py` | 每条视频需要的帧解码一次存原分辨率 JPEG；所有条件（不同 k、分辨率、裁剪）都从缓存读。解耦 decord 与推理环境 | `frames_cache/<uuid>/<idx>.jpg`，29,860 帧 2.9 GB |
| 信念读出 | `belief.py` | `Backend(name)` 加载 deepeyes / qwen35 / qwen38；`belief()` = 四个选项字母首 token 的 softmax，对选项顺序做 4 个循环置换取平均。judge-free。Qwen3.5 系模板预开 `<think>`，读信念时 `enable_thinking=False` | — |

### 1. 阶段一：必要性初筛（`run_screen.py --stage screen`）

对每条题算 7 个条件的信念：k ∈ {2,4,8,16,32} 帧均匀采样 @256·28²、k=8 @768·28²（高分辨率）、纯文本。`pipeline.screen_item` → `pipeline.classify`：

- `text_shortcut`：不看视频 p(正确) ≥ 0.7
- `already_solvable`：8 帧 p ≥ 0.7 且答对 → N=0 对照
- `candidate`：其余 → 进阶段二

> 为什么不在这里判 unsolvable：Cricket 那条均匀 32 帧只有 0.07，但 [1.8,3.6]s 密采到 0.70。"均匀加帧解不了"≠"信息不在视频里"。

### 2. 阶段二：E* 窗搜索 + 认证（`run_screen.py --stage localize`）

对每条 candidate，`pipeline.localize_item`：coarse-8 前缀 + 窗内密采 6 帧@768·28²，窗宽 10/20/30% × 步长 10%，共 24 个窗，取 p(正确) 最高的最小窗为 E*；同宽度最差窗为 decoy。`pipeline.certify`：

- `evidence_required`：E* 答对 且 p ≥ 0.60 且 相对 8 帧提升 ≥ 0.25
- `borderline`：提升不足（粗观测本来就不差）
- `unsolvable`：最好的窗也 < 0.60（给了证据也解不了）

产物：`out/screen.jsonl`（每行一条 (uuid, model, stage)，可续跑），日志 `out/log_screen_<model>.txt`。

### 3. 阶段三：四条件审计 + agentic（`run_audit.py --model X --certifier Y`）

在 Y 认证的 evidence_required 题 + 40 条 already_solvable 对照上，对模型 X 跑 `pipeline.audit_item`：同一 coarse-8 前缀，追加 6 种"工具返回"：

| 条件 | 内容 | 期望 |
|---|---|---|
| none | 只有 8 帧 | 基线 |
| true | E* 窗密采 6 帧 | 涨 |
| decoy | 同宽度最差窗，token 成本相同 | 不涨 |
| irr | 同尺寸灰图 | 不变 |
| twin | 同域兄弟视频的中段帧 | 应翻向兄弟类（忠实用证据） |
| dense_matched | 把 true 的 token 预算换成 26 帧均匀采样 | 若≈true，主动定位没必要 |

`pipeline.metrics` 算 EEG = log p(true) − log p(decoy)、ERG、gain_over_dense、False Resolution、twin 翻转。

同时 `pipeline.agentic_item`：给 X 两个工具（zoom / seek，中文提示词 `prompts_zh.py`），让它自己决定看哪，逐轮记录 think / tool_call / 返回帧；`temporal_hit` = 工具时间窗是否落进 E*；`entropy_profile` = 逐 token 熵 + PEA 式 top-20% 高熵位置 10-bin 直方图 + `<tool_call>`/`<answer>` 的位置百分比。

产物：`out/audit.jsonl`；跑完自动 `traj_viz.export_all` → `out/trajectories_<tag>.json`（精简、按句拆行）、`out/figs/entropy_<tag>/<uuid>.png`（逐条：熵 vs 输出位置%，标工具调用起点）、`out/figs/entropy_summary_<tag>.png`。日志 `out/log_audit_<tag>.txt`。

### 4. 汇总（`analyze.py [--figs]`）

归类分布、信念-预算曲线表、审计表、跨模型交叉表（两个认证者的标签对照 + E* 窗 IoU）。图：`out/figs/fig1_budget_curve.png`、`fig2_audit.png`；逐条 TSV `out/screen_table.tsv`。

## 已跑完的（认证者 = DeepEyes-7B，2026-09-04）

| 步骤 | 命令 | 结果 |
|---|---|---|
| 冒烟 3 条 | `smoke3.py --model deepeyes` | `out/smoke3_deepeyes.json`；发现 Cricket 现象，改了 classify/certify |
| 阶段一+二 | `run_screen.py --model deepeyes` | 339 筛 / 280 窗搜索：**8 evidence_required** / 55 solvable / 36 borderline / **236 unsolvable** / 4 text。236 条里 8 帧 p=0.235 ≈ 随机 → 是 DeepEyes 不认识这些动作，不是信息不在 |
| 阶段三 | `run_audit.py --model deepeyes` | required n=8：none 0.39 → true 0.71，decoy 0.34，等预算密采 0.52，**EEG +0.80**；对照 n=40：**EEG +0.01**（尺子自洽）。agentic：时间命中 25%，seek 使用 0%，每题 zoom 1 次 |

## 正在跑 / 接下来

1. **正在跑**：`run_screen.py --model qwen35`（GPU1，日志 `out/log_screen_qwen35.txt`）——换 Qwen3.5-9B 当认证者。3 条冒烟里它读题明显更准（Ice Hockey 0.98 vs 0.70），且把 Cricket 的 E* 定位到与 DeepEyes 相同的窗。
2. `run_audit.py --model deepeyes --certifier qwen35`：**强模型认证 → 弱模型被测**，这才是正式实验设计。含 agentic + 熵图。
3. `run_audit.py --model qwen35 --no-agentic`：强模型自己的 Utilization（它零样本不调工具）。
4. `analyze.py --figs`：跨模型交叉表——两个模型的 E* 窗一致，说明 E* 是视频的性质不是模型的幻觉。

## 环境 / 资源

- 帧缓存、DeepEyes、Qwen3.5-9B 推理：`ms_swift`（GPU1，与用户的 Jupyter kernel 共存）
- Qwen3.8-27B（54 GB）：登录节点放不下；备用 env `verl_qwen35`（有 causal_conv1d，无 decord，靠帧缓存可跑）
- 计划文件：`/root/.claude/plans/curried-stargazing-bunny.md`
