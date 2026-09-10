# video_entropy_agent

**一句话**：视频问答里的"工具调用"模型（比如 DeepEyes）看起来会主动放大画面、回看片段，但它到底是真的在找证据，还是先想好答案再走个过场？这个仓库用一套可复现的测法回答这个问题，并给出在真实视频（VideoNet）上的结果。

不熟悉背景的人，先读 **[docs/00_大白话实验记录.md](docs/00_大白话实验记录.md)**——从头讲到尾，每个名词都解释。技术细节看 `docs/02_findings_technical.md` 和 `natural/README.md`。

## 结论（三句话）

1. **真实视频里确实存在"粗看答不了、定位到某 0.4 秒才能答"的题**，而且可以用机器认证出来（339 题里 22 条），不用人工标注。
2. **把这段证据喂给模型，它会用**（信念明显向正确答案移动；不需要证据的对照题上完全不动，说明测法本身没偏差）。**但让它自己去找，几乎找不到**（22 题只命中 1 题，时间工具几乎不用，每题机械地放大一次中间帧）。
3. **它的"犹豫"发生在调用工具之前**：六成的高不确定性 token 在工具调用之前，调用后文本几乎是确定性的"确认"；这个不确定性的形状与题目需不需要证据、答没答对都无关。也就是说，工具调用是仪式，不是取证。

![三层总览](docs/figs/fig3_three_layers.png)

## 目录

```
docs/
  00_大白话实验记录.md          ← 先看这个
  01_proposal.md                研究方案（三层框架、指标、数据设计）
  02_findings_technical.md      现象章草稿（技术版，含 22 条逐条表）
  pilot_report.pdf              合成数据 pilot 的完整报告
  figs/                         文中用到的图
common/video_imcot.py           视频会话 + zoom/seek 工具（DeepEyes 原生工具的多帧扩展）
pilot/                          合成孪生样本（5 对）：造样本、认证、notebook
natural/                        真实视频（VideoNet）流水线：筛选 → 定位证据窗 → 四条件审计 → 熵分析
natural/out/                    全部结果（jsonl / 表 / 图 / 精简轨迹）
```

## 复现

环境：一个能跑 Qwen2.5-VL / Qwen3.5 的 conda env（transformers ≥ 5.7、decord、flash-attn、PIL、pyarrow、matplotlib）。代码里的绝对路径（VideoNet 目录、模型目录、字体）在 `natural/items.py`、`natural/frames.py`、`natural/belief.py`、`pilot/make_samples.py` 顶部，按自己的机器改。

```bash
cd natural
python items.py --probe                                   # 选题 + 读视频属性
CUDA_VISIBLE_DEVICES=0 python run_screen.py --model qwen35    # 阶段一+二：筛选 + 证据窗搜索（认证者）
CUDA_VISIBLE_DEVICES=0 python run_audit.py --model deepeyes --certifier qwen35   # 阶段三：审计 + agentic + 熵
python analyze.py --model deepeyes@qwen35 --figs         # 表和图
```

所有阶段可续跑（中断后重跑同一命令即可）。VideoNet 视频和从它派生的合成样本视频**不在仓库里**（gated 数据集）；`pilot/make_samples.py` 可以从本地 VideoNet 重新生成。

## 数据与模型

- VideoNet（`raivn/VideoNet`，gated）：本地 339/5000 条视频，四选一动作识别。
- 被测：DeepEyes-7B。认证者：Qwen3.5-9B。
- 结果日期：2026-09-04 ~ 09-10。
