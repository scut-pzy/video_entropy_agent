# 从"调不调工具"到"能不能补齐证据"：部分观测下的视频主动证据获取与 belief revision

研究方案 v1（2026-09-04）。承接 `PROPOSAL_evidence_gain_260828.md`（"调用≠使用"的证据链），这一版把问题往前推一层：不再问"工具有没有被用"，而是问"样本客观缺什么证据、模型能不能找到、找到后信不信"。PEA 在这里的角色是诊断工具，不是开关，也不是奖励。

---

## 0. 一句话主线

> 在粗粒度视频观测下，有些问题客观上缺少完成判断所需的判别性证据。我研究模型能否**识别这个证据缺口**（Necessity）、**主动定位并取回缺失的时空证据**（Acquisition）、并**真正用证据修正已有判断**（Utilization）。我用反事实证据替换来度量这三层，用 PEA 的 entropy signature 刻画这一过程中的不确定性结构，最后用 Evidence-aware RL 让模型学会这套行为。

三个我明确**不写**的故事：
- 不写"模型看不清所以让它 zoom"——我 07-30 的假证据替换实验已经证明 DeepEyes 的裁剪图基本没进决策（80 题只差 1 题），"看清"不是瓶颈。
- 不写"模型 entropy 高所以让它 zoom"——模型对错误答案也可以非常确信（VideoNet 探针 14/20 先下结论再调工具），主观 entropy 不能定义是否需要工具。
- 不写"两阶段训练"——VideoSearcher 的 BiSPO 已经在 loss 层做了解耦，且它自己的 zoom_in 在 VSQA 上 RL 后掉到 0.2×，说明分阶段本身不解决"调了不用"。

---

## 1. 问题定义：Evidence Necessity → Acquisition → Utilization

记视频 $V$、问题 $q$、候选答案集 $\mathcal{Y}$。模型初始只看到**粗观测** $o_0$（均匀 $K$ 帧、低分辨率）。工具调用 $a$ 返回一段时空证据 $e(a)$。

| 层级 | 定义 | 是谁的属性 | 怎么测 |
|---|---|---|---|
| **Evidence Necessity** $N(x)$ | 在 $o_0$ 下答案不可辨识：存在另一个视频 $V'$，其粗观测与 $V$ 不可区分但正确答案不同 | **样本**的属性，与模型无关 | 成对构造 + 认证协议（§4.3） |
| **Evidence Acquisition** | 调用 $a$ 返回的 $e(a)$ 是否覆盖判别性证据区域 $E^*$（时间窗 × 空间框） | 策略的属性 | 时空命中率 / IoU |
| **Evidence Utilization** | 给定 $e$，模型的答案信念 $b_1=p(y\mid o_0,e)$ 相对 $b_0=p(y\mid o_0)$ 是否发生**与证据内容一致**的改变 | 策略的属性 | 反事实替换：同一前缀换 $e$ 看 $b_1$ 变不变 |

三层是**串联**的：Necessity 决定该不该调，Acquisition 决定调得对不对，Utilization 决定调了有没有用。现有指标（tool call rate、acc、IoU、注意力）只能各碰到其中一层的表面，没有一个能证明三层全部成立。我 08-28 文档的全部证据（DeepEyes 假证据替换 Δ≈0、CoT 塌缩到 0、VideoSearcher zoom_in 0.2×）都落在 Utilization 层的失败。

**什么是有效调用**：$N(x)=1$，$e(a)\cap E^*\neq\emptyset$，且 $b_1$ 向 $y^*$ 收敛而把 $e$ 换成对照证据时 $b_1$ 不收敛。四个条件缺一不可。

**什么是假调用**（taxonomy，§6 会用 PEA 和信念指标区分）：

| 类型 | 定义 | 主要落在哪一层 |
|---|---|---|
| Redundant Call | $N(x)=0$ 或 $b_0$ 已高置信且正确，工具结果几乎不改变判断 | Necessity |
| Overconfident Skip | $N(x)=1$ 但模型低 entropy 直接作答 | Necessity |
| Mislocalized Call | 知道要证据，但时间窗 / 空间框没覆盖 $E^*$ | Acquisition |
| Irrelevant-Evidence Call | 取回的内容与问题无关 | Acquisition |
| Unused Evidence | 取回了 $E^*$，$b_1\approx b_0$ | Utilization |
| False Resolution | 取回非判别性 / 误导性证据后 $H(b_1)<H(b_0)$ 且 $\arg\max b_1\neq y^*$ | Utilization |

---

## 2. 与现有工作的差异

| 工作 | 他们证明了什么 | 用什么证明 | 缺什么 |
|---|---|---|---|
| DeepEyes (2505.14362) | RL 能让模型调 zoom 并涨 acc | 调用率、acc、IoU | 无反事实；我复现：假证据 Δ=1/80，先答后看 84–96% |
| VideoSearcher (2607.02927) | BiSPO 把 tool 与 acc 的优势分开归一化，工具调用更"purposeful" | Fig.7 调用次数分布、消融 acc | 无反事实、无 necessity 概念；zoom_in 在 VSQA 上反而降到 0.2× |
| FaithEyes (2607.28225) | 用 judge 评工具结果有没有被引用 | LLM judge | 无因果干预，judge 可被"提一句"骗过 |
| CoT faithfulness（Turpin 2305.04388, Lanham 2307.13702） | 扰动中间步骤看结论变不变 | 反事实 | 对象是文本 CoT，不是工具返回的视觉证据 |
| PEA (AAAI-26) | 高熵 token 的位置分布是推理结构的描述子，可做过程奖励 | entropy signature + prototype | 对象是纯文本推理，没有外部证据注入点 |
| **本文** | 三层框架 + 反事实证据 benchmark + 不确定性结构诊断 + Evidence-aware RL | 成对样本反事实替换、答案信念变化、PEA signature | — |

一句话：VideoSearcher 把"evidence acquisition"写进了目标却用调用次数来证明；我把它拆成可分别证伪的三层，每层配一个反事实指标。

---

## 3. Evidence Necessity：怎么定义"客观需要额外证据"

这是全文最容易被攻击的地方，必须先于一切定死。

**定义**（可辨识性，不是难度）：$N(x)=1$ 当且仅当存在 $(V', y')$，$y'\neq y^*$，且 $o_0(V')\approx o_0(V)$（在评测分辨率与帧数下不可区分）。也就是说，仅凭 $o_0$，贝叶斯最优答题者也只能在 $\{y^*, y'\}$ 之间猜。

**操作化 = 成对构造**：每个 $N=1$ 样本天然带一个孪生样本 $(V, V')$，粗观测几乎相同，答案不同，差异只在 $E^*$。这比"找难题"强得多——难题可能只是模型不会，孪生对是**任何观察者**在 $o_0$ 下都分不开。

**认证协议**（每对样本都要过，报告通过率）：

| 检验 | 做法 | 通过标准 |
|---|---|---|
| C1 粗观测不可辨 | ≥3 个强 VLM（Gemini / GPT / Qwen3-VL-235B）+ 2 名人工，只看 $o_0$，做对内二选一 | 平均 acc 在 50% ± 10pp |
| C2 证据充分 | 同一批评审，$o_0$ + 标注的 $E^*$ | acc ≥ 90% |
| C3 无捷径 | 只给文字（问题+选项）；只给单帧；只给音频/字幕（若有） | 均 ≈ chance |
| C4 证据可交换 | 把 $E^*$ 与孪生的 $E'^*$ 互换喂给评审 | 评审答案随之翻转 ≥ 80% |

四条全过才算 $N=1$。C4 是关键：它证明答案是由 $E^*$ 的**内容**决定的，而不是由"某处被放大了"这个动作决定的。

**$N=0$ 对照样本**同样要认证：$o_0$ 下评审 acc ≥ 90%，工具无增量。训练和评测都必须混入 $N=0$，否则"永远调用"是最优策略，Necessity 层无法评。

---

## 4. 数据集：Evidence-Required Video Benchmark（暂名 EvidenceGap-V）

### 4.1 两个来源，互为补强

**来源 A：自然孪生对（主体）**——从细粒度动作数据挖。VideoNet 元数据每个动作类带 `start/end` 段和 `definition` 线索文本（例：Triple Salchow — "Look for the back inside edge takeoff without toe pick assistance"），同域内的易混类（Salchow vs Loop vs Toe Loop）就是天然孪生：粗观测都是"一个人跳起来转三圈"，判别证据在起跳瞬间的冰刀（时间窗 ≈ 0.3s，空间 ≈ 脚部）。`definition` 直接告诉我 $E^*$ 该往哪标。候选源：VideoNet（339 视频已本地，4000 MCQ），FineGym / Diving48 这类子动作数据集可作补充。

**来源 B：受控编辑对（对照）**——在真实视频上做最小编辑造孪生：遮挡物在最后 20% 帧才移开露出真实颜色；局部文字只在某几帧可读；同一段动作剪接不同结尾。编辑对的好处是 $E^*$ 完全已知、混淆变量可控；坏处是人工感，所以只做对照，主结论必须在来源 A 上成立。

场景覆盖（两个来源都要有）：遮挡解除、短时窗关键动作、局部文字/标识、物体身份只在局部暴露、前期一致后期分岔、局部结构决定类别、视角切换后才可见。

### 4.2 每个样本的四种证据条件

这里我把用户方案里的"False Evidence"拆成两种，因为它们检验的东西相反：

| 条件 | 内容 | 忠实使用证据的模型**应该**怎么反应 | 检验什么 |
|---|---|---|---|
| $e_{\text{true}}$ | 本视频的 $E^*$ | 收敛到 $y^*$ | Utilization（正向） |
| $e_{\text{twin}}$（反向证据） | 孪生视频的 $E'^*$，同位置同尺寸同格式 | **收敛到 $y'$**——证据是真的，只是指向另一个答案 | Utilization（敏感度）：答案是否跟着证据走 |
| $e_{\text{decoy}}$（诱饵证据） | 形式合理但不含判别信息：本视频非判别区域的高清裁剪，或第三类视频的相似区域 | 不收敛、保持不确定或再次调用 | 抗 False Resolution |
| $e_{\text{irr}}$ / $\emptyset$ | 灰图 / 不给工具结果 | $b_1\approx b_0$ | 基线与 token 成本对照 |

**为什么必须拆**：在 $e_{\text{twin}}$ 下"entropy 下降且答错"对忠实模型是**正确行为**（它看到的证据确实支持 $y'$）；在 $e_{\text{decoy}}$ 下同样的现象才是 False Resolution。不拆开，"对假证据鲁棒"和"对真证据敏感"就成了一对矛盾要求，模型只要忽略工具就能同时"满足"前者。拆开后两个条件各自可证伪，且合起来把"忽略证据"和"盲从证据"两个退化解都排除了。

### 4.3 规模与标注

目标 ≥ 1,000 对认证通过的自然孪生对 + 300 对受控编辑对；每对标 $E^*$ 的时间窗 $[t_s,t_e]$ 与空间框。标注流程：`definition` → 强 VLM 提议时空框 → 人工校验 → C1–C4 认证。功效：检出 5pp 的 ERG 在 80% power、约 15% 不一致率下需要 ~500 对，1,000 对给分层分析留余量。

---

## 5. 时空证据工具：把三种工具收成一个查询

VideoSearcher 用 `choose_frames / find_frame / zoom_in` 三个工具，我合成一个：

```
inspect(t_start, t_end, n_frames, bbox=None) → 在 [t_start, t_end] 内取 n_frames 帧，按 bbox 裁剪后以高分辨率返回
```

这个抽象让四个学习目标各有可测的参数：**When**（时间窗）、**Which**（帧数/密度）、**Where**（bbox）、**What**（隐式，落在调用前的推理文本里，用"调用前是否已写定答案"来诊断）。

两条约束让"无效调用有成本"：总视觉 token 预算 $B$（粗观测 + 全部工具返回），超预算截断；每次调用固定小成本。**匹配预算的无工具基线**（把同样的 token 预算花在更密的均匀采帧上）是必须有的对照——如果密采均匀帧就能达到同样 acc，"主动获取"就没有存在的必要。

---

## 6. PEA 诊断：不确定性结构，而不是 entropy 标量

### 6.1 我从 PEA 原文里拿什么

PEA 的对象是 token 级 entropy $H_t=-\sum_v p_t(v)\log p_t(v)$；**entropy signature** = 取整条推理轨迹 top-20% 高熵 token 的相对位置，分 $B=10$ 个等宽 bin 做归一化直方图 $h\in\mathbb{R}^{10}$；论文的前提是"强推理在关键决策点有局部熵峰，而不是均匀高熵"，signature 刻画的是"在哪里不确定"这个**结构**，与具体内容解耦。prototype = 对专家轨迹 signature 做层次聚类的质心 + 质量分；奖励 = 与 prototype 的对齐（模仿 / 发现+动态更新 / 偏离惩罚）。PEA 自己的结论：静态 prototype 会被 hack（reward 涨 acc 塌），动态多 prototype 才稳。

### 6.2 迁移到工具轨迹：两个层面的 entropy，分工不同

我把 entropy 分成两层，**任何一层都不进 reward**：

| 层面 | 对象 | 用途 |
|---|---|---|
| **答案信念** $b=p(y\mid\cdot)$ | MCQ 选项 token 的分布（强制作答位置的 logits，judge-free） | 定义 Utilization 指标与四种 resolution 模式 |
| **PEA entropy signature** $h$ | 整条轨迹 token 级高熵位置直方图 | 刻画"模型在哪一段深思"，与失败模式对齐 |

工具轨迹天然有地标：`调用前推理 | 工具参数 | 工具返回后推理 | 作答`。我保留 PEA 的原始 signature 定义（相对位置 10 bin），只在图上叠加工具返回点的相对位置；另外报一个**地标对齐的变体**（把三段各自归一化再拼接），明确标注为本文的适配而非 PEA 原定义。

### 6.3 要验证的假设（都是可证伪的）

- **H1 有效轨迹有特定 signature**：在认证 $N=1$ 样本上，Correct Resolution 的轨迹熵峰集中在调用前（推理缺什么）和工具参数（在哪/何时），返回 $e_{\text{true}}$ 后熵快速落下；Redundant / Unused 类轨迹调用前几乎无熵峰。
- **H2 CoT 塌缩 = signature 塌缩**：我 08-28 观察到的思维链长度塌到 0，会表现为 signature 质量全部移到末端 bin。这是判据，不需要 judge。
- **H3 prototype 与 taxonomy 对齐**：对四种证据条件下的轨迹做 PEA 式聚类，簇成员与 §1 的假调用 taxonomy 显著相关（用 ARI / 互信息报）。若不相关，PEA 在这里就只是可视化，我会如实写。

### 6.4 四种 resolution 模式（用答案信念定义，不用 token 熵）

给定条件 $c$ 下的 $b_1^c$，$\Delta H = H(b_1)-H(b_0)$，$\epsilon$ 为噪声阈值：

| 模式 | 条件 | 在 $e_{\text{true}}$ 下 | 在 $e_{\text{decoy}}$ 下 |
|---|---|---|---|
| Correct Resolution | $\Delta H<-\epsilon$ 且 $\arg\max b_1=y^*$ | 期望行为 | — |
| No Resolution | $|\Delta H|\le\epsilon$ 且 $\mathrm{TV}(b_1,b_0)\le\epsilon'$ | Unused Evidence | 期望行为 |
| Conflict Amplification | $\Delta H>+\epsilon$ | 失败（真证据反而更乱） | 可接受 |
| False Resolution | $\Delta H<-\epsilon$ 且 $\arg\max b_1\neq y^*$ | 严重失败 | 核心失败模式 |

同一条轨迹要在多个条件下看才能归类：只看 $e_{\text{true}}$ 下的 Correct Resolution 不够，还要在 $e_{\text{twin}}$ 下翻到 $y'$、在 $e_{\text{decoy}}$ 下不动，才算"证据真的进了决策"。

---

## 7. 评价指标

### 7.1 核心量：有效证据增益 EEG（Effective Evidence Gain）

回答"调用工具后到底获得并利用了多少有效信息"：

$$\mathrm{EEG}(e)=\mathbb{E}\big[\log p(y^*\mid o_0,e)-\log p(y^*\mid o_0,e_{\text{ctrl}})\big]$$

$e_{\text{ctrl}}$ 是与 $e$ 同尺寸同格式的对照证据（$e_{\text{irr}}$ 或错位裁剪）。两个版本：

- **EEG$_{\text{oracle}}$**：$e=E^*$（人工标注），只测 Utilization；
- **EEG$_{\text{policy}}$**：$e=e(a)$，模型自己调的，测 Acquisition × Utilization。

$\mathrm{EEG}_{\text{policy}}/\mathrm{EEG}_{\text{oracle}}$ = 获取效率。08-28 的 ERG = acc(真)−acc(假) 是它的 0/1 版本，保留作可读的伴随指标。EEG 是 log-prob 空间的，对 32 条这种小 eval 也有分辨率，且不依赖 judge。

### 7.2 分层指标

| 层 | 指标 | 定义 |
|---|---|---|
| Necessity | Necessity Alignment | $P(\text{call}\mid N{=}1)$ 与 $P(\text{call}\mid N{=}0)$ 分开报 |
| | Redundant Call Rate | $N=0$ 且调用 且 $\mathrm{TV}(b_1,b_0)\le\epsilon'$ |
| | Overconfident Skip Rate | $N=1$ 且不调用 且 $\max b_0\ge\tau$ |
| Acquisition | Temporal Hit / Spatial IoU | 调用窗与 $E^*$ 时间窗交并；命中帧上的 bbox IoU |
| | Mislocalized Call Rate | 调用了但时空命中为 0 |
| Utilization | EEG$_{\text{oracle}}$ / EEG$_{\text{policy}}$ | 同上 |
| | Evidence Utilization Rate | $P(\text{翻对}\mid e_{\text{true}},\ \emptyset\text{ 下答错})$ |
| | Twin Sensitivity | $P(\arg\max b_1=y'\mid e_{\text{twin}})$——应高 |
| | False Resolution Rate | $P(\Delta H<-\epsilon\wedge\text{答错}\mid e_{\text{decoy}})$——应低 |
| | Belief Revision Magnitude | $\mathrm{KL}(b_1\|b_0)$ 在 $e_{\text{true}}$ 与 $e_{\text{irr}}$ 下之差 |
| 综合 | Tool Necessity Gain | 仅在 $N=1$ 上：acc(带工具) − acc(匹配预算无工具) |
| 过程 | 调用前熵峰质量、signature-taxonomy ARI | §6 |

**Twin Sensitivity 高 + False Resolution 低**这一对是"忠实使用证据"的操作性定义；单独任何一个都能被退化策略满足。

---

## 8. 第一层实验：现象发现

对象：DeepEyes-7B（frame_index 扩展）、VideoSearcher-8B（若权重可得，否则按其报告复现 BiSPO 奖励）、Qwen3-VL-8B / Qwen3.5-9B 零样本 agentic、Video-R1 / VideoChat-R1.5 作无工具参照、我自己的三个 RL 检查点（阶段一 860、阶段二 960、zoom_step1）。

在 EvidenceGap-V pilot（先 200 对）上逐条回答：

| 问题 | 已有的 pilot 证据 | 要补的 |
|---|---|---|
| $N=1$ 时调不调、$N=0$ 时调不调 | 我的阶段二 tool_rate 恒 1.0（不看样本） | 分 $N$ 报 |
| 调的位置对不对 | 07-23：frame_index 22 次 20 次塌到中间帧 | 时空命中率 |
| 屏蔽 / 替换工具结果 acc 变不变 | 07-30：DeepEyes 1/80；oracle +0.0pp | 四条件全做 |
| 低 entropy 下的冗余调用 | 07-23：14/20 先下结论再调 | 用 $b_0$ 量化 |
| 工具结果有没有进后续推理 | 08-28：阶段二 31/32 返回后零推理直接作答 | signature 塌缩曲线 |
| VideoSearcher 是否例外 | Fig.7 zoom_in 0.2× | 它的 EEG |

预期产出：一张表证明"所有现有模型 EEG$_{\text{policy}}\approx 0$，且主要失败模式是 Redundant + Unused，而不是 Mislocalized"——这直接决定 §9 的 reward 该往哪使劲。如果发现 Mislocalized 才是主因，方法重心要移到 Acquisition。

---

## 9. 方法：Evidence-aware RL

### 9.1 设计原则

- 奖励里**没有 entropy 项**。任何形式的"奖励 entropy 下降"都会被 False Resolution hack。
- 用**反事实分叉**制造"同一前缀、不同证据、应得不同答案"的训练信号——这是唯一能在梯度上区分"因看而对"与"先答后看"的方式。
- Necessity 靠 $N=0$ 对照与调用成本教，不靠 entropy 门控。

### 9.2 训练样本与 rollout

训练集 = 认证 $N=1$ 孪生对 + 等量 $N=0$ 对照。每组 $G$ 条 rollout，在**首次工具返回处**对其中比例 $\rho$ 的 rollout 分叉，同一前缀接三种返回之一：

| 分支 | 工具返回 | 该分支上的正确答案 | 教什么 |
|---|---|---|---|
| real | 策略自己 $e(a)$ | $y^*$ | 正常 |
| twin | 孪生的 $E'^*$（同位置同尺寸） | **$y'$** | 答案必须跟着证据走 |
| decoy | 非判别区域高清裁剪 | $y^*$，允许再调用一次 | 不能盲从，要再找 |

### 9.3 奖励

$$R = R_{\text{acc}} + \beta\,R_{\text{rev}} + \lambda\,R_{\text{acq}} - c\cdot n_{\text{calls}} \;[\;+\;\mu\,R_{\text{PEA}}\;]$$

- $R_{\text{acc}}$：各分支按**该分支的正确答案**判对错（MCQ 直接比选项；自由文本沿用双向子串）。twin 分支答 $y'$ 才得分——这一项就把"忽略工具"打掉一半期望奖励。
- $R_{\text{rev}}$（belief revision）：$\mathbb{1}[\text{real 答对}\wedge\text{twin 答案}\neq\text{real 答案}] - \gamma\,\mathbb{1}[\text{real 答错}\wedge\text{twin 答案}=\text{real 答案}]$。只在分叉组上计算。
- $R_{\text{acq}}$：调用窗与 $E^*$ 的时空命中，**只在课程前期开、权重小**，后期关掉——防止模型背类别级位置。
- $c\cdot n_{\text{calls}}$：$N=0$ 样本上"不调且对" > "调且对"，Necessity 由此学出。
- $R_{\text{PEA}}$（可选消融臂）：prototype bank 只从**审计通过**的轨迹（$N=1$ ∧ 命中 $E^*$ ∧ 答对 ∧ twin 敏感）建，按 PEA 三分支奖励对齐，动态更新。这是 PEA 在训练侧的唯一入口，且只付给分叉检验通过的 rollout，防止"只学熵的形状不学内容"。

### 9.4 每一种 hack 对应哪一项挡

| 退化策略 | 被哪项挡 |
|---|---|
| 永远调用 | $N=0$ 上的调用成本 |
| 永远不调用 | $N=1$ 上 $R_{\text{acc}}$ 低；twin 分支无法完成 |
| 调了不看（先答后看） | twin 分支必错 → $R_{\text{acc}}$、$R_{\text{rev}}$ 双罚 |
| 盲从工具返回 | decoy 分支答错；且 Mislocalized 的返回不含信息，盲从无收益 |
| 背 $E^*$ 位置 | 孪生对共享位置，位置本身不给答案；$R_{\text{acq}}$ 前期即关；eval 用留出类别 |
| 学熵的形状不学内容（若开 PEA 臂） | PEA 奖励门控在 twin 敏感通过的 rollout 上；动态 prototype |

### 9.5 训练流程

冷启动 SFT（工具格式，几百条，不是贡献）→ Evidence-aware RL（GRPO/GSPO，单阶段，全部奖励项同时开，$R_{\text{acq}}$ 按课程衰减）。我之前的两阶段 curriculum 作为一个 baseline 保留，不再是主方法。

---

## 10. 主实验

**数据**：EvidenceGap-V（test 留出类别）、VideoNet mcq_test（外部泛化）。

**行**：Base direct；Base + tools（零样本）；DeepEyes-7B；VideoSearcher-8B；E2E RL（DeepEyes 奖励）；BiSPO 形 RL（复现奖励）；两阶段 acc RL（我 08 月的）；Evidence-aware RL（本文）；+PEA 臂；oracle 上界（喂 $E^*$）；匹配预算密采无工具。

**列**：EvidenceGap-V 上 acc(∅) / acc(agentic) / Tool Necessity Gain / 时空命中 / EEG$_{\text{oracle}}$ / EEG$_{\text{policy}}$ / Twin Sensitivity / False Resolution / Redundant Rate；VideoNet 上 acc / 调用率 / definition 标注子集上的 EEG。

**要证的那一格**：本文方法是唯一 EEG$_{\text{policy}}$ 显著 > 0、Twin Sensitivity 高、False Resolution 低三者同时成立的行；且 Tool Necessity Gain 超过匹配预算密采基线。

**VideoNet 的控制实验**（防"基模太弱所以工具有用"）：按 base direct acc 分层报增益；换更强 base（Qwen3-VL-32B）重跑；只在参考模型可解子集上报；匹配预算密采基线同样在 VideoNet 上跑。

---

## 11. 消融

去 twin 分支 / 去 decoy 分支 / 去 $N=0$ 对照 / $R_{\text{acq}}$ 全程开 vs 前期开 vs 不开 / $\rho\in\{0.25,0.5,1.0\}$ / $\beta,\gamma$ / 调用预算 / 只时间工具 vs 只空间工具 vs 合一 / 只用来源 A 训 vs 只用 B / 有无冷启动 / PEA 臂 static vs dynamic（复现 PEA 自己的 hack 曲线在这个任务上是否再现）。

每个消融都要报 EEG 与 False Resolution，不只报 acc——很多消融的 acc 差不多，差别在证据是否真的被用。

---

## 12. 风险与补强

| 风险 | 补强实验 |
|---|---|
| 样本是否真的 Evidence-Required | C1–C4 认证协议全量执行并报通过率；发布时只保留全过样本；附评审者一致性 |
| 语言 / 视频捷径 | C3 文本-only、单帧-only、打乱帧序、只给孪生之外的帧；训练用留出类别测试 |
| 粗观测真的不可辨？ | 扫 $K\in\{8,16,32\}$ 帧与三档分辨率，报 $N=1$ 通过率随观测强度的衰减曲线；论文只报在指定观测预算下的 necessity |
| PEA entropy 能否表示 uncertainty dynamics | 把它当假设（H1–H3）测，不当前提；答案信念与 token 熵分开报；若 H3 不成立就降级为可视化 |
| 错误答案高度确信 | 这正是 Necessity 定义在样本上的理由；Overconfident Skip 作为一个被测量的失败模式，不作为要"修"的前提 |
| decoy 下 entropy 错误下降 | False Resolution Rate 是一等指标；decoy 分支进训练；报 decoy 与 twin 的双条件矩阵 |
| 是否真需要时间+空间 | 消融只时间 / 只空间 / 合一；按场景类型分层（遮挡解除偏时间、局部文字偏空间） |
| 自建 benchmark 太人工 | 来源 A 自然对为主，B 为对照；主结论必须在 A 上成立；VideoNet definition 子集第三方复验 |
| 孪生对不够自然 | 报每对的粗观测相似度（CLIP / 帧差）分布；人工"哪个是编辑过的"二选一 ≈ chance |
| VideoSearcher 已覆盖部分问题 | 直接把它放进第一层实验测 EEG；它的"purposeful"若 EEG≈0 就是本文最有力的动机 |
| 训练提升有限 | 论文重心在框架 + benchmark + 诊断（贡献 1–3），方法是贡献 4；即便方法只把 EEG 从 0 拉到显著正而 acc 涨幅小，也是完整故事；但审稿人会要 acc，所以匹配预算基线必须赢 |
| judge 噪声 | 主指标全部 judge-free（MCQ 信念）；judge 只用于自由文本伴随指标，报双遍一致率 |

---

## 13. 我的判断

**真正有论文价值的核心点**是 §1 的三层框架 + §4.2 的 twin/decoy 双条件反事实设计。它把"工具有没有用"从一个靠调用率和 acc 暗示的问题，变成三个可以分别证伪、且退化解都被排除的测量。这一点 DeepEyes、VideoSearcher、FaithEyes 都没有，CoT faithfulness 那条线也没做到视觉证据上。PEA 是这个框架里的诊断显微镜，不是主角；如果 H3 成立，它会是很好的分析章节，如果不成立，论文不受影响。

**要到 CVPR / NeurIPS / ICLR 的完整度，现在最缺的：**

关键实验（按优先级）：
1. **Benchmark 本体**：≥1,000 对通过 C1–C4 的自然孪生对。没有它，一切都是我 08-28 那种"行为证据"。这是最大的工作量，也是最不可替代的贡献。
2. **第一层现象表**：≥4 个公开工具模型在同一 pilot 上的 EEG$_{\text{policy}}$、Twin Sensitivity、False Resolution。我现在只有 DeepEyes 一家 + 自己的检查点；VideoSearcher 是必须测的。
3. **匹配预算的密采无工具基线**：这是审稿人第一个会问的对照，现在完全没有。

方法创新（现在只有雏形的部分）：
1. **twin 分支奖励**——"同一前缀，孪生证据，应答孪生答案"。这是本文在训练侧唯一真正新的东西，它把 evidence-consistency 变成了可优化目标，且不需要 entropy、不需要 judge。08-28 的 sham 分叉只能罚"不变"，twin 分叉能奖"该变则变"，强得多。
2. **Necessity 的学习**：让模型在 $N=0$ 上少调、$N=1$ 上多调，目前只靠调用成本，比较粗；如果第一层实验发现 Redundant 是主因，这里需要更强的机制（例如把 $N$ 的判断显式化为一个可奖励的中间输出），这是留给后续的空白。

一个诚实的边界：如果 VideoNet 上的 $N=1$ 样本比例很低（07-23 探针 direct 已 70%），VideoNet 就只能做泛化验证，不能做主战场——这与上面的定位一致，但意味着自建 benchmark 的质量决定论文上限。

---

## 14. 下一步（不占训练卡的部分先做）

1. 从 VideoNet 元数据挖同域易混类三元组，用 `definition` 生成 $E^*$ 提议，先做 50 对走一遍 C1–C4，看认证通过率——这个数字决定来源 A 可行不可行。
2. `steplib` 加 `evidence_mode ∈ {true, twin, decoy, irr, none}` 与首次返回处分叉，MCQ 选项 logits 抽取 $b_0,b_1$；在阶段一 860 / 阶段二 960 / DeepEyes-7B 上跑 pilot，出第一张 EEG + 四模式矩阵。
3. 用 pilot 轨迹算 PEA signature，先看 H2（塌缩）是否在图上直接可见，再决定 H1/H3 值不值得做聚类。
