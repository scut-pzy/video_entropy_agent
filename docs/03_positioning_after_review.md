# 相关工作对照与仍开放的点（2026-09-11，基于外部 review + 原文核实）

## 四篇直接相关工作，逐条对照

| | Illusion of Visual Tool-Use (2608.06270) | Evidence-RL (2608.08021) | DeFacto (2509.20912) | AVP (2512.05774) | 我们已有 | 我们还能做 |
|---|---|---|---|---|---|---|
| 模态 / 工具 | 图像，只有 crop-zoom | 图像 | 图像 | 长视频，时间区间 + fps + 分辨率 | 视频，zoom + seek | **时空定位**是主角 |
| 干预 | 轨迹级在线替换（随机裁剪/噪声/黑图）并续生成；步级固定前缀 + `<answer>` 读 logit | 遮除证据区 vs 匹配非证据区 | 正例/反事实/随机遮挡 | 无反事实 | 离线补帧读信念（被 review 指出不等于真实轨迹） | **真实轨迹在工具返回处分叉**（本轮做） |
| 指标 | VEG = 真实返回的边际增益 − 反事实返回的边际增益（概率差） | 支持度下降差 | 证据-答案一致性 | 准确率 + token/时间 | EEG（真/假窗的 log p 差） | **熵响应 EER**：返回后推理段的熵在真/随机返回下的差（无需 MCQ 读出，可用于开放题） |
| 不确定性信号 | 概率差、饱和阈值 0.95；**不用熵** | 无 | 无 | 模型自报置信度 | PEA 式 token 熵 signature | 熵作为**连续、无标签、无读出**的信号：定位不确定发生在轨迹的哪一步 |
| 样本级 necessity | **无**（只事后诊断策略校准） | 无 | 无 | 无 | 窗搜索认证（模型相对） + 合成孪生对（严格） | 保留：严格版只在孪生对；自然集改名"证据增益样本" |
| 训练 | **无**（列为未来工作） | GRPO | GRPO | 无（prompting，Gemini） | 无 | **SFT 数据收集 + 受控 RL**（Illusion 明确说这是它没做的） |
| 效率 | 无 | 无 | 无 | 12.4% token、5.4× 提速（但无因果验证） | dense_matched 对照 | 学到的定位策略 vs 简单基线的 Pareto |

## 还开着的点（都能从上面两篇的"limitation"里找到原话支撑）

1. **视频/时间工具的因果审计**。Illusion 原话："Other tools such as … frame selection in video … may exhibit qualitatively different calibration behaviors." 我们的 seek/zoom 双工具 + E* 窗就是这个。
2. **样本级 necessity**。Illusion 只有事后的策略诊断（no-call / calling-without-looking / looking-without-planning / calibrated），没有"这题需不需要"的样本性质。我们的窗搜索 + 孪生对补上这一层——但要按 review 的口径：自然集是"模型筛选的证据增益样本"，严格不可辨只在孪生对上主张。
3. **熵作为信号**。Illusion 用概率差（要 MCQ、要在固定点读 `<answer>` 的 logit）；AVP 用自报置信度。token 熵是连续的、免标签的、开放题也能算的。要证明它有用，必须回答 review 的质疑（熵含措辞/格式噪声）——办法是**对比**：同一前缀、只换返回内容，格式噪声在两支里相同，差值只剩证据的作用。这就是 EER。
4. **受控训练**。Illusion："Definitively isolating the role of outcome-only rewards would require controlled training studies … beyond the diagnostic scope of this paper." AVP 的未来工作也是 learned policies。我们有认证过的题、E*、分叉实验——正好是 SFT 数据和 RL 奖励的原料。
5. **效率的因果版本**。AVP 报了 token 省了多少，但没验证收益来自取回的片段。我们做"定位策略 vs 均匀/随机/运动选帧"的 Pareto，且每一格都过分叉检验。

## 一句话定位（改后）

> 在初始观测缺信息的视频问答里，模型能否在有限预算下**定位**并**使用**缺失的时空证据？我们用真实轨迹分叉测证据依赖，用熵响应刻画不确定性何时被证据解决，用认证样本构造 SFT/RL 数据，并证明学到的定位策略优于简单基线且收益依赖所取证据。

## 本轮要交的 case 数据
- `natural/out/fork.jsonl`：62 条真实轨迹在首次工具返回处分叉（real / random / gray），记录答案翻转、步级概率差（VEG 口径）、返回后推理段熵（EER 口径）。
- `natural/out/sft_pilot.jsonl`：从认证题 + E* 合成的"理想取证轨迹"（粗看→写出缺什么→seek E*→据证据作答），DeepEyes 工具格式，可直接 SFT。
