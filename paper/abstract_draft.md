# Abstract（2026-09-11 草稿，268 tokens）

Recent work teaches multimodal models to think with videos through reinforcement learning, letting them zoom into frames or replay segments, and reports higher tool-call rates and accuracy. We show that these gains do not imply that tool outputs inform the answer. On videos whose sparse frames lack the decisive evidence, an RL-trained agent commits to an answer before it calls a tool: replacing the returned frames with random frames, blank frames, or even the missing evidence leaves 91% of its answers unchanged. Its token entropy tells the same story, with uncertainty peaking and collapsing before the call rather than after the evidence arrives, and removing the premature conclusion restores the dependence of answers on what the tool returns. Inspired by this, we propose ERA (Entropy-Resolved evidence Acquisition), a framework that treats tool use as the resolution of uncertainty. ERA selects trajectories whose entropy drops in response to genuine evidence but not to counterfactual returns as cold-start data, and then reinforces evidence acquisition under a visual budget. On a VideoNet subset whose answers are unrecoverable from sparse frames but recoverable from a short localized segment, ERA improves accuracy by [X] points, and applying its RL stage to the diagnosed agent recovers [Y] points. By grounding tool calls in the uncertainty that evidence resolves, ERA offers a pathway toward video agents that look before they conclude.

## 每句对应的证据状态

| 句子 | 状态 | 依据 |
|---|---|---|
| 91% 答案不变 | 已有 | `natural/out/fork_v2.jsonl`，DeepEyes，22 条 |
| 熵在调用前达峰并塌下去 | 已有（描述性） | 62% top-20% 高熵 token 在调用前；调用前 0.84 → 后 0.31 |
| 去掉结论后答案重新依赖返回 | 已有 | `natural/out/decommit.jsonl`：中性思考下 E* 对/随机错 6 : 0，p=0.03；原文思考 1 : 1 |
| 按"真证据使熵下降、反事实不降"筛冷启动数据 | **待验证** | DeepEyes 上 EER≈0（去结论后 1.177 vs 1.175）；取决于 Qwen3.5 结果，否则改用"答案随证据变"筛选、熵作诊断 |
| 冷启动 + RL，子集提升 [X] | **未做** | 需：数据扩充 → SFT → RL → 在认证子集评测 |
| 对 DeepEyes 做 RL 提升 [Y] | **未做** | |
| ERA 名字 | 占位 | 可改 |
