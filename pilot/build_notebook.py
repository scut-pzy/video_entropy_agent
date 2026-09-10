"""生成 evidencegap_pilot.ipynb (只建骨架; 执行用 `CUDA_VISIBLE_DEVICES=1 jupyter execute --inplace evidencegap_pilot.ipynb`)."""
import nbformat as nbf, os

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'name': 'ms_swift', 'display_name': 'ms_swift', 'language': 'python'}
C = []

C.append(nbf.v4.new_markdown_cell("""# EvidenceGap-V pilot — 用 DeepEyes-7B 走一遍认证（中文版）

题面、选项、提示词、工具说明全部是中文，并要求模型**用中文思考**（`ev.set_lang('zh')`；改成 `'en'` 就是英文原版）。视频里的数字和图形与语言无关，中英文用的是同一批视频。

这个 notebook 把 `eval_deepeyes.py` 的验证过程摊开看：

1. **数据长什么样**：每条样本的孪生对 A/B——模型实际看到的 8 帧粗采样（≤256·28² 像素）+ E\\* 证据帧（原生分辨率裁剪）。
2. **六个证据条件**：同一粗观测前缀，只换追加的"工具返回帧"（无 / 真 / 孪生 / 诱饵 / 灰图），看模型的答案信念怎么变。信念 = 四个选项字母首 token 的 softmax，对选项顺序做 4 个循环置换取平均（judge-free）。
3. **DeepEyes 自己调工具**：给它 zoom + seek 两个工具，逐轮看 think / tool_call / 返回的裁剪图 / 最终答案。
4. **C1–C4 认证汇总**。

判据（`eval_deepeyes.certify`）：C1 粗观测不可辨 = A/B 信念 TV<0.15 且孪生对平均 p(y\\*)≤0.6；C2 证据充分 = `true` 下都答对且 p(y\\*)≥0.6；C3 无文本捷径 = `text` 下同 C1；C4 证据可交换 = `twin` 下都翻到孪生答案且 p≥0.6；decoy 不产生错误确信。

样本设计与迭代记录见 `README.md`。"""))

C.append(nbf.v4.new_code_cell("""import os, sys, json
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')   # 登录节点 GPU1 通常空闲; 交互运行前按需改
HERE = os.getcwd()
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import eval_deepeyes as ev, pilot_viz as pv, video_imcot as vi
ev.set_lang('zh')      # 题面 / 提示词 / 工具说明全部中文, 并要求模型用中文思考; 改成 'en' 即原版
ev.enable_stop_at_tool_call(True)   # agentic 生成在 </tool_call> 处截停, 掐掉 DeepEyes 的 'addCriterion' 退化输出

import logging, transformers
transformers.logging.set_verbosity_error()            # 关掉 "Setting pad_token_id to eos_token_id" 这类每次 generate 都刷的提示
transformers.utils.logging.disable_progress_bar()     # 关掉权重加载进度条
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

samples = pv.load_samples()
print('samples:', [s['id'] for s in samples])
vi.load_model()
vi.model.generation_config.pad_token_id = vi.processor.tokenizer.eos_token_id   # 显式给定, 提示的根源
from pprint import pprint
letter_ids = ev._letter_ids()
print('选项字母对应的 token id（含带空格变体）:'); pprint(letter_ids)"""))

C.append(nbf.v4.new_markdown_cell("""## 1. 数据：孪生对长什么样

每条样本两行图：上面 8 张是模型在 `none` 条件下**实际看到的**帧（已经缩到评测分辨率），下面 4 张是 E\\* 证据帧（在 `true` 条件下追加给模型的）。
S1/S3/S3b 的证据窗口落在两个粗采样点之间，所以 8 帧里根本没有；S2/S2b 的标签全程在画面里，但缩放后只有 ~4px 高。"""))

C.append(nbf.v4.new_code_cell("""for s in samples:
    pv.show_sample(s)"""))

C.append(nbf.v4.new_markdown_cell("""## 2. 六个证据条件下的信念

每条样本：先展示各条件追加给模型的帧（`text`/`none` 不追加），然后一张信念表 + 一张柱状图。
要看的形状：`none` 下 A/B 两行几乎一样且都在 chance 附近；`true` 下绿色列接近 1；`twin` 下 p(twin's answer) 接近 1；`decoy`/`irr` 回到 `none` 的形状。"""))

C.append(nbf.v4.new_code_cell("""from IPython.display import display, Markdown
all_res = {}
for s in samples:
    display(Markdown(f"---\\n## `{s['id']}`  — A: {s['twins']['A']['answer']} / B: {s['twins']['B']['answer']}"))
    res = pv.run_conditions(s, letter_ids)
    pv.plot_beliefs(res, s)
    all_res[s['id']] = res"""))

C.append(nbf.v4.new_markdown_cell("""## 3. 模型的推理过程：无工具直接推理 vs 带工具的主动推理

每条样本的每个孪生各看两段：

- **无工具直接推理**：只给 8 帧粗采样，要求先在 `<think>` 里用中文推理再作答——看它在证据缺失时是怎么"想"出答案的。
- **带工具的主动推理**：给 zoom + seek 两个工具，让它自己决定看哪一帧、哪个区域、哪个时间段。每一轮显示：`<think>` 的完整内容 → 工具调用及参数 → 工具返回的帧 → 最后一轮的 `<answer>`。

重点看三件事：(a) 首轮 think 里是不是已经先写了结论再调工具；(b) zoom/seek 的位置有没有落到 E\\*（时间窗 × 空间框见第 1 节）；(c) 拿到工具返回的图之后，第二轮 think 有没有真的根据图改判断。"""))

C.append(nbf.v4.new_code_cell("""from IPython.display import display, Markdown
trajs, direct = {}, {}
for s in samples:
    display(Markdown(f"---\\n## `{s['id']}`"))
    for v in 'AB':
        direct[(s['id'], v)] = pv.run_direct_cot(s, v)
        trajs[(s['id'], v)] = pv.run_agentic(s, v)"""))

C.append(nbf.v4.new_markdown_cell("""## 4. 认证汇总"""))

C.append(nbf.v4.new_code_cell("""df = pv.cert_table(all_res)
n_direct = sum(t['correct'] for t in direct.values()); n_ok = sum(t['correct'] for t in trajs.values())
print(f'无工具直接推理 正确: {n_direct}/{len(direct)}   |   带工具主动推理 正确: {n_ok}/{len(trajs)}')
import json
json.dump({k: v for k, v in all_res.items()}, open(os.path.join(HERE, 'eval', 'results_notebook.json'), 'w'),
          indent=1, ensure_ascii=False, default=str)
json.dump({f'{k[0]}/{k[1]}': v for k, v in trajs.items()}, open(os.path.join(HERE, 'eval', 'trajs_notebook.json'), 'w'),
          indent=1, ensure_ascii=False, default=str)      # 完整轨迹 (每轮原文) 也存一份, 以后不用重跑模型就能重排版"""))

nb['cells'] = C
out = os.path.join(HERE, 'evidencegap_pilot.ipynb')
nbf.write(nb, out)
print('->', out)
