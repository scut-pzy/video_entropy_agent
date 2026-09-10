"""用 DeepEyes-7B 认证 EvidenceGap-V pilot 样本: 粗观测不可辨 (C1) / 证据充分 (C2) / 无文本捷径 (C3) / 证据可交换 (C4).

每条样本 × 孪生 A/B × 条件:
  text       只有题目, 没有帧                               → C3
  none_k8    8 帧均匀粗采样 (评测预算)                        → C1
  none_k16 / none_k32   更密的均匀采样 (necessity 随观测预算的衰减曲线)
  true       粗采样 + E* 证据帧 (本视频, 窗口内, 裁到证据框, 高分辨率)    → C2
  twin       粗采样 + 孪生视频的 E* 证据帧                      → C4 (应翻到孪生答案)
  decoy      粗采样 + 同框同尺寸但不含判别信息的帧                → 不应收敛
  irr        粗采样 + 同尺寸灰图                              → 不应变化
  agentic    DeepEyes 自己用 zoom+seek 工具 (信息项: Acquisition)

信念 b = 四个选项字母首 token 的 softmax (judge-free), 对选项顺序做 4 个循环置换取平均 (去字母偏置).
用法: CUDA_VISIBLE_DEVICES=1 python eval_deepeyes.py --tag v1 [--only s1_occlusion_color] [--no-agentic]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import video_imcot as vi  # noqa: E402

vi.VIDEONET = HERE            # 让 harness 从 HERE/videos/ 读视频
N_EVID = 4                    # 证据帧数
LETTERS = 'ABCD'

# ─────────────────────────── 语言: en (原版) / zh (题面、提示词、要求模型用中文思考) ───────────────────────────
LANG = 'en'
_HEADER_EN = vi._header


def _header_zh(sess):
    n = sum(1 for f in sess.frames if f['kind'] == 'coarse')
    return f'以下是从一段 {sess.duration:.1f} 秒的视频中均匀抽取的 {n} 帧画面，每帧都标注了序号和时间戳。\n'


def _tools_zh():
    zoom = json.loads(vi.ZOOM_TOOL_JSON); seek = json.loads(vi.SEEK_TOOL_JSON)
    zoom['function']['description'] = ('通过边界框 (bbox_2d) 裁剪并放大某一视频帧的指定区域，用于查看手部、物体或身体部位等细节。'
                                       '用帧标签中显示的序号来指定帧。')
    zoom['function']['parameters']['properties']['frame_index']['description'] = '要裁剪的帧的序号，即帧标签中显示的数字，例如 [第3帧] 对应 3。'
    zoom['function']['parameters']['properties']['bbox_2d']['description'] = '要放大的区域的边界框 [x1, y1, x2, y2]，(x1, y1) 为左上角，(x2, y2) 为右下角。'
    zoom['function']['parameters']['properties']['label']['description'] = '该区域内物体的名称或标签（可选）。'
    seek['function']['description'] = '以更密集的采样帧回放视频中的一个时间段，用于仔细查看快速或短暂的动作。当关键时刻落在你已看到的帧之间时使用。'
    seek['function']['parameters']['properties']['start_time']['description'] = '时间段起点（秒）。'
    seek['function']['parameters']['properties']['end_time']['description'] = '时间段终点（秒）。'
    seek['function']['parameters']['properties']['num_frames']['description'] = '在该时间段内采样的帧数（2-6，默认 6）。'
    return json.dumps(zoom, ensure_ascii=False) + '\n' + json.dumps(seek, ensure_ascii=False)


SYS_ZH_TMPL = """你是一个乐于助人的助手。

# 工具
你可以调用一个或多个函数来协助回答用户的问题。
可用的函数签名在 <tools></tools> XML 标签内：
<tools>
{tools}
</tools>

# 如何调用工具
在 <tool_call></tool_call> XML 标签内返回一个包含函数名和参数的 json 对象：
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

**示例**：
<tool_call>
{{"name": "image_zoom_in_tool", "arguments": {{"frame_index": 2, "bbox_2d": [10, 20, 100, 200], "label": "右手"}}}}
</tool_call>"""
INSTR_ZH = ('\n请先用中文思考；如有需要，调用 **image_zoom_in_tool**（放大某一帧的某个区域）或 '
            '**video_seek_tool**（密集回看某个时间段），然后作答。'
            '格式严格为：<think>...</think> <tool_call>...</tool_call>（需要工具时）<answer>...</answer>（只填选项字母）')
CONT_ZH = ('以上是你调用工具后返回的画面。请继续用中文思考；如有需要可继续调用工具，否则给出最终答案。'
           '格式严格为：<think>...</think> <tool_call>...</tool_call> 或 <answer>...</answer>')
PRE_ANSWER_RE = r'answer is|correct answer|答案是|答案应该是|正确答案|应该选|应选|选项\s*[ABCD]\b|是\s*[ABCD]\b'


_FRAME_LABEL_EN = vi.VideoSession.frame_label


def _frame_label_zh(self, i):
    import re
    f = self.frames[i]
    note = (f['note'].replace('dense replay', '密集回放').replace('tool-returned crop', '工具返回的裁剪'))
    note = re.sub(r'zoom of Frame (\d+)', r'放大自第\1帧', note)
    return f"[第{i}帧, t={f['t']:.1f}s{', ' + note if note else ''}]"


def set_lang(lang):
    """切换 en/zh: 影响题面、粗采样开头语、帧标签、工具返回说明、agentic 的 system/instruction/continue 提示词."""
    global LANG
    assert lang in ('en', 'zh')
    LANG = lang
    vi._header = _header_zh if lang == 'zh' else _HEADER_EN
    vi.VideoSession.frame_label = _frame_label_zh if lang == 'zh' else _FRAME_LABEL_EN


_GENERATE_ORIG = vi._generate


@torch.inference_mode()
def _generate_stop(messages, max_new_tokens=None):
    """与 vi._generate 相同, 但在 </tool_call> 处停止生成.
    DeepEyes 在 </tool_call> 后不会自己停, 会继续吐 'addCriterion'/重复调用/伪造的 <answer> (07-30 报告里 100% 的 episode 都有),
    把这些垃圾回灌进上下文会让 run_episode 误以为已经作答. 掐掉后 acc 变化在噪声内、速度快 4 倍."""
    from qwen_vl_utils import process_vision_info
    text = vi.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vi.processor(text=[text], images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors='pt').to(vi.model.device)
    out = vi.model.generate(**inputs, max_new_tokens=max_new_tokens or vi.MAX_NEW_TOKENS, do_sample=False,
                            stop_strings=['</tool_call>'], tokenizer=vi.processor.tokenizer)
    return vi.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def enable_stop_at_tool_call(on=True):
    vi._generate = _generate_stop if on else _GENERATE_ORIG


def _opts(sample):
    return sample['options_zh'] if LANG == 'zh' else sample['options']


def _question(sample):
    return sample['question_zh'] if LANG == 'zh' else sample['question']


def _ans(sample, variant):
    return _opts(sample)[sample['twins'][variant]['answer_idx']]


def _letter_ids():
    tok = vi.processor.tokenizer
    ids = {}
    for L in LETTERS:
        s = set()
        for v in (L, ' ' + L):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1:
                s.add(e[0])
        ids[L] = sorted(s)
    return ids


@torch.inference_mode()
def belief_and_greedy(messages, letter_ids):
    from qwen_vl_utils import process_vision_info
    text = vi.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vi.processor(text=[text], images=image_inputs, videos=video_inputs,
                          padding=True, return_tensors='pt').to(vi.model.device)
    out = vi.model.generate(**inputs, max_new_tokens=8, do_sample=False,
                            output_scores=True, return_dict_in_generate=True)
    logits = out.scores[0][0].float()
    lp = torch.log_softmax(logits, -1)
    per = {L: torch.logsumexp(lp[ids], 0).item() for L, ids in letter_ids.items()}
    z = np.array([per[L] for L in LETTERS]); z = np.exp(z - z.max()); z /= z.sum()
    gen = vi.processor.batch_decode(out.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    m = [c for c in gen.strip()[:3] if c in LETTERS]
    return z, (m[0] if m else None), gen.strip()


def _q_text(question, options, suffix=True):
    body = question + '\n' + '\n'.join(f'{L}. {o}' for L, o in zip(LETTERS, options))
    if not suffix:
        return body
    return body + ('\n请只回答正确选项的字母。' if LANG == 'zh' else
                   '\nPlease respond with only the letter of the correct answer.')


def _evidence_frames(sess, video_path, window, bbox, n=N_EVID, gray=False):
    """从 video_path 的 window(帧区间) 里均匀取 n 帧, 裁 bbox, 用 detail 分辨率注册到 sess; 返回索引."""
    import decord
    vr = decord.VideoReader(video_path); fps = vr.get_avg_fps()
    w0, w1 = window if window else (0, len(vr))
    idx = np.linspace(w0, w1 - 1, n).astype(int)
    out = []
    for fi in idx:
        pil = Image.fromarray(vr[int(fi)].asnumpy()).crop(tuple(bbox))
        if gray:
            pil = Image.new('RGB', pil.size, (128, 128, 128))
        out.append(sess._register(fi / fps, pil, 'evidence', sess.detail_pixels, note='tool-returned crop'))
    return out


def run_condition(sample, variant, cond, letter_ids, k=8, viz=None):
    """viz: 传入一个 dict 则把模型实际看到的图 (coarse / evidence 的 pil_disp) 和说明文字写进去, 供 notebook 展示."""
    vids = sample['twins']; me = vids[variant]; other = vids['B' if variant == 'A' else 'A']
    my_path = os.path.join(HERE, 'videos', me['video']); ot_path = os.path.join(HERE, 'videos', other['video'])
    opts = _opts(sample); my_ans = opts[me['answer_idx']]; ot_ans = opts[other['answer_idx']]
    ew, eb = sample['evidence_window_frames'], sample['evidence_bbox']
    dw, db = sample['decoy_window_frames'], sample['decoy_bbox']

    sess = vi.VideoSession(my_path, n_coarse=k) if cond != 'text' else None
    evid = []
    if cond == 'true':
        evid = _evidence_frames(sess, my_path, ew, eb)
    elif cond == 'twin':
        evid = _evidence_frames(sess, ot_path, ew, eb)
    elif cond == 'decoy':
        evid = _evidence_frames(sess, my_path, dw if dw else None, db)
    elif cond == 'irr':
        evid = _evidence_frames(sess, my_path, ew, eb, gray=True)
    coarse = [i for i, f in enumerate(sess.frames) if f['kind'] == 'coarse'] if sess else []
    caption = ''
    if evid:
        t0, t1 = sess.frames[evid[0]]['t'], sess.frames[evid[-1]]['t']
        caption = (f'以下是视频检视工具返回的补充画面（时间窗 {t0:.1f}s–{t1:.1f}s，裁剪区域 {list(eb)}）：' if LANG == 'zh' else
                   f'Additional frames returned by the video inspection tool '
                   f'(time window {t0:.1f}s-{t1:.1f}s, cropped to region {list(eb)}):')
    if viz is not None:
        viz['coarse'] = [sess.frames[i]['pil_disp'] for i in coarse]
        viz['coarse_labels'] = [sess.frame_label(i) for i in coarse]
        viz['evidence'] = [sess.frames[i]['pil_disp'] for i in evid]
        viz['evidence_labels'] = [sess.frame_label(i) for i in evid]
        viz['caption'] = caption

    beliefs, greedy, raw = [], [], []
    for shift in range(4):                         # 选项循环置换, 去字母偏置
        perm = [(j + shift) % 4 for j in range(4)]  # 位置 j 放原选项 perm[j]
        q = _q_text(_question(sample), [opts[i] for i in perm])
        content = []
        if sess is not None:
            content += [{'type': 'text', 'text': vi._header(sess)}] + vi._frames_content(sess, coarse)
        if evid:
            content += [{'type': 'text', 'text': caption}]
            content += vi._frames_content(sess, evid)
        content += [{'type': 'text', 'text': ('问题：' if LANG == 'zh' else 'Question: ') + q}]
        z, g, gen = belief_and_greedy([{'role': 'user', 'content': content}], letter_ids)
        b = np.zeros(4)
        for j, i in enumerate(perm):
            b[i] = z[j]
        beliefs.append(b); greedy.append(opts[perm[LETTERS.index(g)]] if g else None); raw.append(gen)
    b = np.mean(beliefs, 0)
    return dict(belief={o: round(float(v), 4) for o, v in zip(opts, b)},
                greedy=greedy, pred=max(set(greedy), key=greedy.count) if any(greedy) else None,
                p_gt=round(float(b[me['answer_idx']]), 4),
                p_twin=round(float(b[other['answer_idx']]), 4),
                n_frames=len(sess.frames) if sess else 0)


def agentic_prompts():
    """agentic 模式的 system / instruction / continue 提示词 (按 LANG)."""
    if LANG == 'zh':
        return dict(system_prompt=SYS_ZH_TMPL.format(tools=_tools_zh()), instruction=INSTR_ZH, cont_prompt=CONT_ZH)
    return {}


def make_item(sample, variant):
    me = sample['twins'][variant]
    q = _q_text(_question(sample), _opts(sample), suffix=(LANG == 'en'))   # zh 的作答要求已在 INSTR_ZH 里
    return {'question': [{'text': q}, {'video': me['video']}], 'answer': me['answer_letter']}


def run_agentic(sample, variant, verbose=False, max_turns=4):
    item = make_item(sample, variant)
    traj, sess = vi.run_episode(item, 'full', n_coarse=8, max_turns=max_turns, verbose=verbose, **agentic_prompts())
    calls = [dict(name=c['name'], ok=c['ok'], raw=c['raw'][:200], err=c['err']) for c in traj['tool_calls']]
    return dict(pred=traj['pred'], correct=traj['correct'], turns=traj['turns'], tool_calls=calls)


def tv(b1, b2):
    return 0.5 * sum(abs(b1[k] - b2[k]) for k in b1)


def certify(res):
    """对一条样本 (A/B 都有) 打 C1-C4.

    C1/C3 按方案 §3 的定义是"对内二选一 acc ≈ chance", 即 **孪生对平均** 的 p_gt ≤ 0.5+margin, 且 A/B 的信念几乎相同
    (TV 小). 模型对某个选项有语言先验 (如 "right" > "left") 时, 单边 p_gt 会高于 0.5 但配对平均仍是 chance——
    这是先验导致的 over-confidence, 属于要测量的现象而不是样本缺陷, 单独用 prior_bias 报出来.
    decoy 检查的是"不能产生错误的确信" (False Resolution), 而不是"信念不许动": 看到不含证据的帧后变得更不确定是合理的.
    """
    A, B = res['A'], res['B']
    ans = res['answers']
    out = {}
    pair_gt = lambda c: 0.5 * (A[c]['p_gt'] + B[c]['p_gt'])
    out['C1_coarse_indistinct'] = (tv(A['none_k8']['belief'], B['none_k8']['belief']) < 0.15
                                   and pair_gt('none_k8') <= 0.6)
    out['C2_evidence_sufficient'] = all(v['true']['pred'] == ans[k] and v['true']['p_gt'] >= 0.6
                                        for k, v in (('A', A), ('B', B)))
    out['C3_no_text_shortcut'] = (tv(A['text']['belief'], B['text']['belief']) < 0.15
                                  and pair_gt('text') <= 0.6)
    out['C4_evidence_swappable'] = all(v['twin']['pred'] == ans['B' if k == 'A' else 'A']
                                       and v['twin']['p_twin'] >= 0.6 for k, v in (('A', A), ('B', B)))

    def false_res(v, k):   # decoy 后置信度上升且指向错误答案
        pm_d = max(v['decoy']['belief'].values()); pm_0 = max(v['none_k8']['belief'].values())
        return pm_d > pm_0 + 0.1 and v['decoy']['pred'] != ans[k]
    out['decoy_no_false_resolution'] = not any(false_res(v, k) for k, v in (('A', A), ('B', B)))
    out['irr_no_change'] = all(tv(v['irr']['belief'], v['none_k8']['belief']) < 0.25 for v in (A, B))
    out['prior_bias_text'] = round(abs(A['text']['p_gt'] - B['text']['p_gt']), 2)
    out['tv_AB_coarse'] = round(tv(A['none_k8']['belief'], B['none_k8']['belief']), 3)
    # PASS 只看样本本身的有效性 (C1-C4). decoy / irr 下模型怎么反应是对模型的测量 (False Resolution 等), 不是样本缺陷.
    out['PASS'] = all(out[k] for k in ('C1_coarse_indistinct', 'C2_evidence_sufficient',
                                       'C3_no_text_shortcut', 'C4_evidence_swappable'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='v1'); ap.add_argument('--only', nargs='*')
    ap.add_argument('--no-agentic', action='store_true')
    ap.add_argument('--lang', default='en', choices=['en', 'zh'])
    args = ap.parse_args()
    set_lang(args.lang)
    enable_stop_at_tool_call(True)
    samples = json.load(open(os.path.join(HERE, 'samples.json')))
    if args.only:
        samples = [s for s in samples if s['id'] in args.only]
    vi.load_model()
    letter_ids = _letter_ids(); print('letter token ids:', letter_ids)
    os.makedirs(os.path.join(HERE, 'eval'), exist_ok=True)
    all_res = {}
    for s in samples:
        t0 = time.time()
        res = {'answers': {k: _ans(s, k) for k in ('A', 'B')}, 'version': s.get('version'), 'lang': LANG}
        for variant in ('A', 'B'):
            r = {}
            r['text'] = run_condition(s, variant, 'text', letter_ids)
            for k in (8, 16, 32):
                r[f'none_k{k}'] = run_condition(s, variant, 'none', letter_ids, k=k)
            for cond in ('true', 'twin', 'decoy', 'irr'):
                r[cond] = run_condition(s, variant, cond, letter_ids)
            if not args.no_agentic:
                r['agentic'] = run_agentic(s, variant)
            res[variant] = r
        res['cert'] = certify(res)
        all_res[s['id']] = res
        # ---- 打印 ----
        print(f'\n===== {s["id"]} ({s.get("version")})  answers A={res["answers"]["A"]} B={res["answers"]["B"]}'
              f'  [{time.time()-t0:.0f}s] =====')
        conds = ['text', 'none_k8', 'none_k16', 'none_k32', 'true', 'twin', 'decoy', 'irr']
        print(f'{"cond":10s} ' + ' | '.join(f'{v}: pred  p_gt  p_twin' for v in 'AB'))
        for c in conds:
            row = []
            for v in 'AB':
                x = res[v][c]
                row.append(f'{str(x["pred"]):18s} {x["p_gt"]:.2f}  {x["p_twin"]:.2f}')
            print(f'{c:10s} ' + ' | '.join(row))
        if not args.no_agentic:
            for v in 'AB':
                a = res[v]['agentic']
                print(f'agentic {v}: pred={a["pred"]} correct={a["correct"]} calls={[(c["name"], c["ok"]) for c in a["tool_calls"]]}')
        print('cert:', {k: v for k, v in res['cert'].items()})
    out = os.path.join(HERE, 'eval', f'results_{args.tag}{"_zh" if LANG == "zh" else ""}.json')
    json.dump(all_res, open(out, 'w'), indent=1, ensure_ascii=False, default=str)
    print('\n->', out)
    print('SUMMARY:', {k: v['cert']['PASS'] for k, v in all_res.items()})


if __name__ == '__main__':
    main()
