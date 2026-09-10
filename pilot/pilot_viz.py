"""EvidenceGap-V pilot 的 notebook 展示层: 看数据 (孪生对 / 模型实际看到的帧 / E* 证据帧)、看六个证据条件下的信念、
看 DeepEyes 自己调工具的完整轨迹. 计算全部复用 eval_deepeyes.py, 这里只负责画."""
import os, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import html as _html
from IPython.display import display, Markdown, HTML

import eval_deepeyes as ev
import video_imcot as vi

HERE = ev.HERE
CONDS = ['text', 'none', 'true', 'twin', 'decoy', 'irr']
COND_DESC = {
    'text':  '只有题目, 没有任何帧 (C3: 文本捷径)',
    'none':  '8 帧均匀粗采样 (C1: 粗观测能不能分辨)',
    'true':  '粗采样 + 本视频 E* 证据帧 (C2: 证据够不够)',
    'twin':  '粗采样 + 孪生视频的 E* 证据帧 (C4: 答案是否跟着证据走 → 应翻到孪生答案)',
    'decoy': '粗采样 + 同框同尺寸但不含判别信息的帧 (不应产生错误确信)',
    'irr':   '粗采样 + 同尺寸灰图 (不应变化)',
}
_CJK_TTF = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'      # 文泉驿正黑, PIL 和 matplotlib 都用它, 否则中文是方框
_FONT = ImageFont.truetype(_CJK_TTF, 13)
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_samples(only=None):
    S = json.load(open(os.path.join(HERE, 'samples.json')))
    return [s for s in S if not only or s['id'] in only]


def _labelled(pil, label, w=230):
    t = pil.copy(); t.thumbnail((w, w))
    canvas = Image.new('RGB', (w, t.height + 18), 'white')
    canvas.paste(t, ((w - t.width) // 2, 18))
    ImageDraw.Draw(canvas).text((2, 2), label, fill=(0, 0, 0), font=_FONT)
    return canvas


def _strip(pils, labels, w=230):
    tiles = [_labelled(p, l, w) for p, l in zip(pils, labels)]
    H = max(t.height for t in tiles)
    out = Image.new('RGB', (w * len(tiles), H), 'white')
    for i, t in enumerate(tiles):
        out.paste(t, (i * w, 0))
    return out


def show_sample(sample):
    """题面 + 孪生对: 每个 twin 一行 8 张粗采帧 (模型实际看到的分辨率) + 一行 E* 证据帧 (原生分辨率裁剪)."""
    s = sample; opts = ev._opts(s)
    display(Markdown(f"## `{s['id']}`  (version {s.get('version')})\n\n"
                     f"**题目**: {ev._question(s)}\n\n"
                     f"**选项**: " + ' / '.join(f'{L}. {o}' for L, o in zip('ABCD', opts)) + "\n\n"
                     f"**孪生答案**: A → `{ev._ans(s, 'A')}`, B → `{ev._ans(s, 'B')}`  "
                     f"| **E\\***: 帧 {s['evidence_window_frames'] or '全程'} × 框 {s['evidence_bbox']}  "
                     f"| 粗采样帧序号 {s['coarse_frame_idx']}"))
    for v in 'AB':
        path = os.path.join(HERE, 'videos', s['twins'][v]['video'])
        sess = vi.VideoSession(path, n_coarse=8)
        coarse = list(range(len(sess.frames)))
        evid = ev._evidence_frames(sess, path, s['evidence_window_frames'], s['evidence_bbox'])
        display(Markdown(f"**孪生 {v}** — 答案 `{ev._ans(s, v)}`. 上: 模型看到的 8 帧粗采样 "
                         f"({sess.frames[0]['pil_disp'].size[0]}×{sess.frames[0]['pil_disp'].size[1]} px); "
                         f"下: E* 证据帧 (裁剪后 {sess.frames[evid[0]]['pil_disp'].size[0]}×{sess.frames[evid[0]]['pil_disp'].size[1]} px)"))
        display(_strip([sess.frames[i]['pil_disp'] for i in coarse], [sess.frame_label(i) for i in coarse], w=200))
        display(_strip([sess.frames[i]['pil_disp'] for i in evid], [sess.frame_label(i) for i in evid], w=200))


def run_conditions(sample, letter_ids, conds=CONDS, show_frames=True):
    """六个条件 × A/B, 返回 results dict (与 eval_deepeyes 同构), 并展示每个条件追加给模型的证据帧和信念表."""
    s = sample; opts = ev._opts(s); res = {'answers': {k: ev._ans(s, k) for k in 'AB'}, 'lang': ev.LANG}
    _show_question(s)
    rows = []
    for v in 'AB':
        r = {}
        for c in conds:
            viz = {}
            key = 'none_k8' if c == 'none' else c
            r[key] = ev.run_condition(s, v, c, letter_ids, viz=viz)
            if show_frames and viz.get('evidence'):
                display(Markdown(f"*孪生 {v} · `{c}`* — {viz['caption']}"))
                display(_strip(viz['evidence'], viz['evidence_labels'], w=160))
            b = r[key]['belief']
            rows.append(dict(twin=v, cond=c, **{o: b[o] for o in opts}, pred=r[key]['pred'],
                             p_gt=r[key]['p_gt'], p_twin=r[key]['p_twin']))
        res[v] = r
    df = pd.DataFrame(rows).set_index(['twin', 'cond'])

    def _style(row):
        gt = res['answers'][row.name[0]]
        return ['background-color:#d9f2d9' if col == gt else ('background-color:#f6dada' if col == 'pred' and row['pred'] != gt else '')
                for col in df.columns]
    display(Markdown('**信念表** (绿色列 = 该 twin 的正确答案; `pred` 标红 = 答错). 每格是 4 个选项置换平均后的 p(option):'))
    display(df.style.apply(_style, axis=1).format({o: '{:.2f}' for o in opts} | {'p_gt': '{:.2f}', 'p_twin': '{:.2f}'}))
    res['cert'] = _certify_safe(res)
    return res


def _certify_safe(res):
    try:
        return ev.certify(res)
    except KeyError:
        return {}


def plot_beliefs(res, sample, conds=CONDS):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    x = np.arange(len(conds))
    for ax, v in zip(axes, 'AB'):
        keys = ['none_k8' if c == 'none' else c for c in conds]
        ax.bar(x - 0.2, [res[v][k]['p_gt'] for k in keys], 0.4, label='p(本视频的正确答案)', color='#2a9d8f')
        ax.bar(x + 0.2, [res[v][k]['p_twin'] for k in keys], 0.4, label='p(孪生视频的答案)', color='#e76f51')
        ax.axhline(0.25, ls=':', c='gray', lw=1); ax.set_xticks(x); ax.set_xticklabels(conds)
        ax.set_title(f"{sample['id']} — 孪生 {v}（答案：{res['answers'][v]}）", fontsize=10); ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('信念 p'); axes[0].legend(fontsize=8, loc='upper left')
    plt.tight_layout(); plt.show()


def _parse_turn(text):
    think = re.search(r'<think>(.*?)(?:</think>|$)', text, re.S)
    ans = re.search(r'<answer>(.*?)</answer>', text, re.S)
    calls = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
    return (think.group(1).strip() if think else None), calls, (ans.group(1).strip() if ans else None)


# ── 卡片式显示: 题目 / 思考 / 工具 / 答案 各一种颜色, 纯 HTML, 不依赖 Markdown 渲染 ──
_CARD = {'q': ('#555', '#f3f3f3', '❓ 问题'), 'think': ('#2b6cb0', '#eaf2fb', '🧠 思考'),
         'tool': ('#c05621', '#fdf1e7', '🔧 工具调用'), 'ok': ('#2f855a', '#e9f7ee', '✅ 答案'),
         'bad': ('#c53030', '#fdeaea', '❌ 答案'), 'warn': ('#975a16', '#fdf6e3', '⚠️ 说明')}


def _card(kind, body, title=None, raw_html=False):
    color, bg, default_title = _CARD[kind]
    body = body if raw_html else _html.escape(body or '（空）')
    display(HTML(f'<div style="border-left:5px solid {color};background:{bg};padding:8px 14px;margin:6px 0;'
                 f'border-radius:6px;font-size:14px;line-height:1.7">'
                 f'<div style="font-weight:600;color:{color};margin-bottom:2px">{_html.escape(title or default_title)}</div>'
                 f'<div style="white-space:pre-wrap;color:#222">{body}</div></div>'))


def _section(text):
    display(HTML(f'<h3 style="margin:18px 0 6px 0;padding-bottom:4px;border-bottom:1px solid #ddd">{_html.escape(text)}</h3>'))


def _quote(title, body):
    _card('think', body, title=title)


def _show_question(sample):
    opts = ev._opts(sample)
    body = ev._question(sample) + '\n' + '　'.join(f'{L}. {o}' for L, o in zip('ABCD', opts))
    _card('q', body)


def _answer_line(pred, sample, variant):
    opts = ev._opts(sample); gt = sample['twins'][variant]['answer_letter']
    txt = f"{pred}（{opts['ABCD'.index(pred)]}）" if pred in 'ABCD' and pred else str(pred)
    ok = pred == gt
    return f"模型答案：{txt}\n{'正确' if ok else '错误'}（正确答案 {gt}，{ev._ans(sample, variant)}）", ok


def run_direct_cot(sample, variant, max_new_tokens=512):
    """无工具: 只看 8 帧粗采样, 要求先在 <think> 里用中文推理再作答. 看它在证据缺失时是怎么"想"出答案的."""
    s = sample
    _section(f"无工具直接推理 · {s['id']} · 孪生 {variant}")
    _show_question(s)
    path = os.path.join(HERE, 'videos', s['twins'][variant]['video'])
    sess = vi.VideoSession(path, n_coarse=8); coarse = list(range(len(sess.frames)))
    q = ev._q_text(ev._question(s), ev._opts(s), suffix=False)
    instr = ('\n请先在 <think></think> 里用中文写出你的推理过程，然后在 <answer></answer> 里只填选项字母。' if ev.LANG == 'zh'
             else '\nThink first inside <think></think>, then put only the option letter inside <answer></answer>.')
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': vi._header(sess)}] + vi._frames_content(sess, coarse)
                 + [{'type': 'text', 'text': ('问题：' if ev.LANG == 'zh' else 'Question: ') + q + instr}]}]
    resp = ev._GENERATE_ORIG(messages, max_new_tokens=max_new_tokens)
    think, _, ans = _parse_turn(resp)
    _quote('🧠 思考（think）', think if think is not None else resp)
    pred = vi._extract_letter(resp)
    line, ok = _answer_line(pred, s, variant)
    _card('ok' if ok else 'bad', line)
    return dict(pred=pred, correct=ok, raw=resp)


def run_agentic(sample, variant, max_turns=4, max_new_tokens=512):
    """DeepEyes 自己决定调不调工具 (zoom + seek). 每一轮完整显示: think 内容 → 工具调用 → 工具返回的帧 → (最后一轮) 答案."""
    s = sample; me = s['twins'][variant]
    _section(f"带工具的主动推理 · {s['id']} · 孪生 {variant}")
    _show_question(s)
    path = os.path.join(HERE, 'videos', me['video'])
    sess = vi.VideoSession(path, n_coarse=8); coarse = list(range(len(sess.frames)))
    zh = ev.LANG == 'zh'
    p = ev.agentic_prompts()
    system = p.get('system_prompt', vi.SYS_FULL); instr = p.get('instruction', vi.INSTR_FULL); cont = p.get('cont_prompt', vi.CONT_PROMPT)
    q = ev._q_text(ev._question(s), ev._opts(s), suffix=not zh)
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': [{'type': 'text', 'text': vi._header(sess)}] + vi._frames_content(sess, coarse)
                 + [{'type': 'text', 'text': ('问题：' if zh else 'Question: ') + q + instr}]}]
    traj = dict(pred=None, turns=[], tool_calls=[])
    for turn in range(max_turns):
        resp = vi._generate(messages, max_new_tokens=max_new_tokens)     # 已在 </tool_call> 处截停
        traj['turns'].append(resp)
        think, calls, ans = _parse_turn(resp)
        _quote(f'🧠 第 {turn + 1} 轮 · 思考（think）', think if think is not None else resp)
        messages.append({'role': 'assistant', 'content': resp})
        if ans is not None:
            traj['pred'] = vi._extract_letter(resp); break
        if not calls:
            _card('warn', '这一轮既没有工具调用也没有 <answer>，按最后出现的字母取答案')
            traj['pred'] = vi._extract_letter(resp); break
        obs = []
        for c in calls[:3]:
            rec = dict(turn=turn, raw=c.strip()[:400], ok=False, err=None, name=None)
            try:
                call = json.loads(c.strip()); rec['name'] = call.get('name')
                args = call.get('arguments', {})
                if call['name'] == 'image_zoom_in_tool':
                    new_idx = sess.tool_zoom(args)
                elif call['name'] == 'video_seek_tool':
                    new_idx = sess.tool_seek(args)
                else:
                    raise ValueError(f"未知工具 {call.get('name')}")
                rec['ok'] = True
                _card('tool', f"{call['name']}\n{json.dumps(args, ensure_ascii=False, indent=2)}\n→ 返回 {len(new_idx)} 帧：")
                display(_strip([sess.frames[i]['pil_disp'] for i in new_idx], [sess.frame_label(i) for i in new_idx], w=200))
                obs += vi._frames_content(sess, new_idx)
            except Exception as e:
                rec['err'] = str(e)[:200]
                _card('tool', f"{rec['raw'][:300]}\n→ 执行失败：{rec['err']}", title='🔧 工具调用失败')
                obs.append({'type': 'text', 'text': f'Error: {e}'})
            traj['tool_calls'].append(rec)
        obs.append({'type': 'text', 'text': cont})
        messages.append({'role': 'user', 'content': obs})
    if traj['pred'] is None and traj['turns']:
        traj['pred'] = vi._extract_letter(traj['turns'][-1])
    line, ok = _answer_line(traj['pred'], s, variant)
    traj['correct'] = ok
    pre = bool(traj['turns']) and bool(re.search(ev.PRE_ANSWER_RE, traj['turns'][0].split('</think>')[0]))
    n_ok = sum(c['ok'] for c in traj['tool_calls'])
    _card('ok' if ok else 'bad', line + f"\n轮数 {len(traj['turns'])}，工具调用 {len(traj['tool_calls'])} 次（成功 {n_ok}）"
          f"，首轮 think 里先下结论再调工具：{'是' if pre else '否'}")
    return traj


def cert_table(all_res):
    rows = []
    for sid, r in all_res.items():
        c = r.get('cert', {})
        rows.append(dict(sample=sid, **{k: c.get(k) for k in ['C1_coarse_indistinct', 'C2_evidence_sufficient',
                                                                'C3_no_text_shortcut', 'C4_evidence_swappable',
                                                                'decoy_no_false_resolution', 'prior_bias_text',
                                                                'tv_AB_coarse', 'PASS']}))
    df = pd.DataFrame(rows).set_index('sample')
    paint = getattr(df.style, 'map', None) or df.style.applymap
    display(paint(lambda x: 'background-color:#d9f2d9' if x is True else ('background-color:#f6dada' if x is False else '')))
    return df
