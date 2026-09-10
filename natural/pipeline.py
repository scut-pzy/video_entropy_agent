"""三个阶段的核心逻辑: 必要性筛选 / E* 定位 / 四条件审计.

阶段一 screen_item      k ∈ {2,4,8,16,32} 帧 + 高分辨率条件下的信念曲线
阶段二 localize_item    滑窗搜索时间 E*, 同宽度最差窗作 decoy
阶段三 audit_item       none / true / decoy / irr / twin / dense_matched 六条件
       agentic_item     DeepEyes 自己用 zoom+seek (Acquisition 层)

所有函数都吃一个 belief.Backend, 返回可 json 序列化的 dict.
"""
import json, re
import numpy as np

import frames as F
import belief as B

SCREEN_KS = (2, 4, 8, 16, 32)
BUDGET_K = 8            # "粗观测" 的定义: 8 帧 @ COARSE_PIXELS
N_EVID = 6              # 工具返回/证据帧数
WINDOW_FRACS = (0.10, 0.20, 0.30)
WINDOW_STRIDE = 0.10

# 认证阈值 (强模型上判定)
P_LOW = 0.40            # 粗观测下 p_gt 低于此 = 分不出
P_HIGH = 0.70           # 拿到证据后 p_gt 高于此 = 可解


def coarse_prefix(item, k=BUDGET_K, pixel_cap=F.COARSE_PIXELS, lang='zh'):
    idx = F.uniform_indices(item['n_frames'], k)
    pils = F.load(item['uuid'], idx, pixel_cap, item['video'])
    return B.frames_prefix(pils, idx, item['fps'], item['duration'], lang=lang), idx


# ───────────────────────── 阶段一: 必要性筛选 ─────────────────────────

def screen_item(bk, item, ks=SCREEN_KS, hi_res=True, n_perm=4, lang='zh'):
    """返回 {cond: belief_dict}; cond = 'k8' / 'k32' / 'k8_hi' / 'text'."""
    out = {}
    gt = item['answer_idx']
    for k in ks:
        prefix, idx = coarse_prefix(item, k, F.COARSE_PIXELS, lang)
        b = bk.belief(prefix, item['stem'], item['options'], n_perm=n_perm, lang=lang)
        b['n_frames_used'] = len(idx)
        out[f'k{k}'] = b
    if hi_res and item['width'] and min(item['width'], item['height']) >= 480:
        prefix, idx = coarse_prefix(item, BUDGET_K, F.DETAIL_PIXELS, lang)
        b = bk.belief(prefix, item['stem'], item['options'], n_perm=n_perm, lang=lang)
        b['n_frames_used'] = len(idx)
        out[f'k{BUDGET_K}_hi'] = b
    out['text'] = bk.belief([], item['stem'], item['options'], n_perm=n_perm, lang=lang)
    for c, b in out.items():
        b['p_gt'] = b['p'][gt]
        b['correct'] = (b['pred_idx'] == gt)
    return out


def classify(screen, k_low=BUDGET_K):
    """只用均匀预算的信念曲线做**初筛**, 返回 (label, 说明).

    label ∈ {text_shortcut, already_solvable, candidate}
    注意: 这里**不**判 unsolvable —— 3 条冒烟里的 Cricket 那条 (均匀 k32 只有 0.07, 但
    [1.8,3.6]s 密采到 0.70、EEG=1.69) 证明"均匀加帧解不了"完全不等于"信息不在视频里".
    必要性只能由阶段二的窗搜索来认证 (C2 的定义本来就是"给了 E* 才可解", E* 是窗不是均匀预算).
    """
    lo = screen[f'k{k_low}']
    txt = screen.get('text')
    if txt and txt['p_gt'] >= P_HIGH:
        return 'text_shortcut', f"纯文本 p_gt={txt['p_gt']:.2f}，不看视频就能答"
    if lo['p_gt'] >= P_HIGH and lo['correct']:
        return 'already_solvable', f"粗观测 p_gt={lo['p_gt']:.2f}（N=0 对照）"
    return 'candidate', f"粗观测 p_gt={lo['p_gt']:.2f} < {P_HIGH}，进窗搜索"


P_ESTAR = 0.60       # 给了 E* 之后至少要到这个信念才算"可判定"
GAP_MIN = 0.25       # 且相对粗观测至少提升这么多, 才算"证据带来了实质的可判定性"


def certify(screen, loc, k_low=BUDGET_K, k_high=32):
    """screen + localize 合起来给最终标签, 并说明"必要性来自哪里".

    不用单一绝对阈值: 冒烟里的 Cricket 那条 E* p_gt=0.696 差 0.004 就被判成 unsolvable, 但它
    none=0.13→true=0.66、EEG=1.69, 显然是 evidence-required. 所以判据 = 答对 + 绝对值够 + 相对粗观测的**提升**够.
    """
    lo = screen[f'k{k_low}']['p_gt']
    hi = screen.get(f'k{k_high}', {}).get('p_gt')
    hires = screen.get(f'k{k_low}_hi', {}).get('p_gt')
    e = loc['estar']
    gap = e['p_gt'] - lo
    if e['p_gt'] < P_ESTAR or not e.get('correct', True):
        return 'unsolvable', f"最好的窗只到 p_gt={e['p_gt']:.2f}（<{P_ESTAR}），给了证据也解不了"
    if gap < GAP_MIN:
        return 'borderline', f"粗观测 {lo:.2f} → E* 窗 {e['p_gt']:.2f}，提升 {gap:+.2f} 不足 {GAP_MIN}"
    src = []
    if hi is not None and hi >= P_HIGH:
        src.append(f'均匀 k{k_high} 也行({hi:.2f})')
    if hires is not None and hires >= P_HIGH:
        src.append(f'高分辨率也行({hires:.2f})')
    if not src:
        src.append('只有定位到窗才行')
    return 'evidence_required', (f"粗观测 {lo:.2f} → E* 窗 {e['p_gt']:.2f}（{gap:+.2f}）；" + '、'.join(src))


# ───────────────────────── 阶段二: E* 定位 ─────────────────────────

def _windows(duration, fracs=WINDOW_FRACS, stride=WINDOW_STRIDE):
    out = []
    for w in fracs:
        step = max(stride, w / 2) * duration
        width = w * duration
        t = 0.0
        while t + width <= duration + 1e-6:
            out.append((round(t, 3), round(t + width, 3), w))
            t += step
    return out


def localize_item(bk, item, n_perm=2, lang='zh', n_evid=N_EVID, verbose=False):
    """滑窗搜索时间 E*: coarse-8 前缀 + 窗内密采 n_evid 帧, 取 p_gt 最高的窗;
    E* = 满足 p_gt>=P_HIGH 的最小窗宽里最好的那个 (没有就取全局最好).
    decoy = 与 E* 同宽度、p_gt 最低的窗."""
    prefix, _ = coarse_prefix(item, BUDGET_K, F.COARSE_PIXELS, lang)
    gt = item['answer_idx']
    rows = []
    for (t0, t1, w) in _windows(item['duration']):
        idx = F.window_indices(item['n_frames'], t0, t1, n_evid, item['fps'])
        pils = F.load(item['uuid'], idx, F.DETAIL_PIXELS, item['video'])
        ev = B.evidence_prefix(pils, idx, item['fps'], lang=lang, start_at=BUDGET_K)
        b = bk.belief(prefix + ev, item['stem'], item['options'], n_perm=n_perm, lang=lang)
        rows.append(dict(t0=t0, t1=t1, frac=w, p_gt=b['p'][gt], pred_idx=b['pred_idx'],
                         correct=(b['pred_idx'] == gt), entropy=b['entropy'],
                         n_frames=len(idx), idx=idx))
        if verbose:
            print(f"    窗 [{t0:5.2f},{t1:5.2f}] ({w:.0%})  p_gt={b['p'][gt]:.3f}  pred={b['pred_idx']}")
    ok = [r for r in rows if r['p_gt'] >= P_ESTAR and r['pred_idx'] == gt]
    if ok:
        best_w = min(r['frac'] for r in ok)
        star = max([r for r in ok if r['frac'] == best_w], key=lambda r: r['p_gt'])
    else:
        star = max(rows, key=lambda r: r['p_gt'])
    same = [r for r in rows if r['frac'] == star['frac']]
    decoy = min(same, key=lambda r: r['p_gt'])
    if decoy['t0'] == star['t0']:                      # 只有一个窗时退化, 取时间上最远的
        decoy = max(same, key=lambda r: abs(r['t0'] - star['t0'])) if len(same) > 1 else None
    return dict(windows=rows, estar=star, decoy=decoy,
                certified=bool(ok), n_windows=len(rows))


# ───────────────────────── 阶段三: 四条件审计 ─────────────────────────

def _evid_frames(item, win, n_evid=N_EVID, bbox=None, gray=False, src=None):
    src = src or item
    idx = F.window_indices(src['n_frames'], win[0], win[1], n_evid, src['fps'])
    if bbox is not None:
        pils = F.crop(src['uuid'], idx, bbox, F.DETAIL_PIXELS, src['video'])
    else:
        pils = F.load(src['uuid'], idx, F.DETAIL_PIXELS, src['video'])
    if gray:
        from PIL import Image
        pils = [Image.new('RGB', p.size, (128, 128, 128)) for p in pils]
    return pils, idx


def audit_item(bk, item, estar, decoy=None, twin_item=None, twin_win=None,
               n_perm=4, lang='zh', n_evid=N_EVID, bbox=None):
    """六条件. estar/decoy 是 (t0,t1); twin_item 是同域兄弟视频的 item dict."""
    gt = item['answer_idx']
    prefix, coarse_idx = coarse_prefix(item, BUDGET_K, F.COARSE_PIXELS, lang)
    out = {}

    def run(name, ev_content):
        b = bk.belief(prefix + ev_content, item['stem'], item['options'], n_perm=n_perm, lang=lang)
        b['p_gt'] = b['p'][gt]; b['correct'] = (b['pred_idx'] == gt)
        out[name] = b
        return b

    b0 = bk.belief(prefix, item['stem'], item['options'], n_perm=n_perm, lang=lang)
    b0['p_gt'] = b0['p'][gt]; b0['correct'] = (b0['pred_idx'] == gt)
    out['none'] = b0

    pils, idx = _evid_frames(item, estar, n_evid, bbox)
    run('true', B.evidence_prefix(pils, idx, item['fps'], lang=lang, bbox=bbox, start_at=BUDGET_K))

    if decoy is not None:
        pils, idx = _evid_frames(item, decoy, n_evid, bbox)
        run('decoy', B.evidence_prefix(pils, idx, item['fps'], lang=lang, bbox=bbox, start_at=BUDGET_K))

    pils, idx = _evid_frames(item, estar, n_evid, bbox, gray=True)
    run('irr', B.evidence_prefix(pils, idx, item['fps'], lang=lang, bbox=bbox, start_at=BUDGET_K))

    if twin_item is not None and twin_win is not None:
        pils, idx = _evid_frames(twin_item, twin_win, n_evid, bbox=None, src=twin_item)
        b = run('twin', B.evidence_prefix(pils, idx, twin_item['fps'], lang=lang, start_at=BUDGET_K))
        # 兄弟视频的类别在本题选项里的位置
        tn = (twin_item['name'] or '').strip().lower()
        ti = next((i for i, o in enumerate(item['options']) if o.strip().lower() == tn), None)
        b['twin_idx'] = ti
        b['p_twin'] = b['p'][ti] if ti is not None else None

    # 预算对齐对照: 把证据帧的 token 预算换成更密的均匀采样
    k_eq = BUDGET_K + n_evid * int(round(F.DETAIL_PIXELS / F.COARSE_PIXELS))
    prefix2, idx2 = coarse_prefix(item, k_eq, F.COARSE_PIXELS, lang)
    b = bk.belief(prefix2, item['stem'], item['options'], n_perm=n_perm, lang=lang)
    b['p_gt'] = b['p'][gt]; b['correct'] = (b['pred_idx'] == gt); b['k'] = k_eq
    b['n_frames_used'] = len(idx2)
    out['dense_matched'] = b

    out['_meta'] = dict(estar=estar, decoy=decoy, bbox=bbox, n_evid=n_evid,
                        coarse_idx=coarse_idx, gt=gt)
    return out


def metrics(audit):
    """从 audit_item 的结果算 EEG / ERG / False Resolution 等."""
    def lp(c):
        return float(np.log(max(audit[c]['p_gt'], 1e-6))) if c in audit else None
    m = dict(
        p_none=audit['none']['p_gt'], p_true=audit['true']['p_gt'],
        p_decoy=audit['decoy']['p_gt'] if 'decoy' in audit else None,
        p_irr=audit['irr']['p_gt'], p_dense=audit['dense_matched']['p_gt'],
    )
    if 'decoy' in audit:
        m['EEG'] = round(lp('true') - lp('decoy'), 4)
        m['ERG'] = int(audit['true']['correct']) - int(audit['decoy']['correct'])
        pm_d = max(audit['decoy']['p']); pm_0 = max(audit['none']['p'])
        m['false_resolution'] = bool(pm_d > pm_0 + 0.1 and not audit['decoy']['correct'])
    m['gain_over_dense'] = round(m['p_true'] - m['p_dense'], 4)
    m['gain_over_none'] = round(m['p_true'] - m['p_none'], 4)
    if 'twin' in audit and audit['twin'].get('p_twin') is not None:
        m['twin_p_twin'] = audit['twin']['p_twin']
        m['twin_flipped'] = bool(audit['twin']['pred_idx'] == audit['twin']['twin_idx'])
    return m


# ───────────────────────── Acquisition: agentic ─────────────────────────

def overlap(a, b):
    """两个时间窗的 IoU."""
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def temporal_hit(win, estar):
    """工具取到的时间窗有没有命中 E*.

    zoom 只指定一帧, 窗宽为 0, 用 IoU 恒为 0 是不公平的 —— 落在 E* 里就算命中.
    返回 (hit: bool, iou: float).
    """
    if win is None or estar is None:
        return False, 0.0
    if win[1] - win[0] < 1e-6:                     # 点窗 (zoom): 看是否落在 E* 内
        return bool(estar[0] <= win[0] <= estar[1]), 0.0
    return overlap(win, estar) > 0, round(overlap(win, estar), 3)


def agentic_item(bk, item, estar=None, max_turns=4, lang='zh', max_new_tokens=512, verbose=False):
    """DeepEyes 自己决定调不调工具. 复用 video_imcot 的会话/工具, 但走本模块的帧缓存无关路径:
    这里直接用 video_imcot.VideoSession (它自己解码), 因为工具需要在原视频上任意裁剪/回看."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
    import video_imcot as vi
    from prompts_zh import SYS_ZH, INSTR_ZH, CONT_ZH, PRE_ANSWER_RE

    sess = vi.VideoSession(item['video'], n_coarse=BUDGET_K)
    coarse = list(range(len(sess.frames)))
    q = item['stem'] + '\n' + '\n'.join(f'{c}. {o}' for c, o in zip('ABCD', item['options']))
    system = SYS_ZH if lang == 'zh' else vi.SYS_FULL
    instr = INSTR_ZH if lang == 'zh' else vi.INSTR_FULL
    cont = CONT_ZH if lang == 'zh' else vi.CONT_PROMPT
    header = (f'以下是从一段 {sess.duration:.1f} 秒的视频中均匀抽取的 {len(coarse)} 帧画面，'
              f'每帧都标注了序号和时间戳。\n') if lang == 'zh' else vi._header(sess)
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': [{'type': 'text', 'text': header}] +
                 _labeled(sess, coarse, lang) +
                 [{'type': 'text', 'text': ('问题：' if lang == 'zh' else 'Question: ') + q + instr}]}]
    traj = dict(turns=[], tool_calls=[], pred=None, entropy=[], tokens=[], turn_bounds=[])
    for turn in range(max_turns):
        resp, ent, toks = _gen(bk, messages, max_new_tokens)
        traj['turns'].append(resp)
        traj['turn_bounds'].append([len(traj['entropy']), len(traj['entropy']) + len(ent)])
        traj['entropy'] += ent; traj['tokens'] += toks
        if verbose:
            print(f'  --- 第 {turn+1} 轮 ---\n  {resp[:400]}')
        messages.append({'role': 'assistant', 'content': resp})
        if '<answer>' in resp:
            traj['pred'] = vi._extract_letter(resp); break
        calls = re.findall(r'<tool_call>(.*?)</tool_call>', resp, re.S)
        if not calls:
            traj['pred'] = vi._extract_letter(resp); break
        obs = []
        for c in calls[:3]:
            rec = dict(turn=turn, raw=c.strip()[:300], ok=False, err=None, name=None,
                       window=None, bbox=None, frame_index=None)
            try:
                call = json.loads(c.strip()); rec['name'] = call.get('name')
                args = call.get('arguments', {})
                if call['name'] == 'image_zoom_in_tool':
                    new = sess.tool_zoom(args)
                    rec['bbox'] = [float(v) for v in args['bbox_2d']]
                    rec['frame_index'] = int(args['frame_index'])
                    fi = rec['frame_index']
                    if 0 <= fi < len(sess.frames):
                        t = sess.frames[fi]['t']; rec['window'] = [t, t]
                elif call['name'] == 'video_seek_tool':
                    new = sess.tool_seek(args)
                    rec['window'] = [float(args['start_time']), float(args['end_time'])]
                else:
                    raise ValueError(f"未知工具 {call.get('name')}")
                rec['ok'] = True
                obs += _labeled(sess, new, lang)
            except Exception as e:
                rec['err'] = str(e)[:200]
                obs.append({'type': 'text', 'text': f'Error: {e}'})
            traj['tool_calls'].append(rec)
        obs.append({'type': 'text', 'text': cont})
        messages.append({'role': 'user', 'content': obs})
    if traj['pred'] is None and traj['turns']:
        traj['pred'] = vi._extract_letter(traj['turns'][-1])
    traj['correct'] = (traj['pred'] == item['answer_letter'])
    traj['pre_answer'] = bool(traj['turns'] and
                              re.search(PRE_ANSWER_RE, traj['turns'][0].split('</think>')[0]))
    traj.update(entropy_profile(traj))
    traj['tools_used'] = sorted({c['name'] for c in traj['tool_calls'] if c['ok']})
    traj['n_calls_ok'] = sum(1 for c in traj['tool_calls'] if c['ok'])
    if estar:
        hits = [temporal_hit(rec['window'], estar) for rec in traj['tool_calls']
                if rec['ok'] and rec['window']]
        traj['temporal_hit'] = bool(any(h for h, _ in hits))
        traj['temporal_iou_max'] = max([i for _, i in hits], default=0.0)
    return traj


def _labeled(sess, idx_list, lang):
    out = []
    for i in idx_list:
        f = sess.frames[i]
        if lang == 'zh':
            note = f['note'].replace('dense replay', '密集回放')
            note = re.sub(r'zoom of Frame (\d+)', r'放大自第\1帧', note)
            lab = f"[第{i}帧, t={f['t']:.1f}s{('，' + note) if note else ''}]"
        else:
            lab = sess.frame_label(i)
        out.append({'type': 'text', 'text': lab})
        out.append({'type': 'image', 'image': f['pil_disp']})
    return out


def _gen(bk, messages, max_new_tokens, with_entropy=True):
    """多轮对话的生成 (Backend.generate 只吃单轮 content), 在 </tool_call> 处截停.

    返回 (text, ent, toks): ent = 每个生成 token 的 next-token 熵 (PEA 式 H_t = -Σ p log p, 全词表),
    toks = 对应的 token 字符串. 用来画"熵 vs 输出位置百分比"和高熵 token 位置直方图."""
    import torch
    text = bk.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [c['image'] for m in messages if isinstance(m['content'], list)
              for c in m['content'] if c.get('type') == 'image']
    inputs = bk.processor(text=[text], images=images or None, padding=True,
                          return_tensors='pt').to(bk.model.device)
    with torch.inference_mode():
        out = bk.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                stop_strings=['</tool_call>'], tokenizer=bk.processor.tokenizer,
                                output_scores=with_entropy, return_dict_in_generate=True)
    seq = out.sequences[:, inputs.input_ids.shape[1]:]
    gen_text = bk.processor.batch_decode(seq, skip_special_tokens=True)[0]
    ent, toks = [], []
    if with_entropy and out.scores:
        ids = seq[0].tolist()
        for i, s in enumerate(out.scores):
            lp = torch.log_softmax(s[0].float(), -1)
            ent.append(round(float(-(lp.exp() * lp).sum()), 4))
        tk = bk.processor.tokenizer
        toks = [tk.decode([i], skip_special_tokens=False) for i in ids[:len(ent)]]   # 逐 id 解码, 中文不乱码
    return gen_text, ent, toks


# ───────────────────────── PEA 式 entropy 画像 ─────────────────────────

def _find_token_pos(tokens, pattern):
    """在 token 串里找子串 pattern 第一次出现时的 token 下标 (按累计字符定位)."""
    text = ''
    starts = []
    for t in tokens:
        starts.append(len(text)); text += t
    text_clean = text
    pos = []
    k = 0
    while True:
        j = text_clean.find(pattern, k)
        if j < 0:
            break
        # 该字符落在哪个 token
        idx = max(i for i, s in enumerate(starts) if s <= j) if starts else 0
        pos.append(idx); k = j + len(pattern)
    return pos


def entropy_profile(traj, top_frac=0.20, n_bins=10):
    """整条轨迹 (多轮拼接) 的 entropy 画像:
       signature      = PEA 的 top-20% 高熵 token 相对位置 10-bin 直方图 (归一化)
       markers_pct    = <tool_call> 开始 / </think> / <answer> 出现的位置 (占输出总长的百分比)
       ent_pre/post   = 首个 tool_call 前后的平均熵"""
    ent = traj.get('entropy') or []
    toks = traj.get('tokens') or []
    n = len(ent)
    if n == 0:
        return dict(entropy_profile=None)
    e = np.array(ent)
    k = max(1, int(round(top_frac * n)))
    top_idx = np.argsort(-e)[:k]
    rel = top_idx / max(1, n - 1)
    hist, _ = np.histogram(rel, bins=n_bins, range=(0, 1))
    sig = (hist / hist.sum()).round(4).tolist()
    def pct(idxs):
        return [round(100.0 * i / max(1, n - 1), 1) for i in idxs]
    tc = _find_token_pos(toks, '<tool_call>')
    te = _find_token_pos(toks, '</think>')
    an = _find_token_pos(toks, '<answer>')
    first_tc = tc[0] if tc else None
    prof = dict(n_tokens=n, mean_entropy=round(float(e.mean()), 4),
                signature=sig,
                markers_pct=dict(tool_call=pct(tc), think_end=pct(te), answer=pct(an),
                                 turn_starts=pct([b[0] for b in traj.get('turn_bounds', [])])),
                ent_pre_tool=round(float(e[:first_tc].mean()), 4) if first_tc else None,
                ent_post_tool=round(float(e[first_tc:].mean()), 4) if first_tc else None,
                top_tokens=[(toks[i] if i < len(toks) else '', round(float(e[i]), 2), round(100 * i / max(1, n - 1), 1))
                            for i in sorted(top_idx[:12].tolist())])
    return dict(entropy_profile=prof)
