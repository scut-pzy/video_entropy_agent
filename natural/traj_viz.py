"""把审计/agentic 结果整理成 (1) 一个精简、按行拆开、便于人看的 JSON; (2) PEA 式 entropy 图.

JSON 里每条题只放这些: 题目/选项/答案, 模型看到的帧时间戳, E*/decoy 窗, 六条件的 p(正确),
每一轮的 think(按句拆行) / 工具调用 / 返回帧时间戳 / 答案, 以及 entropy 画像 (signature, 工具调用起点百分比).

图: 每条题一张 "熵 vs 输出位置%" (标出 <tool_call> 起点、</think>、<answer>), 右侧 top-20% 高熵 token
位置直方图; 再一张全体汇总 (平均 signature, 按答对/答错分, 工具调用起点分布).
"""
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'out', 'figs')
CJK = ['WenQuanYi Zen Hei', 'DejaVu Sans']


def _lines(text, width=60):
    """长文本按句/换行拆成列表, JSON 里一行一条, 好看."""
    if not text:
        return []
    text = text.replace('<think>', '').strip()
    parts = re.split(r'(?<=[。！？；\n])', text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > width:                  # 超长句再按宽度硬切
            out.append(p[:width]); p = p[width:]
        out.append(p)
    return out


def _turn_records(traj, fps=None):
    """把 agentic 的 turns/tool_calls 合并成逐轮记录."""
    calls_by_turn = {}
    for c in traj.get('tool_calls', []):
        calls_by_turn.setdefault(c['turn'], []).append(c)
    rounds = []
    for i, t in enumerate(traj.get('turns', [])):
        think = t.split('</think>')[0]
        ans = re.search(r'<answer>(.*?)</answer>', t, re.S)
        r = dict(轮=i + 1, 思考=_lines(think))
        cs = []
        for c in calls_by_turn.get(i, []):
            d = dict(工具=c['name'])
            if c['window'] is not None:
                d['时间窗s'] = [round(c['window'][0], 2), round(c['window'][1], 2)]
            if c.get('bbox') is not None:
                d['bbox'] = c['bbox']; d['帧序号'] = c.get('frame_index')
            if not c['ok']:
                d['错误'] = c['err']
            cs.append(d)
        if cs:
            r['工具调用'] = cs
        if ans:
            r['答案'] = ans.group(1).strip()
        rounds.append(r)
    return rounds


def concise_record(rec, item=None):
    """rec = run_audit 的一条记录 (含 audit/metrics/agentic). 返回精简 dict."""
    au = rec.get('audit', {}); m = rec.get('metrics', {}); ag = rec.get('agentic')
    meta = au.get('_meta', {})
    out = {
        'uuid': rec['uuid'], '域': rec.get('domain'), '动作': rec.get('name'),
        '组': rec.get('group'), '认证者': rec.get('certifier'),
    }
    if item:
        out['题目'] = item.get('question_zh') or item.get('stem')
        out['选项'] = {L: o for L, o in zip('ABCD', item.get('options_zh') or item['options'])}
        out['正确答案'] = item['answer_letter']
        fps = item.get('fps') or 1
        out['粗观测帧时间s'] = [round(i / fps, 2) for i in meta.get('coarse_idx', [])]
    out['E*窗s'] = [round(v, 2) for v in rec['estar']] if rec.get('estar') else None
    out['decoy窗s'] = [round(v, 2) for v in rec['decoy']] if rec.get('decoy') else None
    out['六条件 p(正确)'] = {k: au[k]['p_gt'] for k in
                          ('none', 'true', 'decoy', 'irr', 'twin', 'dense_matched') if k in au}
    out['指标'] = {k: m[k] for k in ('EEG', 'ERG', 'gain_over_dense', 'false_resolution', 'twin_flipped')
                 if k in m}
    if ag:
        prof = ag.get('entropy_profile') or {}
        out['agentic'] = {
            '预测': ag.get('pred'), '正确': ag.get('correct'),
            '时间命中E*': ag.get('temporal_hit'), '先答后看': ag.get('pre_answer'),
            '用过的工具': ag.get('tools_used'),
            '逐轮': _turn_records(ag),
            'entropy': {
                '输出token数': prof.get('n_tokens'), '平均熵': prof.get('mean_entropy'),
                '工具调用起点%': prof.get('markers_pct', {}).get('tool_call'),
                '</think>位置%': prof.get('markers_pct', {}).get('think_end'),
                '<answer>位置%': prof.get('markers_pct', {}).get('answer'),
                '调用前平均熵': prof.get('ent_pre_tool'), '调用后平均熵': prof.get('ent_post_tool'),
                'top20%高熵token位置直方图(10bin)': prof.get('signature'),
                '最高熵的token(token,熵,位置%)': prof.get('top_tokens'),
            } if prof else None,
        }
    return out


def _compact_dumps(obj):
    """indent=1 但把纯数字/短字符串的列表压回一行, 免得 [2.52, 2.52] 占四行."""
    s = json.dumps(obj, ensure_ascii=False, indent=1)
    def _collapse(m):
        inner = re.sub(r'\s+', ' ', m.group(0)).replace('[ ', '[').replace(' ]', ']')
        return inner if len(inner) <= 110 else m.group(0)
    # 只压不含嵌套 [ { 的列表
    return re.sub(r'\[(?:[^\[\]{}]|\n)*?\]', _collapse, s)


def export_json(recs, items_by_uuid, path):
    data = [concise_record(r, items_by_uuid.get(r['uuid'])) for r in recs]
    with open(path, 'w') as f:
        f.write(_compact_dumps(data))
    return path


# ───────────────────────── 图 ─────────────────────────

def plot_item(rec, item, path):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = CJK; plt.rcParams['axes.unicode_minus'] = False
    ag = rec.get('agentic'); prof = (ag or {}).get('entropy_profile')
    if not prof:
        return None
    e = np.array(ag['entropy']); n = len(e)
    x = 100 * np.arange(n) / max(1, n - 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[3, 1.2]))
    a1.plot(x, e, lw=0.8, color='#457b9d')
    thr = np.sort(e)[::-1][max(0, int(round(0.2 * n)) - 1)] if n else 0
    hi = e >= thr
    a1.scatter(x[hi], e[hi], s=10, color='#e76f51', zorder=3, label='top-20% 高熵 token')
    mk = prof['markers_pct']
    for p in mk.get('tool_call', []):
        a1.axvline(p, color='#c05621', ls='--', lw=1.2)
        a1.text(p, e.max() * 0.95, f'<tool_call> {p:.0f}%', rotation=90, va='top', ha='right', fontsize=8, color='#c05621')
    for p in mk.get('think_end', []):
        a1.axvline(p, color='#2b6cb0', ls=':', lw=1)
    for p in mk.get('answer', []):
        a1.axvline(p, color='#2f855a', ls='-', lw=1.2)
        a1.text(p, e.max() * 0.95, f'<answer> {p:.0f}%', rotation=90, va='top', ha='right', fontsize=8, color='#2f855a')
    for p in mk.get('turn_starts', [])[1:]:
        a1.axvline(p, color='gray', ls='-', lw=0.6, alpha=0.5)
    a1.set_xlabel('输出位置（占总输出 token 的 %）'); a1.set_ylabel('next-token 熵')
    ok = '✓' if ag.get('correct') else '✗'
    a1.set_title(f"{rec.get('domain')} / {rec.get('name')}  预测 {ag.get('pred')} {ok}  "
                 f"（橙虚线=工具调用起点，蓝点线=</think>，绿线=<answer>）", fontsize=9)
    a1.legend(fontsize=7, loc='upper left')
    a2.bar(np.arange(10) * 10 + 5, prof['signature'], width=9, color='#e76f51')
    a2.set_xlabel('位置 %'); a2.set_ylabel('高熵 token 占比'); a2.set_title('entropy signature (PEA)', fontsize=9)
    a2.set_xlim(0, 100)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_aggregate(recs, path, title=''):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = CJK; plt.rcParams['axes.unicode_minus'] = False
    ags = [r['agentic'] for r in recs if r.get('agentic') and r['agentic'].get('entropy_profile')]
    if not ags:
        return None
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.6))
    # 1) 平均 signature, 按答对/答错
    xs = np.arange(10) * 10 + 5
    for ok, col, lab in [(True, '#2a9d8f', '答对'), (False, '#e76f51', '答错')]:
        S = np.array([a['entropy_profile']['signature'] for a in ags if bool(a.get('correct')) == ok])
        if len(S):
            a1.plot(xs, S.mean(0), 'o-', color=col, label=f'{lab} (n={len(S)})')
    a1.set_xlabel('输出位置 %'); a1.set_ylabel('top-20% 高熵 token 占比'); a1.set_title('平均 entropy signature', fontsize=10)
    a1.legend(fontsize=8); a1.set_xlim(0, 100)
    # 2) 工具调用起点分布
    tc = [p for a in ags for p in a['entropy_profile']['markers_pct'].get('tool_call', [])[:1]]
    if tc:
        a2.hist(tc, bins=10, range=(0, 100), color='#c05621', alpha=0.8)
        a2.axvline(np.median(tc), color='k', ls='--', lw=1)
        a2.text(np.median(tc), a2.get_ylim()[1] * 0.9, f' 中位 {np.median(tc):.0f}%', fontsize=8)
    a2.set_xlabel('首个 <tool_call> 出现位置 %'); a2.set_ylabel('条数'); a2.set_title('工具调用起点', fontsize=10)
    # 3) 调用前 vs 调用后平均熵
    pre = [a['entropy_profile']['ent_pre_tool'] for a in ags if a['entropy_profile'].get('ent_pre_tool') is not None]
    post = [a['entropy_profile']['ent_post_tool'] for a in ags if a['entropy_profile'].get('ent_post_tool') is not None]
    if pre and post:
        a3.bar([0, 1], [np.mean(pre), np.mean(post)], color=['#2b6cb0', '#457b9d'])
        a3.set_xticks([0, 1]); a3.set_xticklabels(['调用前', '调用后'])
        for i, v in enumerate([np.mean(pre), np.mean(post)]):
            a3.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    a3.set_ylabel('平均 next-token 熵'); a3.set_title(f'工具调用前后 (n={len(pre)})', fontsize=10)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def export_all(recs, items_by_uuid, tag, per_item=True):
    os.makedirs(FIGS, exist_ok=True)
    jp = export_json(recs, items_by_uuid, os.path.join(HERE, 'out', f'trajectories_{tag}.json'))
    n = 0
    if per_item:
        d = os.path.join(FIGS, f'entropy_{tag}'); os.makedirs(d, exist_ok=True)
        for r in recs:
            if r.get('agentic'):
                if plot_item(r, items_by_uuid.get(r['uuid']), os.path.join(d, f"{r['uuid'][:8]}_{r.get('group','')}.png")):
                    n += 1
    ap = plot_aggregate(recs, os.path.join(FIGS, f'entropy_summary_{tag}.png'), title=f'agentic 轨迹 entropy 汇总 · {tag}')
    return dict(json=jp, per_item_figs=n, summary_fig=ap)
