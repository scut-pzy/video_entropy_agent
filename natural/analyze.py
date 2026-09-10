"""把 screen.jsonl / audit.jsonl 变成表和图 (不用 GPU).

用法:
  python analyze.py                 # 打印全部表
  python analyze.py --figs          # 另外生成 out/figs/*.png
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P, driver as D

HERE = os.path.dirname(os.path.abspath(__file__))
SCREEN = os.path.join(HERE, 'out', 'screen.jsonl')
AUDIT = os.path.join(HERE, 'out', 'audit.jsonl')
FIGS = os.path.join(HERE, 'out', 'figs')
CJK = ['WenQuanYi Zen Hei', 'DejaVu Sans']


def final_labels(screen_path=SCREEN, model='deepeyes'):
    """每条题的最终标签 + 关键数字."""
    scr = {r['uuid']: r for r in D.read_records(screen_path, stage='screen', model=model)}
    loc = {r['uuid']: r for r in D.read_records(screen_path, stage='localize', model=model)}
    rows = []
    for u, s in scr.items():
        lab, why = s['label'], s['why']
        est = None
        if lab == 'candidate':
            if u in loc:
                lab, why = P.certify(s['screen'], loc[u]['localize'])
                est = loc[u]['localize']['estar']
            else:
                lab, why = 'pending', '窗搜索还没跑'
        sc = s['screen']
        rows.append(dict(uuid=u, domain=s['domain'], name=s['name'], duration=s['duration'],
                         width=s.get('width'), height=s.get('height'),
                         label=lab, why=why,
                         **{c: sc[c]['p_gt'] for c in sc},
                         perm_std_k8=sc['k8']['perm_std'],
                         p_estar=est['p_gt'] if est else None,
                         estar=[est['t0'], est['t1']] if est else None,
                         estar_frac=est['frac'] if est else None,
                         only_window=('只有定位到窗才行' in why)))
    return rows


def table_labels(rows):
    c = Counter(r['label'] for r in rows)
    print(f'\n=== 归类分布 (n={len(rows)}) ===')
    for k, v in c.most_common():
        print(f'  {k:20s} {v:4d}  {v/len(rows):5.1%}')
    er = [r for r in rows if r['label'] == 'evidence_required']
    if er:
        ow = [r for r in er if r['only_window']]
        print(f'\nEvidence-Required@8 = {len(er)} 条，其中"只有定位到窗才行" {len(ow)} 条 '
              f'(均匀加帧、加分辨率都解不了 —— 这类最能说明主动取证的必要性)')
        print('  域分布:', dict(Counter(r['domain'] for r in er).most_common(8)))
        print(f"  E* 窗宽中位数: {np.median([r['estar_frac'] for r in er if r['estar_frac']]):.0%} of 时长; "
              f"视频时长中位数 {np.median([r['duration'] for r in er]):.1f}s")
    return c


def table_budget_curve(rows):
    """necessity vs 观测预算: 各标签下 p_gt 随 k 的均值."""
    ks = ['text', 'k2', 'k4', 'k8', 'k16', 'k32', 'k8_hi']
    print(f"\n=== 信念 vs 观测预算 (各组 p(正确) 均值) ===\n{'组':22s}" +
          ''.join(f'{k:>8s}' for k in ks) + '     n')
    for lab in ['already_solvable', 'evidence_required', 'borderline', 'unsolvable', 'text_shortcut']:
        g = [r for r in rows if r['label'] == lab]
        if not g:
            continue
        cells = []
        for k in ks:
            v = [r[k] for r in g if r.get(k) is not None]
            cells.append(f'{np.mean(v):8.3f}' if v else f'{"-":>8s}')
        print(f'{lab:22s}' + ''.join(cells) + f'{len(g):6d}')


def table_audit(audit_path=AUDIT, model='deepeyes'):
    recs = D.read_records(audit_path, stage='audit', model=model)
    if not recs:
        print('\n(还没有审计结果)')
        return []
    print(f'\n=== 四条件审计 (n={len(recs)}) ===')
    for group in ('required', 'control'):
        g = [r for r in recs if r.get('group') == group]
        if not g:
            continue
        M = [r['metrics'] for r in g]
        mean = lambda k: float(np.mean([m[k] for m in M if m.get(k) is not None])) \
            if any(m.get(k) is not None for m in M) else float('nan')
        print(f'\n[{group}] n={len(g)}')
        print(f"  p(正确):  none={mean('p_none'):.3f}  true={mean('p_true'):.3f}  "
              f"decoy={mean('p_decoy'):.3f}  irr={mean('p_irr'):.3f}  dense_matched={mean('p_dense'):.3f}")
        print(f"  EEG={mean('EEG'):+.3f}   ERG={mean('ERG'):+.3f}   "
              f"比等预算密采多={mean('gain_over_dense'):+.3f}")
        fr = [bool(m['false_resolution']) for m in M if m.get('false_resolution') is not None]
        tw = [bool(m['twin_flipped']) for m in M if m.get('twin_flipped') is not None]
        if fr:
            print(f"  False Resolution 率 = {np.mean(fr):.0%} (n={len(fr)})")
        if tw:
            print(f"  Twin 翻转率 = {np.mean(tw):.0%} (n={len(tw)})  "
                  f"—— 忠实用证据的模型这里应该高, 但要结合 decoy 一起看")
        ag = [r['agentic'] for r in g if r.get('agentic')]
        if ag:
            hit_note = '（对照组的 "E*" 是人为取的中段窗，而 DeepEyes 本来就爱裁中间帧，'
            hit_note += '所以这一格的高命中率是重合假象，不可解读）' if group == 'control' else ''
            print(f"  agentic: acc={np.mean([x['correct'] for x in ag]):.0%}  "
                  f"时间命中 E*={np.mean([x.get('temporal_hit', False) for x in ag]):.0%}{hit_note}  "
                  f"先答后看={np.mean([x['pre_answer'] for x in ag]):.0%}  "
                  f"平均成功调用={np.mean([x['n_calls_ok'] for x in ag]):.1f}  "
                  f"用过 seek={np.mean([('video_seek_tool' in x['tools_used']) for x in ag]):.0%}")
    return recs


def figs(rows, recs, tag=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = CJK
    plt.rcParams['axes.unicode_minus'] = False
    os.makedirs(FIGS, exist_ok=True)

    # 图1: 信念 vs 观测预算
    ks = ['k2', 'k4', 'k8', 'k16', 'k32']
    x = [2, 4, 8, 16, 32]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for lab, col in [('already_solvable', '#2a9d8f'), ('evidence_required', '#e76f51'),
                     ('borderline', '#e9c46a'), ('unsolvable', '#8d99ae')]:
        g = [r for r in rows if r['label'] == lab]
        if not g:
            continue
        y = [np.mean([r[k] for r in g if r.get(k) is not None]) for k in ks]
        ax.plot(x, y, 'o-', color=col, label=f'{lab} (n={len(g)})')
        if lab == 'evidence_required':
            pe = [r['p_estar'] for r in g if r.get('p_estar')]
            if pe:
                ax.axhline(np.mean(pe), ls='--', c=col, lw=1)
                ax.text(32, np.mean(pe) + .01, f'给了 E* 窗: {np.mean(pe):.2f}', ha='right',
                        color=col, fontsize=9)
    ax.axhline(0.25, ls=':', c='gray', lw=1); ax.set_xscale('log', base=2); ax.set_xticks(x)
    ax.set_xticklabels(x); ax.set_xlabel('均匀采样帧数 k'); ax.set_ylabel('p(正确答案)')
    ax.set_title('信念 vs 观测预算：加帧解决不了的题，定位到窗就能解')
    ax.legend(fontsize=8); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(f'{FIGS}/fig1_budget_curve{tag}.png', dpi=150); plt.close(fig)

    # 图2: 审计四条件
    if recs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        conds = ['none', 'true', 'decoy', 'irr', 'dense_matched']
        keys = ['p_none', 'p_true', 'p_decoy', 'p_irr', 'p_dense']
        for ax, group in zip(axes, ('required', 'control')):
            g = [r['metrics'] for r in recs if r.get('group') == group]
            if not g:
                continue
            y = [np.mean([m[k] for m in g if m.get(k) is not None]) for k in keys]
            ax.bar(range(len(conds)), y, color=['#8d99ae', '#2a9d8f', '#e76f51', '#adb5bd', '#457b9d'])
            ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds, rotation=20)
            ax.set_title(f'{group} (n={len(g)})'); ax.axhline(0.25, ls=':', c='gray', lw=1)
        axes[0].set_ylabel('p(正确答案)')
        fig.suptitle('四条件审计：证据是不是真的被用了', fontsize=11)
        fig.tight_layout(); fig.savefig(f'{FIGS}/fig2_audit{tag}.png', dpi=150); plt.close(fig)
    print(f'\n图已存到 {FIGS}/')


def table_cross(screen_path=SCREEN, m1='deepeyes', m2='qwen35'):
    """两个模型的最终标签交叉表 + E* 窗一致性."""
    r1 = {r['uuid']: r for r in final_labels(screen_path, m1)}
    r2 = {r['uuid']: r for r in final_labels(screen_path, m2)}
    common = sorted(set(r1) & set(r2))
    if not common:
        print(f'\n({m2} 还没有筛选结果)'); return
    labs = ['already_solvable', 'evidence_required', 'borderline', 'unsolvable', 'text_shortcut', 'pending']
    print(f'\n=== 跨模型归类交叉表 (行={m1}, 列={m2}, n={len(common)}) ===')
    print(f'{"":20s}' + ''.join(f'{l[:12]:>13s}' for l in labs))
    for a_ in labs:
        row = [sum(1 for u in common if r1[u]['label'] == a_ and r2[u]['label'] == b_) for b_ in labs]
        if sum(row):
            print(f'{a_:20s}' + ''.join(f'{v:13d}' for v in row))
    both = [u for u in common if r1[u]['estar'] and r2[u]['estar']]
    if both:
        er2 = [u for u in both if r2[u]['label'] == 'evidence_required']
        rng = np.random.RandomState(0)

        def agree(u, v, tol=0.10):
            """两窗有交 或 中心距 ≤ tol×时长 (10% 宽的相邻窗算"接近")."""
            a_, b_ = r1[v]['estar'], r2[u]['estar']; dur = r2[u]['duration'] or 1
            return P.overlap(a_, b_) > 0 or abs((a_[0] + a_[1]) / 2 - (b_[0] + b_[1]) / 2) <= tol * dur

        for name, S in [('都做了窗搜索', both), (f'{m2} 认证 evidence_required', er2)]:
            if not S:
                continue
            hit = np.mean([P.overlap(r1[u]['estar'], r2[u]['estar']) > 0 for u in S])
            near = np.mean([agree(u, u) for u in S])
            null = np.mean([np.mean([agree(u, v) for u, v in zip(S, rng.permutation(S))]) for _ in range(200)])
            print(f'\n{name} {len(S)} 条: 两模型 E* 窗 IoU>0 占 {hit:.0%}, 有交或相邻(中心距≤10%) 占 {near:.0%}, '
                  f'随机配对基线 {null:.0%}' + ('  —— 跨模型一致才说明 E* 是视频的性质不是模型的性质' if S is er2 else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepeyes', help='审计记录的 tag, 如 deepeyes / deepeyes@qwen35 / qwen35')
    ap.add_argument('--certifier', default=None, help='筛选标签来自哪个模型; 默认 = --model 里 @ 后面的, 没有 @ 就是 --model 本身')
    ap.add_argument('--screen', default=SCREEN); ap.add_argument('--audit', default=AUDIT)
    ap.add_argument('--figs', action='store_true')
    ap.add_argument('--dump', default=None, help='把逐条结果写成 tsv')
    a = ap.parse_args()
    cert = a.certifier or (a.model.split('@')[1] if '@' in a.model else a.model)
    rows = final_labels(a.screen, cert)
    if not rows:
        print('screen.jsonl 还是空的'); return
    print(f'(筛选标签来自 {cert}; 审计记录 tag = {a.model})')
    table_labels(rows)
    table_budget_curve(rows)
    recs = table_audit(a.audit, a.model)
    if cert == 'qwen35':
        table_cross(a.screen, 'deepeyes', 'qwen35')
    if a.figs:
        figs(rows, recs, tag='_' + a.model.replace('@', '_at_'))
    if a.dump:
        cols = ['uuid', 'domain', 'name', 'duration', 'label', 'text', 'k8', 'k32', 'k8_hi',
                'p_estar', 'estar_frac', 'only_window', 'why']
        with open(a.dump, 'w') as f:
            f.write('\t'.join(cols) + '\n')
            for r in rows:
                f.write('\t'.join(str(r.get(c, '')) for c in cols) + '\n')
        print(f'-> {a.dump}')


if __name__ == '__main__':
    main()
