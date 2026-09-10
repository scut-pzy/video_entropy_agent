"""阶段三: 在认证通过的题上跑六条件审计 + agentic, 出 EEG / Twin Sensitivity / 时空命中率. 可续跑.

依赖 out/screen.jsonl 里已有的 screen + localize 记录 (E* 和 decoy 窗从那里取).
对照组: 同时对一部分 already_solvable (N=0) 也跑审计 —— 它们的 EEG 应该 ≈ 0, 是内部自洽性检查.

--certifier: 用哪个模型的筛选/窗搜索来决定"哪些题需要证据、E* 在哪". 真正的实验设计是
  强模型认证 (certifier=qwen35) → 被测模型审计 (model=deepeyes): 强模型说这题需要证据且证据在这儿,
  看 DeepEyes 拿到证据用不用、自己找不找得到. 默认 certifier=model (自己认证自己, 弱).

用法:
  CUDA_VISIBLE_DEVICES=1 python run_audit.py --model deepeyes --certifier qwen35
  CUDA_VISIBLE_DEVICES=1 python run_audit.py --model qwen35 --no-agentic     # 强模型自己的 Utilization
  python run_audit.py --summarize --model deepeyes --certifier qwen35
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import items as I, frames as F, belief as B, pipeline as P, driver as D

HERE = os.path.dirname(os.path.abspath(__file__))
SCREEN = os.path.join(HERE, 'out', 'screen.jsonl')
OUT = os.path.join(HERE, 'out', 'audit.jsonl')


def select(screen_path, certifier, n_control=40):
    """返回 [(item, group, estar, decoy)]; group ∈ {'required','control'}. 题集与 E* 都来自 certifier."""
    scr = {r['uuid']: r for r in D.read_records(screen_path, stage='screen', model=certifier)}
    loc = {r['uuid']: r for r in D.read_records(screen_path, stage='localize', model=certifier)}
    all_items = {x['uuid']: x for x in I.load_items()}
    req, ctrl = [], []
    for u, s in scr.items():
        it = all_items.get(u)
        if it is None:
            continue
        if s['label'] == 'candidate' and u in loc:
            lab, _ = P.certify(s['screen'], loc[u]['localize'])
            if lab == 'evidence_required':
                e = loc[u]['localize']['estar']; d = loc[u]['localize']['decoy']
                req.append((it, 'required', (e['t0'], e['t1']),
                            (d['t0'], d['t1']) if d else None))
        elif s['label'] == 'already_solvable':
            ctrl.append(u)
    # N=0 对照: 用中段窗当 "E*", 同宽度最远窗当 decoy (它们本来就不需要证据)
    rng = np.random.RandomState(0)
    for u in rng.permutation(sorted(ctrl))[:n_control]:
        it = all_items[u]; dur = it['duration']
        ctrl_e = (0.35 * dur, 0.65 * dur)
        ctrl_d = (0.0, 0.30 * dur)
        req.append((it, 'control', ctrl_e, ctrl_d))
    return req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepeyes')
    ap.add_argument('--certifier', default=None, help='默认与 --model 相同')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--lang', default='zh')
    ap.add_argument('--n-perm', type=int, default=4)
    ap.add_argument('--n-control', type=int, default=40)
    ap.add_argument('--screen', default=SCREEN)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--no-agentic', action='store_true')
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    cert = a.certifier or a.model
    tag = a.model if cert == a.model else f'{a.model}@{cert}'    # 记录里的 model 字段带上认证者

    if a.summarize:
        summarize(a.out, tag); export(a.out, tag); return

    units = select(a.screen, cert, a.n_control)
    n_req = sum(1 for _, g, _, _ in units if g == 'required')
    print(f'审计单元 (认证者={cert}, 被测={a.model}): {n_req} 条 evidence_required + {len(units)-n_req} 条 N=0 对照')
    if not units:
        print('screen.jsonl 里还没有认证通过的题, 先跑 run_screen.py'); return

    sm = I.sibling_map(I.load_items())
    all_items = {x['uuid']: x for x in I.load_items()}
    bk = B.Backend(a.model, device=a.device)

    def one(payload):
        it, group, estar, decoy = payload
        twin_item = twin_win = None
        sibs = sm.get(it['uuid'], [])
        if sibs:
            twin_item = all_items.get(sibs[0]['uuid'])
            if twin_item and twin_item.get('duration'):
                twin_win = (0.35 * twin_item['duration'], 0.65 * twin_item['duration'])
                F.prefetch([twin_item], ks=(8,), verbose=False)
        au = P.audit_item(bk, it, estar, decoy, twin_item, twin_win,
                          n_perm=a.n_perm, lang=a.lang)
        m = P.metrics(au)
        rec = dict(group=group, certifier=cert, domain=it['domain'], name=it['name'], estar=estar,
                   decoy=decoy, twin=sibs[0]['option'] if sibs else None,
                   audit=au, metrics=m)
        if not a.no_agentic:
            rec['agentic'] = P.agentic_item(bk, it, estar=estar, lang=a.lang)
        return rec

    D.run([((it['uuid'], tag, 'audit'), (it, g, e, d)) for it, g, e, d in units],
          one, a.out, desc=f'audit/{tag}', flush_every=5)
    summarize(a.out, tag)
    export(a.out, tag)


def export(path, tag):
    """精简 JSON (out/trajectories_<tag>.json) + entropy 图 (out/figs/entropy_<tag>/, entropy_summary_<tag>.png)."""
    import traj_viz as V
    recs = D.read_records(path, stage='audit', model=tag)
    if not recs:
        return
    by = {x['uuid']: x for x in I.load_items()}
    # 中文题面从 pilot 同款字段来: 这里 items 只有英文 stem/options, 精简 JSON 里照原样放
    info = V.export_all(recs, by, tag)
    print(f"\n精简 JSON: {info['json']}\nentropy 图: {info['per_item_figs']} 张逐条 + 汇总 {info['summary_fig']}")


def summarize(path=OUT, model=None):
    recs = D.read_records(path, stage='audit', model=model)
    if not recs:
        print('还没有审计结果'); return
    print(f'\n=== 四条件审计汇总 ({len(recs)} 条) ===')
    for group in ('required', 'control'):
        g = [r for r in recs if r.get('group') == group]
        if not g:
            continue
        M = [r['metrics'] for r in g]

        def mean(k):
            v = [m[k] for m in M if m.get(k) is not None]
            return float(np.mean(v)) if v else float('nan')

        def rate(k):
            v = [bool(m[k]) for m in M if m.get(k) is not None]
            return (float(np.mean(v)), len(v)) if v else (float('nan'), 0)

        print(f'\n[{group}] n={len(g)}')
        print(f"  p(正确): none={mean('p_none'):.3f}  true={mean('p_true'):.3f}  "
              f"decoy={mean('p_decoy'):.3f}  irr={mean('p_irr'):.3f}  dense_matched={mean('p_dense'):.3f}")
        print(f"  EEG = log p(true) - log p(decoy) = {mean('EEG'):+.3f}")
        print(f"  ERG = acc(true) - acc(decoy)     = {np.mean([m['ERG'] for m in M if m.get('ERG') is not None]):+.3f}")
        print(f"  比等预算密采多拿到          = {mean('gain_over_dense'):+.3f}")
        fr, nfr = rate('false_resolution'); print(f"  False Resolution 率 = {fr:.0%} (n={nfr})")
        tf, ntf = rate('twin_flipped'); print(f"  Twin 翻转率        = {tf:.0%} (n={ntf})")
        ag = [r['agentic'] for r in g if r.get('agentic')]
        if ag:
            print(f"  agentic: acc={np.mean([x['correct'] for x in ag]):.0%}  "
                  f"时间命中={np.mean([x.get('temporal_hit', False) for x in ag]):.0%}  "
                  f"先答后看={np.mean([x['pre_answer'] for x in ag]):.0%}  "
                  f"平均工具调用={np.mean([x['n_calls_ok'] for x in ag]):.1f}  "
                  f"用过 seek 的比例={np.mean([('video_seek_tool' in x['tools_used']) for x in ag]):.0%}")


if __name__ == '__main__':
    main()
