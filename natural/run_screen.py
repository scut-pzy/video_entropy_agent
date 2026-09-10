"""阶段一+二: 必要性筛选 (信念 vs 观测预算) → 对候选题做 E* 窗搜索 → 认证. 可续跑.

为什么两步合在一个脚本里: 必要性的定义是"给了 E* 才可解", 而 E* 是**窗**不是均匀预算 ——
冒烟里的 Cricket 那条均匀 k32 只有 0.07, 但 [1.8,3.6]s 密采到 0.70, EEG=1.69. 所以初筛只能
筛掉 text_shortcut / already_solvable, 剩下的都得进窗搜索才知道是不是 evidence-required.

用法:
  CUDA_VISIBLE_DEVICES=1 python run_screen.py --model deepeyes            # 两个阶段都跑
  CUDA_VISIBLE_DEVICES=1 python run_screen.py --model deepeyes --stage screen
  python run_screen.py --summarize                                        # 只看统计, 不用 GPU
输出 out/screen.jsonl, 断了重跑同一命令即可续.
"""
import argparse, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import items as I, frames as F, belief as B, pipeline as P, driver as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'screen.jsonl')


def summarize(path=OUT, model=None):
    scr = {r['uuid']: r for r in D.read_records(path, stage='screen', model=model)}
    loc = {r['uuid']: r for r in D.read_records(path, stage='localize', model=model)}
    rows = []
    for u, s in scr.items():
        label, why = s['label'], s['why']
        if label == 'candidate' and u in loc:
            label, why = P.certify(s['screen'], loc[u]['localize'])
        rows.append(dict(uuid=u, domain=s['domain'], name=s['name'], label=label, why=why,
                         p_k8=s['screen']['k8']['p_gt'],
                         p_k32=s['screen'].get('k32', {}).get('p_gt'),
                         p_hi=s['screen'].get('k8_hi', {}).get('p_gt'),
                         p_text=s['screen']['text']['p_gt'],
                         p_estar=loc[u]['localize']['estar']['p_gt'] if u in loc else None,
                         estar=[loc[u]['localize']['estar']['t0'], loc[u]['localize']['estar']['t1']]
                         if u in loc else None,
                         duration=s['duration']))
    c = Counter(r['label'] for r in rows)
    print(f'\n=== 阶段一+二 汇总 ({len(rows)} 条已筛, {len(loc)} 条已做窗搜索) ===')
    for k, v in c.most_common():
        print(f'  {k:20s} {v:4d}  ({v/max(1,len(rows)):.0%})')
    need = c['evidence_required']
    pend = c['candidate']
    print(f'\nEvidence-Required@8 = {need} 条' + (f'（还有 {pend} 条候选没做窗搜索）' if pend else ''))
    if not pend:
        print('  → ' + ('≥50，可直接进四条件审计' if need >= 50 else
                        ('20-50，审计照做但统计力偏弱，建议补下载' if need >= 20 else
                         '<20，短剪辑里 necessity 稀有，走加长上下文的 contingency')))
    er = [r for r in rows if r['label'] == 'evidence_required']
    if er:
        dc = Counter(r['domain'] for r in er)
        print('  域分布:', dict(dc.most_common(10)))
        only_win = [r for r in er if '只有定位到窗才行' in r['why']]
        print(f'  其中"只有定位到窗才行"（均匀加帧/加分辨率都不行）: {len(only_win)} 条')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepeyes')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--lang', default='zh')
    ap.add_argument('--n-perm', type=int, default=4)
    ap.add_argument('--loc-n-perm', type=int, default=2)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--domains', nargs='*', default=None)
    ap.add_argument('--stage', default='both', choices=['screen', 'localize', 'both'])
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--prefetch-only', action='store_true')
    a = ap.parse_args()

    if a.summarize:
        summarize(a.out, None if a.model == 'all' else a.model)
        return

    its = I.load_items(domains=a.domains)
    if a.limit:
        its = its[:a.limit]
    print(f'{len(its)} 条题; 帧缓存现状 {F.cache_stats()}')
    if a.prefetch_only:
        print(F.prefetch(its)); return

    bk = B.Backend(a.model, device=a.device)

    if a.stage in ('screen', 'both'):
        def one(it):
            F.prefetch([it], verbose=False)
            sc = P.screen_item(bk, it, n_perm=a.n_perm, lang=a.lang)
            label, why = P.classify(sc)
            return dict(key=it['key'], domain=it['domain'], name=it['name'],
                        duration=it['duration'], width=it['width'], height=it['height'],
                        n_frames=it['n_frames'], answer_idx=it['answer_idx'],
                        label=label, why=why, screen=sc)
        D.run([((it['uuid'], a.model, 'screen'), it) for it in its], one, a.out,
              desc=f'screen/{a.model}', flush_every=5)

    if a.stage in ('localize', 'both'):
        scr = {r['uuid']: r for r in D.read_records(a.out, stage='screen', model=a.model)}
        cand = [it for it in its if scr.get(it['uuid'], {}).get('label') == 'candidate']
        print(f'\n候选 {len(cand)} 条进窗搜索（跳过 text_shortcut / already_solvable）')

        def one_loc(it):
            loc = P.localize_item(bk, it, n_perm=a.loc_n_perm, lang=a.lang)
            lab, why = P.certify(scr[it['uuid']]['screen'], loc)
            return dict(domain=it['domain'], name=it['name'], label=lab, why=why, localize=loc)
        D.run([((it['uuid'], a.model, 'localize'), it) for it in cand], one_loc, a.out,
              desc=f'localize/{a.model}', flush_every=5)

    summarize(a.out, a.model)


if __name__ == '__main__':
    main()
