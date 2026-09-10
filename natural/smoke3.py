"""3 条自然题的端到端冒烟: 帧缓存 → 信念曲线 → E* 滑窗定位 → 六条件审计 → agentic 轨迹.

用法:
  CUDA_VISIBLE_DEVICES=1 python smoke3.py --model deepeyes [--n 3] [--domains Cooking]
  (27B 认证者: 在 verl_qwen35 env 里 --model qwen38, 需要独立 GPU)
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import items as I, frames as F, belief as B, pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))


def pick(n=3, domains=None, seed=0):
    """挑覆盖面广的几条: 不同域、长短片各有、优先 >=720p."""
    its = I.load_items(domains=domains)
    hi = [x for x in its if x['width'] and min(x['width'], x['height']) >= 720]
    hi.sort(key=lambda x: x['duration'])
    if len(hi) < n:
        hi = sorted(its, key=lambda x: x['duration'])
    picks, seen = [], set()
    # 短 / 中 / 长 各一条, 且域不重复
    for frac in [0.15, 0.5, 0.85][:n]:
        j = int(frac * (len(hi) - 1))
        for step in range(len(hi)):
            for jj in (j + step, j - step):
                if 0 <= jj < len(hi) and hi[jj]['domain'] not in seen:
                    picks.append(hi[jj]); seen.add(hi[jj]['domain']); break
            else:
                continue
            break
    return picks[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepeyes')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--domains', nargs='*', default=None)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--lang', default='zh')
    ap.add_argument('--no-agentic', action='store_true')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()

    picks = pick(a.n, a.domains)
    print(f'挑中 {len(picks)} 条:')
    for it in picks:
        print(f"  {it['uuid'][:8]}  {it['domain']:20s} {str(it['name'])[:22]:22s} "
              f"{it['duration']:5.1f}s {it['width']}x{it['height']} {it['n_frames']}帧  "
              f"答案 {it['answer_letter']}. {it['answer']}")
    print('\n预取帧 ...', F.prefetch(picks, verbose=False))

    sm = I.sibling_map(I.load_items())
    bk = B.Backend(a.model, device=a.device)
    out = {}
    for it in picks:
        t0 = time.time()
        print(f"\n{'='*80}\n{it['uuid'][:8]} | {it['domain']} | {it['name']} | "
              f"{it['duration']:.1f}s\n题目: {it['stem']}\n选项: " +
              ' / '.join(f'{c}. {o}' for c, o in zip('ABCD', it['options'])) +
              f"\n正确答案: {it['answer_letter']} ({it['answer']})\n定义: {str(it['definition'])[:150]}\n{'='*80}")

        # ── 阶段一 ──
        sc = P.screen_item(bk, it, lang=a.lang)
        print('[阶段一] 信念曲线 (p_gt / 预测 / 熵 / 置换标准差):')
        for c in ['text'] + [f'k{k}' for k in P.SCREEN_KS] + [f'k{P.BUDGET_K}_hi']:
            if c not in sc:
                continue
            b = sc[c]
            print(f"  {c:8s} p_gt={b['p_gt']:.3f}  pred={'ABCD'[b['pred_idx']]}"
                  f"{'✓' if b['correct'] else '✗'}  H={b['entropy']:.2f}  "
                  f"σ_perm={b['perm_std']:.3f}  帧数={b.get('n_frames_used','-')}")
        label, why = P.classify(sc)
        print(f'  → 初筛: {label}  ({why})')

        # ── 阶段二 ──
        print('[阶段二] 滑窗定位 E* ...')
        loc = P.localize_item(bk, it, lang=a.lang)
        e = loc['estar']; d = loc['decoy']
        print(f"  共 {loc['n_windows']} 个窗; E* = [{e['t0']:.2f},{e['t1']:.2f}]s "
              f"({e['frac']:.0%}) p_gt={e['p_gt']:.3f} {'(认证通过)' if loc['certified'] else '(未达阈值,取最优窗)'}")
        if d:
            print(f"  decoy = [{d['t0']:.2f},{d['t1']:.2f}]s p_gt={d['p_gt']:.3f}")
        top = sorted(loc['windows'], key=lambda r: -r['p_gt'])[:5]
        print('  top-5 窗: ' + '  '.join(f"[{r['t0']:.1f},{r['t1']:.1f}]={r['p_gt']:.2f}" for r in top))
        label, why = P.certify(sc, loc)
        print(f'  → 最终归类: {label}  ({why})')

        # ── 阶段三 ──
        twin_item, twin_win = None, None
        sibs = sm.get(it['uuid'], [])
        if sibs:
            all_items = {x['uuid']: x for x in I.load_items()}
            twin_item = all_items.get(sibs[0]['uuid'])
            if twin_item and twin_item.get('duration'):
                twin_win = (0.35 * twin_item['duration'], 0.65 * twin_item['duration'])
                F.prefetch([twin_item], ks=(8,), verbose=False)
        print(f"[阶段三] 六条件审计 (twin={'有 ' + sibs[0]['option'] if twin_item else '无同域兄弟视频'})")
        au = P.audit_item(bk, it, (e['t0'], e['t1']),
                          (d['t0'], d['t1']) if d else None,
                          twin_item, twin_win, lang=a.lang)
        for c in ['none', 'true', 'decoy', 'irr', 'twin', 'dense_matched']:
            if c not in au:
                continue
            b = au[c]
            extra = f"  k={b['k']}(实际{b['n_frames_used']}帧)" if c == 'dense_matched' else ''
            extra += f"  p_twin={b['p_twin']:.3f}" if c == 'twin' and b.get('p_twin') is not None else ''
            print(f"  {c:14s} p_gt={b['p_gt']:.3f}  pred={'ABCD'[b['pred_idx']]}"
                  f"{'✓' if b['correct'] else '✗'}  H={b['entropy']:.2f}{extra}")
        m = P.metrics(au)
        print('  指标: ' + json.dumps(m, ensure_ascii=False))

        # ── agentic ──
        ag = None
        if not a.no_agentic:
            print('[Acquisition] agentic (zoom+seek 自己决定) ...')
            ag = P.agentic_item(bk, it, estar=(e['t0'], e['t1']), lang=a.lang, verbose=True)
            calls = [(c['name'], 'ok' if c['ok'] else c['err'], c['window']) for c in ag['tool_calls']]
            print(f"  预测={ag['pred']} {'✓' if ag['correct'] else '✗'} | 轮数={len(ag['turns'])} | "
                  f"工具={calls} | 先答后看={ag['pre_answer']} | "
                  f"时间命中 IoU={ag.get('temporal_iou_max')}")

        out[it['uuid']] = dict(item={k: it[k] for k in
                               ('key', 'uuid', 'domain', 'name', 'definition', 'stem', 'options',
                                'answer_letter', 'answer_idx', 'duration', 'n_frames', 'fps',
                                'width', 'height')},
                               screen=sc, label=label, why=why,
                               localize={k: v for k, v in loc.items()},
                               audit=au, metrics=m, agentic=ag,
                               wall_s=round(time.time() - t0, 1))
        print(f"  用时 {time.time()-t0:.0f}s")

    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    tag = a.tag or a.model
    p = os.path.join(HERE, 'out', f'smoke3_{tag}.json')
    json.dump(out, open(p, 'w'), ensure_ascii=False, indent=1, default=str)
    print(f'\n-> {p}\n帧缓存: {F.cache_stats()}')


if __name__ == '__main__':
    main()
