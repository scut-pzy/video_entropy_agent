"""去结论分叉: 检验"承诺效应"——证据在模型写下结论之前会被用, 之后不会.

对 fork_v2 里每条已分叉的轨迹, 只改第一轮的**思考文本**, 工具调用参数保持模型自己的原样,
工具返回固定为两种之一 (oracle = 认证过的证据窗 E*; random = 同形状随机返回), 各自续写到作答.
思考文本三种:
  orig     模型自己写的原文 (55% 在调用前已下结论)
  trunc    截在第一次点名选项 / 出现"答案是/因此/所以…"之前 (没有结论就等于 orig)
  neutral  换成一句中性的"还定不了、需要看细节" (与题无关, 不含任何选项信息)
若承诺效应成立: orig 下 oracle≈random; trunc/neutral 下 oracle > random.
用法: CUDA_VISIBLE_DEVICES=1 python decommit.py --model deepeyes
"""
import argparse, json, os, random, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
import items as I, frames as F, belief as B, driver as D, pipeline as P
import video_imcot as vi
from prompts_zh import SYS_ZH, INSTR_ZH, CONT_ZH, PRE_ANSWER_RE
import fork as FK

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'decommit.jsonl')
NEUTRAL = '从这几帧粗采样画面还不能确定是哪一个动作，关键的动作细节可能发生在两个采样帧之间，我需要调用工具查看更多画面再判断。'
CONCL = re.compile(r'答案是|正确答案|应该选|应选|因此|所以|综上|可以判断|可以确定|这是一个|这是典型|属于|即为|看起来像|选项\s*[ABCD]|[\(（]?[ABCD][\)）]?\s*[\.．、]')


def split_turn1(t1):
    """第一轮 → (think, tool_call 块). DeepEyes 格式: <think>…</think> <tool_call>…</tool_call>"""
    m = re.search(r'<tool_call>.*?</tool_call>', t1, re.S)
    call = m.group(0) if m else ''
    think = t1[:m.start()] if m else t1
    think = re.sub(r'</?think>', '', think).strip()
    return think, call


def truncate(think, options):
    """截在第一次点名选项 / 结论性措辞之前 (按句切, 保留之前的完整句子)."""
    low = think.lower(); cut = len(think)
    for o in options:
        j = low.find(o.lower())
        if j >= 0:
            cut = min(cut, j)
    m = CONCL.search(think)
    if m:
        cut = min(cut, m.start())
    if cut >= len(think):
        return think, False
    head = think[:cut]
    k = max(head.rfind(p) for p in '。！？；\n')       # 回退到上一个完整句子
    head = head[:k + 1] if k >= 0 else ''
    return head.strip(), True


def run_item(bk, it, rec, seed=0, max_turns=3, mnt=1024):
    rng = random.Random(seed + int(it['uuid'][:6], 16))
    sess = vi.VideoSession(it['video'], n_coarse=P.BUDGET_K); coarse = list(range(len(sess.frames)))
    header = f'以下是从一段 {sess.duration:.1f} 秒的视频中均匀抽取的 {len(coarse)} 帧画面，每帧都标注了序号和时间戳。\n'
    q = it['stem'] + '\n' + '\n'.join(f'{c}. {o}' for c, o in zip('ABCD', it['options']))
    base = [{'role': 'system', 'content': SYS_ZH},
            {'role': 'user', 'content': [{'type': 'text', 'text': header}] + P._labeled(sess, coarse, 'zh')
             + [{'type': 'text', 'text': '问题：' + q + INSTR_ZH}]}]
    gt = it['answer_letter']; gi = it['answer_idx']
    think, call_txt = split_turn1(rec['turn1'])
    tr, did_trunc = truncate(think, it['options'])
    variants = {'orig': think, 'trunc': tr, 'neutral': NEUTRAL}
    # 两种返回: 预先生成帧索引 (三种思考共用同一组返回, 只让思考文本变)
    call = rec['call']
    first = dict(name=call['name'], args=call['args'])
    ret = {'oracle': sess.tool_seek(dict(start_time=float(rec['estar'][0]), end_time=float(rec['estar'][1]), num_frames=6)),
           'random': FK._random_return(sess, first, rng)}
    out = dict(did_trunc=did_trunc, think_orig=think, think_trunc=tr, call=call_txt, pre_answer=rec['pre_answer'], branches={})
    for vname, th in variants.items():
        # Qwen3.5 的模板会删掉历史 assistant 里的 <think>…</think>, 所以对它把思考写成普通文本放在调用前
        t1 = (f'{th}\n{call_txt}' if getattr(bk, 'thinking_template', False) else f'<think>{th}</think>\n{call_txt}')
        for rname, idx in ret.items():
            obs = P._labeled(sess, idx, 'zh') + [{'type': 'text', 'text': CONT_ZH}]
            msgs = base + [{'role': 'assistant', 'content': t1}, {'role': 'user', 'content': obs}]
            p, gap, pv = FK._probe_gap(bk, msgs, gi)
            cont = FK._continue(bk, [m for m in msgs], sess, gt, max_turns, mnt)
            cont.pop('text', None) if False else None
            out['branches'][f'{vname}|{rname}'] = dict(pred=cont['pred'], correct=cont['correct'], gap=gap, p_gt=p,
                                                        ent_think=cont['ent_think'], n_tokens=cont['n_tokens'],
                                                        text=cont['text'][:1500])
    return out


def summarize(path=OUT, model=None):
    recs = D.read_records(path, stage='decommit', model=model)
    if not recs:
        print('还没有结果'); return
    for grp in ('required', 'control'):
        g = [r for r in recs if r.get('group') == grp]
        if not g:
            continue
        print(f'\n[{grp}] n={len(g)}   其中第一轮被截断(原文含结论)的 {sum(r["did_trunc"] for r in g)} 条')
        print(f"{'思考':8s} | {'acc E*':>7s} {'acc 随机':>8s} {'Δacc':>6s} | {'gap E*':>7s} {'gap 随机':>8s} {'VEG':>6s} | {'H E*':>6s} {'H 随机':>7s} {'EER':>6s} | 答案随返回变")
        for v in ('orig', 'trunc', 'neutral'):
            b = lambda r, x: r['branches'][f'{v}|{x}']
            ao = np.mean([b(r, 'oracle')['correct'] for r in g]); ar = np.mean([b(r, 'random')['correct'] for r in g])
            go = np.mean([b(r, 'oracle')['gap'] for r in g]); gr = np.mean([b(r, 'random')['gap'] for r in g])
            ho = [b(r, 'oracle')['ent_think'] for r in g if b(r, 'oracle')['ent_think'] is not None]
            hr = [b(r, 'random')['ent_think'] for r in g if b(r, 'random')['ent_think'] is not None]
            flip = np.mean([b(r, 'oracle')['pred'] != b(r, 'random')['pred'] for r in g])
            print(f"{v:8s} | {ao:7.2f} {ar:8.2f} {ao-ar:+6.2f} | {go:+7.2f} {gr:+8.2f} {go-gr:+6.2f} | {np.mean(ho):6.3f} {np.mean(hr):7.3f} {np.mean(hr)-np.mean(ho):+6.3f} | {flip:.0%}")
        # 只看原文含结论的子集 (承诺效应的核心子集)
        t = [r for r in g if r['did_trunc']]
        if t:
            print(f'  —— 仅"原文调用前含结论"的 {len(t)} 条:')
            for v in ('orig', 'trunc', 'neutral'):
                ao = np.mean([r['branches'][f'{v}|oracle']['correct'] for r in t]); ar = np.mean([r['branches'][f'{v}|random']['correct'] for r in t])
                go = np.mean([r['branches'][f'{v}|oracle']['gap'] for r in t]); gr = np.mean([r['branches'][f'{v}|random']['gap'] for r in t])
                print(f"     {v:8s} acc E* {ao:.2f} vs 随机 {ar:.2f} (Δ {ao-ar:+.2f}) | VEG {go-gr:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepeyes'); ap.add_argument('--certifier', default='qwen35')
    ap.add_argument('--device', type=int, default=0); ap.add_argument('--out', default=OUT)
    ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--prefix-model', default=None, help='第一轮前缀来自哪个模型的 fork 记录 (默认 = --model)')
    ap.add_argument('--max-new-tokens', type=int, default=1024)
    a = ap.parse_args()
    pm = a.prefix_model or a.model
    tag = f'{a.model}@{a.certifier}' + ('' if pm == a.model else f'|prefix={pm}')
    if a.summarize:
        summarize(a.out, tag); return
    recs = [r for r in D.read_records(FK.OUT, stage='fork', model=f'{pm}@{a.certifier}') if r.get('status') == 'forked' and r.get('estar')]
    items = {x['uuid']: x for x in I.load_items()}
    print(f'去结论分叉单元: {len(recs)} 条 ({sum(r["group"]=="required" for r in recs)} required)')
    bk = B.Backend(a.model, device=a.device)

    def one(r):
        it = items[r['uuid']]
        o = run_item(bk, it, r, mnt=a.max_new_tokens); o.update(group=r['group'], domain=r.get('domain'), name=r.get('name'))
        return o
    D.run([((r['uuid'], tag, 'decommit'), r) for r in recs], one, a.out, desc=f'decommit/{tag}', flush_every=5)
    summarize(a.out, tag)


if __name__ == '__main__':
    main()
