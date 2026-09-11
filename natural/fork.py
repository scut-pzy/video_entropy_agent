"""真实轨迹在首次工具返回处的分叉实验 (回应 review 第 2/4/5 条).

流程 (每条题):
  1. 让模型自己跑第一轮: think + <tool_call>; 执行工具得到真实返回 (real).
  2. 固定同一前缀 (system / 帧 / 问题 / 第一轮全文含调用参数), 只换"工具返回":
       real    真实返回的帧
       random  同形状的随机返回 (seek→随机同长时间窗的密采; zoom→随机帧上随机位置的同尺寸裁剪)  —— Illusion 的 random-crop 口径
       gray    与真实返回同尺寸的灰图
       oracle  在同一个调用点塞进认证过的 E* 窗密采 6 帧 (它自己没找到的那段证据)  —— Illusion 没有 E*, 做不了这支
     四支各自续生成到作答 (后续若再调工具, 真实执行).
  3. 每支记录: 最终答案/对错; 步级概率差 (固定前缀 + "<answer>" 读字母 logit, Illusion 的 VEG 口径, 含调用前基线);
     返回后**推理段**的 token 熵 (剔除 <tool_call>…</tool_call> 与 <answer>…</answer> 内的 token; review 指出原 ent_post 混入了参数文本).
  指标:
     VEG  = (gap_real − gap_pre) − (gap_random − gap_pre) = gap_real − gap_random
     EER  = H_post(random) − H_post(real)     熵响应: 真返回是否比随机返回更多地"解决"了不确定性
     flip = 三支答案是否一致
用法:  CUDA_VISIBLE_DEVICES=1 python fork.py --model deepeyes --certifier qwen35 [--n-control 40]
"""
import argparse, json, os, random, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
import items as I, frames as F, belief as B, driver as D, pipeline as P
import video_imcot as vi
from prompts_zh import SYS_ZH, INSTR_ZH, CONT_ZH, PRE_ANSWER_RE
from run_audit import select

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'fork_v2.jsonl')
LETTERS = 'ABCD'


def _think_entropy(text, ent, toks):
    """返回推理段 (剔除 <tool_call>…</tool_call> / <answer>…</answer> 内 token) 的平均熵与 token 数.
    按 token 拼接的字符偏移定位标签区间."""
    if not ent:
        return None, 0
    pos, s = [], ''
    for t in toks:
        pos.append(len(s)); s += t
    spans = [(m.start(), m.end()) for m in re.finditer(r'<tool_call>.*?(?:</tool_call>|$)', s, re.S)]
    spans += [(m.start(), m.end()) for m in re.finditer(r'<answer>.*?(?:</answer>|$)', s, re.S)]
    keep = [e for p, e in zip(pos, ent) if not any(a <= p < b for a, b in spans)]
    return (round(float(np.mean(keep)), 4) if keep else None), len(keep)


def _probe_gap(bk, messages, gt_idx):
    """固定前缀 + '<answer>' 读下一 token 的字母分布 → (p_gt, gap=p_gt − max_other, p[4])."""
    import torch
    kw = {'enable_thinking': False} if getattr(bk, 'thinking_template', False) else {}   # Qwen3.5: 关掉思考再读答案字母
    text = bk.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw) + '<answer>'
    images = [c['image'] for m in messages if isinstance(m['content'], list) for c in m['content'] if c.get('type') == 'image']
    inputs = bk.processor(text=[text], images=images or None, padding=True, return_tensors='pt').to(bk.model.device)
    with torch.inference_mode():
        lp = torch.log_softmax(bk.model(**inputs).logits[0, -1].float(), -1)
    z = np.array([torch.logsumexp(lp[bk.letter_ids[c]], 0).item() for c in LETTERS]); z = np.exp(z - z.max()); z /= z.sum()
    other = max(v for i, v in enumerate(z) if i != gt_idx)
    return round(float(z[gt_idx]), 4), round(float(z[gt_idx] - other), 4), [round(float(v), 4) for v in z]


def _random_return(sess, rec, rng):
    """与真实返回同形状的随机返回, 返回新帧索引列表 (不含真实证据的位置)."""
    args = rec['args']
    if rec['name'] == 'video_seek_tool':
        t0, t1 = float(args['start_time']), float(args['end_time']); w = max(0.05, t1 - t0)
        cands = [s for s in np.arange(0, max(0.0, sess.duration - w) + 1e-6, max(0.1, w / 2)) if s + w <= sess.duration and (s + w <= t0 or s >= t1)]
        s = float(rng.choice(cands)) if cands else float(rng.uniform(0, max(0.0, sess.duration - w)))
        return sess.tool_seek(dict(start_time=s, end_time=s + w, num_frames=args.get('num_frames', 6)))
    # zoom: 随机一帧 (粗采帧) 上的随机同尺寸框
    l, t, r, b = [float(v) for v in args['bbox_2d']]; w, h = r - l, b - t
    fi = int(rng.choice([i for i, f in enumerate(sess.frames) if f['kind'] == 'coarse']))
    disp = sess.frames[fi]['pil_disp']
    nl = rng.uniform(0, max(0.0, disp.width - w)); nt = rng.uniform(0, max(0.0, disp.height - h))
    return sess.tool_zoom(dict(frame_index=fi, bbox_2d=[nl, nt, nl + w, nt + h], label=args.get('label', '')))


def _gray_like(sess, idx_list):
    from PIL import Image
    out = []
    for i in idx_list:
        f = sess.frames[i]
        j = sess._register(f['t'], Image.new('RGB', f['pil_orig'].size, (128, 128, 128)), f['kind'], sess.detail_pixels, note=f['note'])
        sess.frames[j]['pil_disp'] = Image.new('RGB', f['pil_disp'].size, (128, 128, 128))
        out.append(j)
    return out


def _continue(bk, messages, sess, gt_letter, max_turns, mnt, lang='zh'):
    """从给定 messages 续生成直到作答 (后续工具真实执行). 返回 dict(pred, correct, text, ent, toks, n_extra_calls)."""
    text_all, ent_all, toks_all, pred, n_calls = '', [], [], None, 0
    for turn in range(max_turns):
        resp, ent, toks = P._gen(bk, messages, mnt)
        text_all += resp; ent_all += ent; toks_all += toks
        messages.append({'role': 'assistant', 'content': resp})
        if '<answer>' in resp:
            pred = vi._extract_letter(resp); break
        calls = re.findall(r'<tool_call>(.*?)</tool_call>', resp, re.S)
        if not calls:
            pred = vi._extract_letter(resp); break
        obs = []
        for c in calls[:3]:
            try:
                call = json.loads(c.strip()); a = call.get('arguments', {})
                new = sess.tool_zoom(a) if call['name'] == 'image_zoom_in_tool' else sess.tool_seek(a)
                obs += P._labeled(sess, new, lang); n_calls += 1
            except Exception as e:
                obs.append({'type': 'text', 'text': f'Error: {e}'})
        obs.append({'type': 'text', 'text': CONT_ZH}); messages.append({'role': 'user', 'content': obs})
    if pred is None:
        pred = vi._extract_letter(text_all)
    h, n = _think_entropy(text_all, ent_all, toks_all)
    return dict(pred=pred, correct=(pred == gt_letter), text=text_all, n_tokens=len(ent_all),
                ent_think=h, n_think_tokens=n, n_extra_calls=n_calls)


def fork_item(bk, it, estar, seed=0, max_turns=4, mnt=1024, lang='zh'):
    rng = random.Random(seed + int(it['uuid'][:6], 16))
    sess = vi.VideoSession(it['video'], n_coarse=P.BUDGET_K); coarse = list(range(len(sess.frames)))
    header = f'以下是从一段 {sess.duration:.1f} 秒的视频中均匀抽取的 {len(coarse)} 帧画面，每帧都标注了序号和时间戳。\n'
    q = it['stem'] + '\n' + '\n'.join(f'{c}. {o}' for c, o in zip(LETTERS, it['options']))
    base = [{'role': 'system', 'content': SYS_ZH},
            {'role': 'user', 'content': [{'type': 'text', 'text': header}] + P._labeled(sess, coarse, lang)
             + [{'type': 'text', 'text': '问题：' + q + INSTR_ZH}]}]
    gt = it['answer_letter']; gi = it['answer_idx']
    # 调用前基线 (固定 system+帧+问题, 直接读 <answer>)
    p_pre, gap_pre, _ = _probe_gap(bk, base, gi)
    # 第一轮
    t1, ent1, toks1 = P._gen(bk, base, mnt)
    calls = re.findall(r'<tool_call>(.*?)</tool_call>', t1, re.S)
    rec = dict(gt=gt, estar=estar, p_pre=p_pre, gap_pre=gap_pre, turn1=t1, turn1_n_tokens=len(ent1),
               turn1_ent_think=_think_entropy(t1, ent1, toks1)[0],
               pre_answer=bool(re.search(PRE_ANSWER_RE, t1.split('</think>')[0])))
    first = None
    for c in calls[:3]:
        try:
            call = json.loads(c.strip()); a = call.get('arguments', {})
            if call['name'] == 'image_zoom_in_tool':
                new = sess.tool_zoom(a); fi = int(a['frame_index']); tt = sess.frames[fi]['t']; win = [tt, tt]
            elif call['name'] == 'video_seek_tool':
                new = sess.tool_seek(a); win = [float(a['start_time']), float(a['end_time'])]
            else:
                continue
            first = dict(name=call['name'], args=a, window=win, idx=new); break
        except Exception as e:
            rec.setdefault('call_errors', []).append(str(e)[:120])
    if first is None:
        rec.update(status='no_call', pred=vi._extract_letter(t1), correct=(vi._extract_letter(t1) == gt))
        return rec
    hit, iou = P.temporal_hit(first['window'], estar) if estar else (None, None)
    rec.update(status='forked', call=dict(name=first['name'], args=first['args'], window=first['window'], hit=hit))
    # 三支
    branches = {'real': first['idx'], 'random': _random_return(sess, first, rng), 'gray': _gray_like(sess, first['idx'])}
    if estar:
        branches['oracle'] = sess.tool_seek(dict(start_time=float(estar[0]), end_time=float(estar[1]), num_frames=6))
    out = {}
    for name, idx in branches.items():
        obs = P._labeled(sess, idx, lang) + [{'type': 'text', 'text': CONT_ZH}]
        msgs = base + [{'role': 'assistant', 'content': t1}, {'role': 'user', 'content': obs}]
        p, gap, pv = _probe_gap(bk, msgs, gi)                           # 步级: 返回后立刻读 <answer>
        cont = _continue(bk, [m for m in msgs], sess, gt, max_turns - 1, mnt, lang)   # 轨迹级: 续生成
        cont.update(p_probe=p, gap_probe=gap, p_vec=pv, frames_t=[round(sess.frames[i]['t'], 2) for i in idx])
        out[name] = cont
    rec['branches'] = out
    g = {k: v['gap_probe'] for k, v in out.items()}
    rec['VEG_random'] = round(g['real'] - g['random'], 4); rec['VEG_gray'] = round(g['real'] - g['gray'], 4)
    h = {k: v['ent_think'] for k, v in out.items()}
    rec['EER_random'] = round(h['random'] - h['real'], 4) if (h['random'] is not None and h['real'] is not None) else None
    rec['EER_gray'] = round(h['gray'] - h['real'], 4) if (h['gray'] is not None and h['real'] is not None) else None
    if 'oracle' in out:
        rec['VEG_oracle'] = round(g['oracle'] - g['random'], 4)          # 真证据到手 vs 随机: 步级
        rec['EER_oracle'] = round(h['random'] - h['oracle'], 4) if (h['random'] is not None and h['oracle'] is not None) else None
        rec['flip_oracle'] = out['oracle']['pred'] != out['random']['pred']
        rec['oracle_correct'] = out['oracle']['correct']
    rec['flip_random'] = out['real']['pred'] != out['random']['pred']
    rec['flip_gray'] = out['real']['pred'] != out['gray']['pred']
    rec['pred'] = out['real']['pred']; rec['correct'] = out['real']['correct']
    return rec


def summarize(path=OUT, model=None):
    recs = D.read_records(path, stage='fork', model=model)
    if not recs:
        print('还没有结果'); return
    for grp in ('required', 'control'):
        g = [r for r in recs if r.get('group') == grp]
        if not g:
            continue
        fk = [r for r in g if r.get('status') == 'forked']
        print(f'\n[{grp}] n={len(g)}  调了工具并成功分叉 {len(fk)}  未调/失败 {len(g)-len(fk)}')
        if not fk:
            continue
        m = lambda k: float(np.mean([r[k] for r in fk if r.get(k) is not None]))
        acc = lambda b: float(np.mean([r['branches'][b]['correct'] for r in fk]))
        gp = lambda b: float(np.mean([r['branches'][b]['gap_probe'] for r in fk]))
        ht = lambda b: float(np.mean([r['branches'][b]['ent_think'] for r in fk if r['branches'][b]['ent_think'] is not None]))
        print(f"  准确率:   real {acc('real'):.2f} | random {acc('random'):.2f} | gray {acc('gray'):.2f}   (调用前基线 p_gt {m('p_pre'):.2f})")
        print(f"  步级概率差 gap: pre {m('gap_pre'):+.3f} → real {gp('real'):+.3f} | random {gp('random'):+.3f} | gray {gp('gray'):+.3f}")
        print(f"  VEG(real−random) = {m('VEG_random'):+.3f}   VEG(real−gray) = {m('VEG_gray'):+.3f}")
        print(f"  返回后推理段熵 H: real {ht('real'):.3f} | random {ht('random'):.3f} | gray {ht('gray'):.3f}   "
              f"EER(random−real) = {m('EER_random'):+.3f}   EER(gray−real) = {m('EER_gray'):+.3f}")
        print(f"  答案翻转率: vs random {np.mean([r['flip_random'] for r in fk]):.0%} | vs gray {np.mean([r['flip_gray'] for r in fk]):.0%}   "
              f"调用前已下结论 {np.mean([r['pre_answer'] for r in fk]):.0%}   命中E* {np.mean([bool(r['call'].get('hit')) for r in fk if r['call'].get('hit') is not None] or [0]):.0%}")
        # 逐条: 证据依赖 (VEG>0.1 或 翻转) 与 熵响应 的关系
        dep = [r for r in fk if r['VEG_random'] > 0.1 or r['flip_random']]
        print(f"  证据依赖 (VEG>0.1 或翻转) 的轨迹: {len(dep)}/{len(fk)}"
              + (f"; 它们的 EER(random−real) 均值 {np.mean([r['EER_random'] for r in dep if r['EER_random'] is not None]):+.3f}" if dep else ''))
        v = [(r['VEG_random'], r['EER_random']) for r in fk if r['EER_random'] is not None]
        if len(v) > 3:
            a, b = zip(*v); print(f"  corr(VEG, EER) = {np.corrcoef(a, b)[0, 1]:+.2f}  (n={len(v)})")
        oc = [r for r in fk if 'oracle' in r['branches']]
        if oc:
            print(f"  [oracle 支: 在它自己的调用点塞进 E*] n={len(oc)}  准确率 {np.mean([r['branches']['oracle']['correct'] for r in oc]):.2f} "
                  f"| 概率差 {np.mean([r['branches']['oracle']['gap_probe'] for r in oc]):+.3f} "
                  f"| VEG(oracle−random) = {np.mean([r['VEG_oracle'] for r in oc]):+.3f} "
                  f"| 熵 H {np.mean([r['branches']['oracle']['ent_think'] for r in oc if r['branches']['oracle']['ent_think'] is not None]):.3f} "
                  f"| EER(random−oracle) = {np.mean([r['EER_oracle'] for r in oc if r['EER_oracle'] is not None]):+.3f} "
                  f"| 答案翻转 vs random {np.mean([r['flip_oracle'] for r in oc]):.0%}")
            dep = [r for r in oc if r['VEG_oracle'] > 0.1 or r['flip_oracle']]
            print(f"    真证据到手后证据依赖的轨迹: {len(dep)}/{len(oc)}")
            v2 = [(r['VEG_oracle'], r['EER_oracle']) for r in oc if r['EER_oracle'] is not None]
            if len(v2) > 3:
                a, b = zip(*v2); print(f"    corr(VEG_oracle, EER_oracle) = {np.corrcoef(a, b)[0, 1]:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepeyes'); ap.add_argument('--certifier', default='qwen35')
    ap.add_argument('--device', type=int, default=0); ap.add_argument('--n-control', type=int, default=40)
    ap.add_argument('--max-turns', type=int, default=4); ap.add_argument('--max-new-tokens', type=int, default=1024)
    ap.add_argument('--out', default=OUT); ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    tag = f'{a.model}@{a.certifier}'
    if a.summarize:
        summarize(a.out, tag); return
    units = select(os.path.join(HERE, 'out', 'screen.jsonl'), a.certifier, a.n_control)
    print(f'分叉单元: {sum(1 for _, g, _, _ in units if g == "required")} required + {sum(1 for _, g, _, _ in units if g == "control")} control')
    bk = B.Backend(a.model, device=a.device)

    def one(payload):
        it, group, estar, _ = payload
        F.prefetch([it], ks=(8,), verbose=False)
        r = fork_item(bk, it, estar, max_turns=a.max_turns, mnt=a.max_new_tokens)
        r.update(group=group, domain=it['domain'], name=it['name'])
        return r
    D.run([((it['uuid'], tag, 'fork'), (it, g, e, d)) for it, g, e, d in units], one, a.out, desc=f'fork/{tag}', flush_every=5)
    summarize(a.out, tag)


if __name__ == '__main__':
    main()
