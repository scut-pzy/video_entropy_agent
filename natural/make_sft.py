"""从认证过的题 + E* 合成"理想取证轨迹", 作 SFT 种子数据 (回应用户: 收集这类数据去 SFT).

每条样本 = DeepEyes 工具格式的两轮对话:
  轮1 assistant: <think>粗帧能看到什么 / 为什么还定不了 / 要回看哪一段 (由 Qwen3.5 只看粗帧生成, 不给答案)</think>
                 <tool_call>{"name":"video_seek_tool","arguments":{"start_time":E*_t0,"end_time":E*_t1,"num_frames":6}}</tool_call>
  user:          E* 窗密采 6 帧 (工具返回) + 续轮提示
  轮2 assistant: <think>回看到了什么 / 如何区分选项 (由 Qwen3.5 看粗帧+E*帧生成)</think><answer>正确字母</answer>
过滤: Qwen3.5 拿到 E* 帧后自己作答也对 (certify 已保证), 且第一轮 think 不含答案字母/选项名.
输出: out/sft_pilot.jsonl (帧以 frames_cache 路径引用, 不内嵌图片)
用法: CUDA_VISIBLE_DEVICES=1 python make_sft.py --certifier qwen35 --writer qwen35 [--include-borderline]
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import items as I, frames as F, belief as B, driver as D, pipeline as P
from prompts_zh import SYS_ZH, INSTR_ZH, CONT_ZH

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'sft_pilot.jsonl')
N_EVID = 6

T1 = ('你是在写一条"主动取证"的示范推理。只根据上面这 {k} 帧粗采样画面，用中文写 2–4 句话：'
      '(1) 画面里能看到什么；(2) 为什么仅凭这些帧还不能在四个选项里确定答案——具体缺哪种细节或哪个时刻；'
      '(3) 你打算密集回看 {t0:.1f}–{t1:.1f} 秒这一段，说明为什么是这一段。'
      '**不要**给出或暗示任何选项字母或选项名。只输出这段话，不要标题。')
T2 = ('上面最后 {n} 帧是你回看 {t0:.1f}–{t1:.1f} 秒后拿到的画面。用中文写 2–3 句话：'
      '(1) 回看到了什么关键动作或细节；(2) 它为什么能把选项 {gt}（{gt_name}）和其它选项区分开。'
      '只输出这段话，不要给出字母，不要标题。')


def build(bk, it, estar, lang='zh'):
    gt = it['answer_letter']; gi = it['answer_idx']
    coarse_idx = F.uniform_indices(it['n_frames'], P.BUDGET_K)
    coarse = F.load(it['uuid'], coarse_idx, F.COARSE_PIXELS, it['video'])
    prefix = B.frames_prefix(coarse, coarse_idx, it['fps'], it['duration'], lang=lang)
    q = it['stem'] + '\n' + '\n'.join(f'{c}. {o}' for c, o in zip('ABCD', it['options']))
    t0, t1 = estar
    # 轮1 think (只看粗帧, 不许给答案)
    think1 = bk.generate(prefix + [{'type': 'text', 'text': '问题：' + q + '\n\n' + T1.format(k=len(coarse), t0=t0, t1=t1)}],
                         max_new_tokens=300).strip()
    bad = re.search(r'\b[ABCD]\b', think1) or any(o.lower() in think1.lower() for o in it['options'])
    # 轮2 think (粗帧 + E* 帧)
    e_idx = F.window_indices(it['n_frames'], t0, t1, N_EVID, it['fps'])
    e_pils = F.load(it['uuid'], e_idx, F.DETAIL_PIXELS, it['video'])
    ev = B.evidence_prefix(e_pils, e_idx, it['fps'], lang=lang, start_at=P.BUDGET_K)
    think2 = bk.generate(prefix + ev + [{'type': 'text', 'text': '问题：' + q + '\n\n' + T2.format(n=len(e_pils), t0=t0, t1=t1, gt=gt, gt_name=it['answer'])}],
                         max_new_tokens=300).strip()
    # Qwen3.5 自己在 E* 下的信念 (再验一遍)
    b = bk.belief(prefix + ev, it['stem'], it['options'], n_perm=4, lang=lang)
    call = json.dumps(dict(name='video_seek_tool', arguments=dict(start_time=round(t0, 2), end_time=round(t1, 2), num_frames=N_EVID)), ensure_ascii=False)
    header = f"以下是从一段 {it['duration']:.1f} 秒的视频中均匀抽取的 {len(coarse)} 帧画面，每帧都标注了序号和时间戳。\n"
    messages = [
        {'role': 'system', 'content': SYS_ZH},
        {'role': 'user', 'content': [{'type': 'text', 'text': header}]
         + [x for i, fi in enumerate(coarse_idx) for x in ({'type': 'text', 'text': f'[第{i}帧, t={fi/it["fps"]:.1f}s]'}, {'type': 'image', 'image': F._path(it['uuid'], fi), 'pixel_cap': F.COARSE_PIXELS})]
         + [{'type': 'text', 'text': '问题：' + q + INSTR_ZH}]},
        {'role': 'assistant', 'content': f'<think>{think1}</think>\n<tool_call>\n{call}\n</tool_call>'},
        {'role': 'user', 'content': [x for j, fi in enumerate(e_idx) for x in ({'type': 'text', 'text': f'[第{P.BUDGET_K + j}帧, t={fi/it["fps"]:.1f}s，密集回放]'}, {'type': 'image', 'image': F._path(it['uuid'], fi), 'pixel_cap': F.DETAIL_PIXELS})]
         + [{'type': 'text', 'text': CONT_ZH}]},
        {'role': 'assistant', 'content': f'<think>{think2}</think>\n<answer>{gt}</answer>'},
    ]
    return dict(messages=messages, think1=think1, think2=think2, leak_in_think1=bool(bad),
                p_gt_with_estar=b['p_gt'] if 'p_gt' in b else b['p'][gi], correct_with_estar=(b['pred_idx'] == gi),
                estar=[t0, t1], gt=gt, answer=it['answer'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--certifier', default='qwen35'); ap.add_argument('--writer', default='qwen35')
    ap.add_argument('--device', type=int, default=0); ap.add_argument('--include-borderline', action='store_true')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    sp = os.path.join(HERE, 'out', 'screen.jsonl')
    scr = {r['uuid']: r for r in D.read_records(sp, stage='screen', model=a.certifier)}
    loc = {r['uuid']: r for r in D.read_records(sp, stage='localize', model=a.certifier)}
    all_items = {x['uuid']: x for x in I.load_items()}
    units = []
    for u, s in scr.items():
        if s['label'] != 'candidate' or u not in loc:
            continue
        lab, _ = P.certify(s['screen'], loc[u]['localize'])
        if lab == 'evidence_required' or (a.include_borderline and lab == 'borderline'):
            e = loc[u]['localize']['estar']; units.append((all_items[u], lab, (e['t0'], e['t1'])))
    print(f'合成 {len(units)} 条 (certifier={a.certifier}, writer={a.writer})')
    bk = B.Backend(a.writer, device=a.device)

    def one(payload):
        it, lab, estar = payload
        F.prefetch([it], ks=(8,), verbose=False)
        r = build(bk, it, estar); r.update(label=lab, domain=it['domain'], name=it['name'])
        return r
    recs = D.run([((it['uuid'], a.writer, 'sft'), (it, lab, e)) for it, lab, e in units], one, a.out, desc='sft', flush_every=5)
    ok = [r for r in recs if r.get('stage') == 'sft' and not r.get('error')]
    keep = [r for r in ok if r['correct_with_estar'] and not r['leak_in_think1']]
    print(f'\n合成 {len(ok)} 条; 通过过滤 (E* 下答对 ∧ 轮1不泄露答案) {len(keep)} 条')
    for r in keep[:3]:
        print(f"\n--- {r['domain']} / {r['name']}  E*={r['estar']}  答案 {r['gt']} ({r['answer']})")
        print('轮1 think:', r['think1'][:300]); print('轮2 think:', r['think2'][:300])
    clean = os.path.join(HERE, 'out', 'sft_pilot_clean.jsonl')
    with open(clean, 'w') as f:
        for r in keep:
            f.write(json.dumps(dict(uuid=r['uuid'], domain=r['domain'], name=r['name'], estar=r['estar'], gt=r['gt'], messages=r['messages']), ensure_ascii=False) + '\n')
    print('->', clean)


if __name__ == '__main__':
    main()
