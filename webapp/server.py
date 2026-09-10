"""可视化网站: 选一条 VideoNet 题 → 看模型输入的帧 → 改问题/提示词 → 点推理 → 右侧流式显示
think(蓝) / 工具调用(橙, 含返回帧) / 答案(绿=对, 红=错), 下方 entropy 曲线并标出工具调用起点.

启动:  cd webapp && CUDA_VISIBLE_DEVICES=1 python server.py --port 4175 [--model deepeyes]
依赖:  fastapi / uvicorn / sse_starlette (ms_swift env 里都有); 模型与帧缓存路径沿用 natural/ 里的配置.
"""
import argparse, asyncio, base64, io, json, os, re, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'natural'))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse, JSONResponse
from transformers import LogitsProcessor, LogitsProcessorList, TextIteratorStreamer

import items as I, frames as F, belief as B, driver as D, pipeline as P
import video_imcot as vi
from prompts_zh import SYS_ZH, INSTR_ZH, CONT_ZH, PRE_ANSWER_RE

app = FastAPI()
STATE = dict(backend=None, model_name=None, loading=False, lock=threading.Lock())
ITEMS = {}
LABELS = {}          # uuid -> dict(label, why, estar, p_k8, p_estar)  (来自 natural/out/screen.jsonl, 认证者 qwen35)


# ───────────────────────── 数据 ─────────────────────────

def load_data():
    global ITEMS, LABELS
    its = I.load_items()
    ITEMS = {x['uuid']: x for x in its}
    sp = os.path.join(HERE, '..', 'natural', 'out', 'screen.jsonl')
    if os.path.exists(sp):
        scr = {r['uuid']: r for r in D.read_records(sp, stage='screen', model='qwen35')}
        loc = {r['uuid']: r for r in D.read_records(sp, stage='localize', model='qwen35')}
        for u, s in scr.items():
            lab, why = s['label'], s['why']; est = None
            if lab == 'candidate' and u in loc:
                lab, why = P.certify(s['screen'], loc[u]['localize']); est = loc[u]['localize']['estar']
            LABELS[u] = dict(label=lab, why=why, p_k8=s['screen']['k8']['p_gt'],
                             estar=[est['t0'], est['t1']] if est else None,
                             p_estar=est['p_gt'] if est else None)
    print(f'[data] {len(ITEMS)} items, {len(LABELS)} with qwen35 labels')


def ensure_backend(name):
    if STATE['model_name'] == name and STATE['backend'] is not None:
        return STATE['backend']
    STATE['loading'] = True
    try:
        if STATE['backend'] is not None:
            del STATE['backend']; STATE['backend'] = None; torch.cuda.empty_cache()
        bk = B.Backend(name, device=0)
        STATE['backend'] = bk; STATE['model_name'] = name
        return bk
    finally:
        STATE['loading'] = False


def pil_b64(pil, q=85):
    buf = io.BytesIO(); pil.save(buf, format='JPEG', quality=q)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


# ───────────────────────── 流式生成 (带逐 token 熵) ─────────────────────────

class EntropyTap(LogitsProcessor):
    def __init__(self):
        self.ent = []
    def __call__(self, input_ids, scores):
        lp = torch.log_softmax(scores[0].float(), -1)
        self.ent.append(float(-(lp.exp() * lp).sum()))
        return scores


def stream_generate(bk, messages, max_new_tokens=512):
    """逐块 yield (text_delta, entropies_so_far). 在 </tool_call> 处截停."""
    tok = bk.processor.tokenizer
    text = bk.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [c['image'] for m in messages if isinstance(m['content'], list)
              for c in m['content'] if c.get('type') == 'image']
    inputs = bk.processor(text=[text], images=images or None, padding=True, return_tensors='pt').to(bk.model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    tap = EntropyTap()
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer,
              logits_processor=LogitsProcessorList([tap]),
              stop_strings=['</tool_call>'], tokenizer=tok)
    err = {}
    def run():
        try:
            with torch.inference_mode():
                bk.model.generate(**inputs, **kw)
        except Exception as e:
            err['e'] = e
            streamer.end()
    th = threading.Thread(target=run, daemon=True); th.start()
    for delta in streamer:
        yield delta, list(tap.ent)
    th.join()
    if 'e' in err:
        raise err['e']
    yield '', list(tap.ent)


# ───────────────────────── 推理主循环 (SSE) ─────────────────────────

def sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def infer_events(req):
    it = ITEMS[req['uuid']]
    lang = 'zh'
    bk = ensure_backend(req.get('model', 'deepeyes'))
    k = int(req.get('k', 8)); max_turns = int(req.get('max_turns', 4)); mnt = int(req.get('max_new_tokens', 512))
    sess = vi.VideoSession(it['video'], n_coarse=k)
    coarse = list(range(len(sess.frames)))
    yield sse(dict(type='frames', frames=[dict(idx=i, t=round(sess.frames[i]['t'], 2), img=pil_b64(sess.frames[i]['pil_disp']))
                                        for i in coarse]))
    header = f'以下是从一段 {sess.duration:.1f} 秒的视频中均匀抽取的 {len(coarse)} 帧画面，每帧都标注了序号和时间戳。\n'
    q = req['question']
    system = req.get('system_prompt') or SYS_ZH
    instr = req.get('instruction') or INSTR_ZH
    cont = req.get('cont_prompt') or CONT_ZH
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': [{'type': 'text', 'text': header}] + P._labeled(sess, coarse, lang)
                 + [{'type': 'text', 'text': '问题：' + q + instr}]}]
    lab = LABELS.get(it['uuid'], {}); estar = lab.get('estar')
    all_ent, all_text, markers, tool_recs = [], '', dict(tool_call=[], answer=[], think_end=[], turn_starts=[]), []
    pred = None
    for turn in range(max_turns):
        markers['turn_starts'].append(len(all_ent))
        yield sse(dict(type='turn_start', turn=turn + 1))
        base_n = len(all_ent); turn_text = ''
        seen = set()
        for delta, ent in stream_generate(bk, messages, mnt):
            turn_text += delta
            all_ent = all_ent[:base_n] + ent
            for tag in ('<tool_call>', '</think>', '<answer>'):
                key = {'<tool_call>': 'tool_call', '</think>': 'think_end', '<answer>': 'answer'}[tag]
                if tag not in seen and tag in turn_text:
                    seen.add(tag); markers[key].append(len(all_ent))
            if delta:
                yield sse(dict(type='token', text=delta, n_tokens=len(all_ent), entropy=ent[-8:]))
        all_text += turn_text
        yield sse(dict(type='entropy', entropy=all_ent, markers=markers))
        messages.append({'role': 'assistant', 'content': turn_text})
        if '<answer>' in turn_text:
            pred = vi._extract_letter(turn_text); break
        calls = re.findall(r'<tool_call>(.*?)</tool_call>', turn_text, re.S)
        if not calls:
            pred = vi._extract_letter(turn_text); break
        obs = []
        for c in calls[:3]:
            rec = dict(turn=turn, ok=False, err=None, name=None, window=None, args=None)
            try:
                call = json.loads(c.strip()); rec['name'] = call.get('name'); args = call.get('arguments', {}); rec['args'] = args
                if call['name'] == 'image_zoom_in_tool':
                    new = sess.tool_zoom(args); fi = int(args['frame_index'])
                    t = sess.frames[fi]['t'] if 0 <= fi < len(sess.frames) else None
                    rec['window'] = [t, t] if t is not None else None
                elif call['name'] == 'video_seek_tool':
                    new = sess.tool_seek(args); rec['window'] = [float(args['start_time']), float(args['end_time'])]
                else:
                    raise ValueError(f"未知工具 {call.get('name')}")
                rec['ok'] = True
                obs += P._labeled(sess, new, lang)
                hit, iou = P.temporal_hit(rec['window'], estar) if estar else (None, None)
                yield sse(dict(type='tool_call', name=rec['name'], args=args, ok=True, window=rec['window'],
                               hit=hit, frames=[dict(idx=i, t=round(sess.frames[i]['t'], 2),
                                                     label=sess.frames[i]['note'], img=pil_b64(sess.frames[i]['pil_disp']))
                                                for i in new]))
            except Exception as e:
                rec['err'] = str(e)[:200]
                obs.append({'type': 'text', 'text': f'Error: {e}'})
                yield sse(dict(type='tool_call', name=rec['name'], args=rec.get('args'), ok=False, err=rec['err']))
            tool_recs.append(rec)
        obs.append({'type': 'text', 'text': cont})
        messages.append({'role': 'user', 'content': obs})
    if pred is None:
        pred = vi._extract_letter(all_text)
    gt = it['answer_letter']
    n = len(all_ent)
    first_tc = markers['tool_call'][0] if markers['tool_call'] else None
    e = np.array(all_ent) if all_ent else np.zeros(1)
    pre_ans = bool(re.search(PRE_ANSWER_RE, all_text.split('</think>')[0]))
    hits = [P.temporal_hit(r['window'], estar)[0] for r in tool_recs if r['ok'] and r['window'] and estar]
    yield sse(dict(type='answer', pred=pred, gt=gt, correct=(pred == gt)))
    yield sse(dict(type='done', summary=dict(
        n_tokens=n, tool_call_pct=[round(100 * i / max(1, n - 1), 1) for i in markers['tool_call']],
        answer_pct=[round(100 * i / max(1, n - 1), 1) for i in markers['answer']],
        ent_pre=round(float(e[:first_tc].mean()), 3) if first_tc else None,
        ent_post=round(float(e[first_tc:].mean()), 3) if first_tc else None,
        mean_entropy=round(float(e.mean()), 3), pre_answer=pre_ans,
        temporal_hit=(any(hits) if hits else None), estar=estar, label=lab.get('label'))))


# ───────────────────────── 路由 ─────────────────────────

@app.get('/', response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, 'static', 'index.html'), encoding='utf-8').read()


@app.get('/api/status')
def status():
    return dict(model=STATE['model_name'], loading=STATE['loading'], n_items=len(ITEMS),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


@app.get('/api/items')
def list_items():
    out = []
    for u, it in ITEMS.items():
        lab = LABELS.get(u, {})
        out.append(dict(uuid=u, domain=it['domain'], name=it['name'], duration=round(it['duration'] or 0, 1),
                        label=lab.get('label'), estar=lab.get('estar'), p_k8=lab.get('p_k8')))
    order = {'evidence_required': 0, 'borderline': 1, 'already_solvable': 2, 'unsolvable': 3, 'text_shortcut': 4, None: 5}
    out.sort(key=lambda r: (order.get(r['label'], 5), r['domain'] or '', r['name'] or ''))
    return out


@app.get('/api/item/{uuid}')
def get_item(uuid: str):
    it = ITEMS[uuid]; lab = LABELS.get(uuid, {})
    q = it['stem'] + '\n' + '\n'.join(f'{L}. {o}' for L, o in zip('ABCD', it['options']))
    return dict(uuid=uuid, domain=it['domain'], name=it['name'], definition=it['definition'],
                duration=it['duration'], n_frames=it['n_frames'], fps=it['fps'], width=it['width'], height=it['height'],
                question=q, options=it['options'], answer_letter=it['answer_letter'], answer=it['answer'],
                label=lab.get('label'), why=lab.get('why'), estar=lab.get('estar'), p_k8=lab.get('p_k8'), p_estar=lab.get('p_estar'),
                coarse_idx=F.uniform_indices(it['n_frames'], 8))


@app.get('/api/frame/{uuid}/{idx}')
def get_frame(uuid: str, idx: int, cap: str = 'coarse'):
    it = ITEMS[uuid]
    pil = F.load(uuid, [idx], F.COARSE_PIXELS if cap == 'coarse' else F.DETAIL_PIXELS, it['video'])[0]
    buf = io.BytesIO(); pil.save(buf, format='JPEG', quality=85)
    return Response(buf.getvalue(), media_type='image/jpeg')


@app.get('/api/prompts')
def prompts():
    return dict(system_prompt=SYS_ZH, instruction=INSTR_ZH, cont_prompt=CONT_ZH)


@app.post('/api/infer')
async def infer(req: dict):
    if not STATE['lock'].acquire(blocking=False):
        return JSONResponse({'error': '有一条推理正在跑，等它结束'}, status_code=429)
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def worker():
        try:
            for ev in infer_events(req):
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as e:
            import traceback
            loop.call_soon_threadsafe(queue.put_nowait, sse(dict(type='error', error=f'{type(e).__name__}: {e}',
                                                                 tb=traceback.format_exc()[-800:])))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            STATE['lock'].release()

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        yield sse(dict(type='status', text='开始推理' if STATE['backend'] else '正在加载模型（约 1 分钟）…'))
        while True:
            ev = await queue.get()
            if ev is None:
                break
            yield ev
    return StreamingResponse(gen(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=4175); ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--model', default='deepeyes'); ap.add_argument('--no-preload', action='store_true')
    a = ap.parse_args()
    load_data()
    if not a.no_preload:
        threading.Thread(target=ensure_backend, args=(a.model,), daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level='info')
