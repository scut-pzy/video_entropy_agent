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
LOAD = dict(target=None, phase='空闲', percent=0, log=[], error=None, t0=None)   # 加载进度, 给前端进度条+日志框
LOAD_LOCK = threading.Lock()


def _log(msg):
    LOAD['log'].append(f'[{time.strftime("%H:%M:%S")}] {msg}')
    LOAD['log'] = LOAD['log'][-60:]
    print('[load]', msg)


def _model_bytes(name):
    import glob
    d = B.MODELS[name]
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, '*.safetensors')))
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
    """加载/切换模型. 进度 = 本进程显存增长 / 权重文件总大小 (bf16 落盘即上卡, 近似准确)."""
    if STATE['model_name'] == name and STATE['backend'] is not None:
        return STATE['backend']
    if not LOAD_LOCK.acquire(blocking=False):
        # 别的线程正在加载: 等它
        while STATE['loading']:
            time.sleep(0.5)
        if STATE['model_name'] == name and STATE['backend'] is not None:
            return STATE['backend']
        raise RuntimeError('模型加载失败: ' + str(LOAD.get('error')))
    try:
        STATE['loading'] = True
        LOAD.update(target=name, phase='准备', percent=0, log=[], error=None, t0=time.time())
        if STATE['backend'] is not None:
            _log(f"卸载 {STATE['model_name']}，释放显存")
            LOAD['phase'] = '释放旧模型'
            del STATE['backend']; STATE['backend'] = None; STATE['model_name'] = None
            import gc; gc.collect(); torch.cuda.empty_cache()
        expected = max(1, _model_bytes(name)); base = torch.cuda.memory_allocated()
        _log(f'开始加载 {name}（权重 {expected/1e9:.1f} GB，路径 {B.MODELS[name]}）')
        LOAD['phase'] = '加载权重到 GPU'
        stop = threading.Event()

        def watch():
            last = -1
            while not stop.is_set():
                pct = min(99, int(100 * (torch.cuda.memory_allocated() - base) / expected))
                if pct != last:
                    LOAD['percent'] = pct
                    if pct // 10 != max(last, 0) // 10:
                        _log(f'权重加载 {pct}%（{(torch.cuda.memory_allocated()-base)/1e9:.1f} GB，{time.time()-LOAD["t0"]:.0f}s）')
                    last = pct
                time.sleep(0.3)
        th = threading.Thread(target=watch, daemon=True); th.start()
        try:
            bk = B.Backend(name, device=0)
        except Exception as e:
            LOAD.update(phase='失败', error=f'{type(e).__name__}: {e}')
            _log('加载失败: ' + LOAD['error']); raise
        finally:
            stop.set(); th.join(timeout=2)
        LOAD.update(phase='就绪', percent=100)
        _log(f'{name} 加载完成，用时 {time.time()-LOAD["t0"]:.0f}s，显存 {torch.cuda.memory_allocated()/1e9:.1f} GB'
             f'（thinking_template={bk.thinking_template}）')
        STATE['backend'] = bk; STATE['model_name'] = name
        return bk
    finally:
        STATE['loading'] = False
        LOAD_LOCK.release()


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
    bk = ensure_backend(req.get('model', 'qwen35'))
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
    pre_think = all_text.split('</think>')[0]
    pre_ans = bool(re.search(PRE_ANSWER_RE, pre_think))
    m_pre = re.findall(r'(?:答案是|正确答案是|应该选|应选|选项|是)\s*([ABCD])\b', pre_think)
    pre_letter = m_pre[-1] if m_pre else None
    if pre_letter is None:
        # 调用前的 think 里直接点了某个选项的名字 (如 "这看起来像 interception"), 也算已下结论
        low = pre_think.lower(); best = (-1, None)
        for i, o in enumerate(it['options']):
            j = low.rfind(o.lower())
            if j > best[0]:
                best = (j, 'ABCD'[i])
        if best[1] is not None:
            pre_letter = best[1]; pre_ans = True
    hits = [P.temporal_hit(r['window'], estar)[0] for r in tool_recs if r['ok'] and r['window'] and estar]
    called = any(r['ok'] for r in tool_recs)
    hit = any(hits) if hits else None
    correct = (pred == gt)
    # 熵: 首次工具调用前后
    ent_pre = float(e[:first_tc].mean()) if first_tc else None
    ent_post = float(e[first_tc:].mean()) if first_tc else None
    kk = max(1, int(round(0.2 * n))); thr = float(np.sort(e)[::-1][kk - 1]) if n else 0.0
    hi_idx = [i for i, v in enumerate(all_ent) if v >= thr and v > 0]
    n_hi_pre = sum(1 for i in hi_idx if first_tc is not None and i < first_tc)
    n_hi_post = len(hi_idx) - n_hi_pre if first_tc is not None else None
    # 诊断: 这次推理落在 idea 的哪一层失败 (taxonomy 见 docs/01_proposal.md §1)
    required = lab.get('label') == 'evidence_required'
    solvable = lab.get('label') == 'already_solvable'
    if not called:
        if required:
            code, title, color = 'overconfident_skip', 'Overconfident Skip：需要证据却没调工具，直接作答', 'red'
        elif solvable:
            code, title, color = 'no_tool_needed', '合理：这题粗看可解，不调工具没问题', 'green'
        else:
            code, title, color = 'no_call', '没调工具（这题未被认证为需要证据，无法判定对错）', 'gray'
    elif solvable:
        code, title, color = 'redundant_call', 'Redundant Call：粗看就能答，调用是多余的（仪式性）', 'orange'
    elif estar is None:
        code, title, color = 'uncertain', '调了工具，但这题没有认证过的 E*，无法判定是否找对', 'gray'
    elif not hit:
        code, title, color = 'mislocalized', 'Mislocalized Call：知道要看，但看错了地方（工具窗未落进 E*）', 'red'
    elif pre_ans and (pre_letter is None or pre_letter == pred):
        code, title, color = 'unused_evidence', 'Unused Evidence / 先答后看：找对了地方，但结论在调用前就定了', 'orange'
    elif correct:
        code, title, color = 'effective', '有效取证：找对了地方、看完才作答、答对了', 'green'
    else:
        code, title, color = 'hit_but_wrong', '找对了地方仍答错：证据没被用上（Utilization 失败或能力天花板）', 'red'
    details = []
    details.append(f"这题：{'需要证据' if required else ('粗看可解' if solvable else (lab.get('label') or '未认证'))}"
                   + (f"，E* = [{estar[0]:.2f}, {estar[1]:.2f}] s" if estar else ''))
    details.append(f"调工具：{'是' if called else '否'}" + (f"，时间窗{'命中' if hit else '未命中'} E*" if (called and hit is not None) else ''))
    details.append(f"调用前是否已下结论：{'是' if pre_ans else '否'}" + (f"（写的是 {pre_letter}，最终答 {pred}）" if pre_letter else ''))
    if ent_pre is not None:
        drop = (1 - ent_post / ent_pre) * 100 if ent_pre > 0 else 0
        details.append(f"熵：调用前平均 {ent_pre:.2f} → 调用后 {ent_post:.2f}（降 {drop:.0f}%）；top-20% 高熵 token 调用前 {n_hi_pre} 个、调用后 {n_hi_post} 个"
                       + ("——不确定性在证据到来前就释放完了" if (n_hi_pre >= 2 * max(1, n_hi_post or 0)) else ''))
    details.append(f"答案：{pred} {'✓' if correct else '✗'}（正确 {gt}）")
    yield sse(dict(type='answer', pred=pred, gt=gt, correct=correct))
    yield sse(dict(type='done', summary=dict(
        n_tokens=n, tool_call_pct=[round(100 * i / max(1, n - 1), 1) for i in markers['tool_call']],
        answer_pct=[round(100 * i / max(1, n - 1), 1) for i in markers['answer']],
        ent_pre=round(ent_pre, 3) if ent_pre is not None else None,
        ent_post=round(ent_post, 3) if ent_post is not None else None,
        n_hi_pre=n_hi_pre if first_tc is not None else None, n_hi_post=n_hi_post, n_hi=len(hi_idx),
        first_tc=first_tc, mean_entropy=round(float(e.mean()), 3), pre_answer=pre_ans, pre_letter=pre_letter,
        temporal_hit=hit, estar=estar, label=lab.get('label'),
        diagnosis=dict(code=code, title=title, color=color, details=details))))


# ───────────────────────── 路由 ─────────────────────────

@app.get('/', response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, 'static', 'index.html'), encoding='utf-8').read()


@app.get('/api/status')
def status():
    return dict(model=STATE['model_name'], loading=STATE['loading'], n_items=len(ITEMS),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                gpu_mem_gb=round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else None,
                load=dict(target=LOAD['target'], phase=LOAD['phase'], percent=LOAD['percent'],
                          log=LOAD['log'][-30:], error=LOAD['error'],
                          elapsed=round(time.time() - LOAD['t0'], 0) if LOAD['t0'] else None))


@app.post('/api/load')
def load_model(req: dict):
    """切换/预加载模型 (不推理). 前端在下拉框改变时调用, 然后轮询 /api/status 画进度条."""
    name = req.get('model', 'deepeyes')
    if name not in B.MODELS:
        return JSONResponse({'error': f'未知模型 {name}'}, status_code=400)
    if STATE['model_name'] == name and STATE['backend'] is not None:
        return dict(ok=True, already=True)
    if STATE['loading']:
        return JSONResponse({'error': f"正在加载 {LOAD['target']}，等它结束"}, status_code=409)
    threading.Thread(target=ensure_backend, args=(name,), daemon=True).start()
    return dict(ok=True, started=True)


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
    ap.add_argument('--model', default='qwen35'); ap.add_argument('--no-preload', action='store_true')
    a = ap.parse_args()
    load_data()
    if not a.no_preload:
        threading.Thread(target=ensure_backend, args=(a.model,), daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level='info')
