"""可续跑的批量驱动: 记录键 (item, model, stage), JSONL 追加, 每条 try/except.

比 gaze_probe.run_arm 的做法改了一处: 它每条都重写整个 JSON (O(n²)), 这里改成 JSONL 追加.
"""
import json, os, time, traceback


def load_done(path, keyfn=lambda r: (r.get('uuid'), r.get('model'), r.get('stage'))):
    """读已完成的记录, 返回 (done_keys, records)."""
    done, recs = set(), []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                       # 上次被杀时写了半行, 丢掉
            recs.append(r)
            if not r.get('error'):
                done.add(keyfn(r))
    return done, recs


def run(units, fn, out_path, keyfn=None, desc='', flush_every=1, verbose=True):
    """units: [(key_tuple, payload), ...]; fn(payload) -> dict.

    每条结果写成一行 JSON, 含 key 的三个字段 (uuid/model/stage) + fn 的返回值.
    已完成的 key 直接跳过 —— 中断后重跑同一命令即可续上.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    done, _ = load_done(out_path)
    todo = [(k, p) for k, p in units if k not in done]
    if verbose:
        print(f'[{desc}] 共 {len(units)} 个单元, 已完成 {len(units)-len(todo)}, 待跑 {len(todo)}')
    f = open(out_path, 'a', buffering=1)
    t_start = time.time()
    n_err = 0
    for i, (key, payload) in enumerate(todo):
        rec = dict(uuid=key[0], model=key[1], stage=key[2])
        t0 = time.time()
        try:
            rec.update(fn(payload))
        except Exception as e:
            n_err += 1
            rec['error'] = f'{type(e).__name__}: {e}'
            rec['traceback'] = traceback.format_exc()[-1500:]
            if verbose:
                print(f'  ✘ {key[0][:8]} {key[2]}: {rec["error"]}')
        rec['wall_s'] = round(time.time() - t0, 1)
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        if verbose and ((i + 1) % flush_every == 0 or i == len(todo) - 1):
            el = time.time() - t_start
            eta = el / (i + 1) * (len(todo) - i - 1)
            print(f'  [{i+1}/{len(todo)}] {key[0][:8]} {key[2]} {rec["wall_s"]}s '
                  f'| 已用 {el/60:.1f}min ETA {eta/60:.1f}min | 错误 {n_err}', flush=True)
    f.close()
    return load_done(out_path)[1]


def read_records(path, stage=None, model=None):
    _, recs = load_done(path)
    return [r for r in recs if (stage is None or r.get('stage') == stage)
            and (model is None or r.get('model') == model) and not r.get('error')]
