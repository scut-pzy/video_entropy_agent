"""从 VideoNet 的 MCQ 里选出本地有视频的题, 并 join 元数据 (domain_name / name / definition).

不用 video_imcot.load_items: 它没有 seed、DL_VIDS 优先、只读 test split, 且 VIDEONET 是会被
eval_deepeyes 在 import 时改写的模块级全局.

用法:
    from items import load_items, by_domain
    items = load_items()                       # 339 条 (274 test + 65 val)
    items = load_items(min_side=720)           # 只要 >=720p 的 (195 条)
"""
import json, os, re
from collections import defaultdict

VIDEONET = '/mnt/afs/panzhengyuan/VideoNet'
VIDEOS = os.path.join(VIDEONET, 'videos')
META = os.path.join(VIDEONET, 'benchmark_video_metadata.parquet')
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_CACHE = os.path.join(HERE, 'cache', 'video_probe.json')   # 视频属性缓存 (需要 decord)

_OPT_RE = re.compile(r'^\s*([ABCD])[\.\)]\s*(.+?)\s*$')
_TAIL = 'Please respond with only the letter of the correct answer.'


def parse_question(text):
    """题面 -> (stem, [4 个选项]). 选项行形如 'A. split jump'."""
    lines = [l for l in text.split('\n') if l.strip()]
    stem, opts = [], {}
    for l in lines:
        if l.strip() == _TAIL:
            continue
        m = _OPT_RE.match(l)
        if m and m.group(1) not in opts:
            opts[m.group(1)] = m.group(2)
        elif not opts:
            stem.append(l.strip())
    if len(opts) != 4:
        raise ValueError(f'expected 4 options, got {sorted(opts)} from: {text[:120]!r}')
    return ' '.join(stem), [opts[L] for L in 'ABCD']


def _read_meta():
    import pyarrow.parquet as pq
    t = pq.read_table(META).to_pydict()
    return {u: dict(yt_id=y, start=s, end=e, name=n, domain_name=d, definition=f)
            for u, y, s, e, n, d, f in zip(t['uuid'], t['yt_id'], t['start'], t['end'],
                                           t['name'], t['domain_name'], t['definition'])}


def _load_probe():
    return json.load(open(PROBE_CACHE)) if os.path.exists(PROBE_CACHE) else {}


def probe_videos(uuids=None, force=False):
    """用 decord 读每条视频的 fps/帧数/分辨率, 缓存到 cache/video_probe.json. 需要 ms_swift env."""
    import decord
    cache = {} if force else _load_probe()
    have = {f[:-4] for f in os.listdir(VIDEOS) if f.endswith('.mp4')}
    todo = sorted((set(uuids) if uuids else have) - set(cache))
    for i, u in enumerate(todo):
        try:
            vr = decord.VideoReader(os.path.join(VIDEOS, f'{u}.mp4'))
            h, w, _ = vr[0].shape
            cache[u] = dict(n_frames=len(vr), fps=float(vr.get_avg_fps()),
                            width=int(w), height=int(h), duration=len(vr) / float(vr.get_avg_fps()))
        except Exception as e:
            cache[u] = dict(error=str(e)[:200])
        if (i + 1) % 50 == 0:
            print(f'  probed {i+1}/{len(todo)}')
    os.makedirs(os.path.dirname(PROBE_CACHE), exist_ok=True)
    json.dump(cache, open(PROBE_CACHE, 'w'), indent=0)
    return cache


def load_items(splits=('test', 'val'), min_side=None, domains=None, max_duration=None):
    """返回本地有视频的 MCQ 列表, 每条含题面/选项/答案/元数据/视频属性.

    splits     'test' / 'val' / 两者
    min_side   只保留短边 >= 该值的视频 (需要 probe 缓存; 720 -> 约 195 条)
    domains    只保留这些 domain_name
    max_duration  只保留时长 <= 该秒数的
    """
    meta = _read_meta()
    probe = _load_probe()
    have = {f[:-4] for f in os.listdir(VIDEOS) if f.endswith('.mp4')}
    out = []
    for sp in splits:
        path = os.path.join(VIDEONET, 'benchmarks', f'mcq_{sp}.jsonl')
        for line in open(path):
            it = json.loads(line)
            vid = next(c['video'] for c in it['question'] if c['type'] == 'video')
            uuid = vid[:-4]
            if uuid not in have:
                continue
            text = next(c['text'] for c in it['question'] if c['type'] == 'text')
            stem, opts = parse_question(text)
            m = meta.get(uuid, {})
            p = probe.get(uuid, {})
            rec = dict(key=it['key'], split=sp, uuid=uuid,
                       video=os.path.join(VIDEOS, vid),
                       stem=stem, options=opts,
                       answer_letter=it['answer'], answer_idx='ABCD'.index(it['answer']),
                       answer=opts['ABCD'.index(it['answer'])],
                       domain=m.get('domain_name'), name=m.get('name'),
                       definition=m.get('definition'),
                       duration=p.get('duration'), n_frames=p.get('n_frames'),
                       fps=p.get('fps'), width=p.get('width'), height=p.get('height'))
            if domains and rec['domain'] not in domains:
                continue
            if min_side is not None:
                if not rec['width'] or min(rec['width'], rec['height']) < min_side:
                    continue
            if max_duration is not None and (rec['duration'] or 1e9) > max_duration:
                continue
            out.append(rec)
    out.sort(key=lambda r: r['uuid'])
    return out


def by_domain(items):
    d = defaultdict(list)
    for it in items:
        d[it['domain']].append(it)
    return dict(sorted(d.items(), key=lambda kv: -len(kv[1])))


def sibling_map(items):
    """同域兄弟: uuid -> 该题 4 个选项里, 哪些干扰类在本地也有视频 (用于 twin 条件).

    返回 {uuid: [{'option_idx':i, 'option':名字, 'uuid':兄弟视频uuid}, ...]}
    """
    by_name = defaultdict(list)
    for it in items:
        if it['name']:
            by_name[it['name'].strip().lower()].append(it['uuid'])
    out = {}
    for it in items:
        sibs = []
        for i, o in enumerate(it['options']):
            if i == it['answer_idx']:
                continue
            for u in by_name.get(o.strip().lower(), []):
                if u != it['uuid']:
                    sibs.append(dict(option_idx=i, option=o, uuid=u))
                    break
        out[it['uuid']] = sibs
    return out


if __name__ == '__main__':
    import sys
    if '--probe' in sys.argv:
        c = probe_videos(force='--force' in sys.argv)
        ok = [v for v in c.values() if 'error' not in v]
        print(f'probed {len(c)} videos, {len(c)-len(ok)} errors')
    items = load_items()
    print(f'{len(items)} items with local video '
          f"({sum(i['split']=='test' for i in items)} test / {sum(i['split']=='val' for i in items)} val)")
    hd = load_items(min_side=720)
    print(f'{len(hd)} items with >=720p')
    d = by_domain(items)
    print('top domains:', [(k, len(v)) for k, v in list(d.items())[:8]])
    sm = sibling_map(items)
    n = sum(1 for v in sm.values() if v)
    print(f'items with >=1 local sibling (twin 条件可用): {n}/{len(items)}')
    it = items[0]
    print('\nexample:', json.dumps({k: it[k] for k in
          ('key', 'uuid', 'stem', 'options', 'answer_letter', 'domain', 'name', 'duration', 'width', 'height')},
          ensure_ascii=False, indent=1)[:900])
