"""帧缓存: 把需要的帧解码一次存成原分辨率 JPEG, 之后所有条件 (不同 k / 不同分辨率 / 裁剪) 都从缓存读.

这么做有两个理由:
  1. 解耦环境 —— 解码需要 decord (只有 ms_swift 有), 但 Qwen3.8-27B 要在 verl_qwen35 里跑; 有缓存后
     推理进程只读 JPEG, 两个 env 都不用改.
  2. 同一条视频在筛选/定位/审计里会被反复读几十次, 解码一次省很多时间.

缓存布局: frames_cache/<uuid>/<frame_idx:06d>.jpg  (原分辨率, q=88)
用法:
    import frames as F
    idx  = F.uniform_indices(n_total, k=8)
    pils = F.load(uuid, idx, pixel_cap=F.COARSE_PIXELS)      # 已按 smart_resize 缩放
    orig = F.load(uuid, idx)                                  # 原分辨率
"""
import math, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'frames_cache')
VIDEOS = '/mnt/afs/panzhengyuan/VideoNet/videos'

COARSE_PIXELS = 256 * 28 * 28      # 粗观测分辨率上限 (与 07-23 探针、pilot 一致)
DETAIL_PIXELS = 768 * 28 * 28      # 工具返回帧/裁剪的分辨率上限
JPEG_Q = 88


def smart_resize(height, width, factor=28, min_pixels=4 * 28 * 28, max_pixels=16384 * 28 * 28):
    """与 video_imcot.smart_resize 一致 (Qwen-VL 的 28px patch 对齐)."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def uniform_indices(n_total, k):
    """k 帧均匀采样的帧号. 视频太短时会去重, 所以实际帧数可能 < k —— 调用方要记录实际值."""
    return np.unique(np.linspace(0, n_total - 1, k).astype(int)).tolist()


def window_indices(n_total, t0, t1, n, fps):
    """时间窗 [t0,t1] 秒内均匀取 n 帧的帧号."""
    f0 = max(0, int(round(t0 * fps)))
    f1 = min(n_total - 1, int(round(t1 * fps)))
    if f1 <= f0:
        f1 = min(n_total - 1, f0 + 1)
    return np.unique(np.linspace(f0, f1, n).astype(int)).tolist()


def _path(uuid, idx):
    return os.path.join(CACHE, uuid, f'{idx:06d}.jpg')


def ensure(uuid, indices, video_path=None):
    """确保这些帧已在缓存里; 缺的用 decord 解码补上 (需要 ms_swift env). 返回路径列表."""
    paths = [_path(uuid, i) for i in indices]
    missing = [(i, p) for i, p in zip(indices, paths) if not os.path.exists(p)]
    if missing:
        import decord
        os.makedirs(os.path.join(CACHE, uuid), exist_ok=True)
        vr = decord.VideoReader(video_path or os.path.join(VIDEOS, f'{uuid}.mp4'))
        n = len(vr)
        for i, p in missing:
            arr = vr[int(min(max(i, 0), n - 1))].asnumpy()
            Image.fromarray(arr).save(p, quality=JPEG_Q)
    return paths


def load(uuid, indices, pixel_cap=None, video_path=None):
    """读帧; pixel_cap 不为 None 时按 smart_resize 缩放到该像素上限."""
    out = []
    for p in ensure(uuid, indices, video_path):
        im = Image.open(p).convert('RGB')
        if pixel_cap is not None:
            h, w = smart_resize(im.height, im.width, max_pixels=pixel_cap)
            im = im.resize((w, h), Image.BICUBIC)
        out.append(im)
    return out


def crop(uuid, indices, bbox, pixel_cap=DETAIL_PIXELS, video_path=None):
    """在原分辨率帧上裁 bbox (原图像素坐标), 再按 pixel_cap 缩放. 用于空间 E* / decoy 框."""
    out = []
    for p in ensure(uuid, indices, video_path):
        im = Image.open(p).convert('RGB')
        l, t, r, b = [int(v) for v in bbox]
        l, t = max(0, l), max(0, t)
        r, b = min(im.width, r), min(im.height, b)
        if r - l < 8 or b - t < 8:
            raise ValueError(f'bbox too small after clamp: {bbox} on {im.size}')
        c = im.crop((l, t, r, b))
        h, w = smart_resize(c.height, c.width, max_pixels=pixel_cap)
        out.append(c.resize((w, h), Image.BICUBIC))
    return out


def cache_stats():
    n, size = 0, 0
    if not os.path.isdir(CACHE):
        return dict(videos=0, frames=0, mb=0.0)
    vids = os.listdir(CACHE)
    for v in vids:
        d = os.path.join(CACHE, v)
        for f in os.listdir(d):
            n += 1; size += os.path.getsize(os.path.join(d, f))
    return dict(videos=len(vids), frames=n, mb=round(size / 1e6, 1))


def prefetch(items, ks=(2, 4, 8, 16, 32), verbose=True):
    """把筛选阶段要用的所有帧一次性解码进缓存."""
    for j, it in enumerate(items):
        idx = sorted({i for k in ks for i in uniform_indices(it['n_frames'], k)})
        ensure(it['uuid'], idx, it['video'])
        if verbose and (j + 1) % 25 == 0:
            print(f"  {j+1}/{len(items)}  {cache_stats()}")
    return cache_stats()


if __name__ == '__main__':
    import sys, items as I
    its = I.load_items()
    if '--prefetch' in sys.argv:
        print(prefetch(its))
    else:
        it = its[0]
        idx = uniform_indices(it['n_frames'], 8)
        pil = load(it['uuid'], idx, COARSE_PIXELS, it['video'])
        print(it['uuid'], 'n_frames', it['n_frames'], 'idx', idx, 'sizes', [p.size for p in pil][:3])
        print('cache', cache_stats())
