"""EvidenceGap-V pilot: 在真实 VideoNet 视频上做最小合成编辑, 造 3 条"粗观测客观不可辨"的孪生样本.

每条样本 = 一对视频 (A/B), 粗观测 (8 帧均匀采样, ≤256*28² 像素) 下两者不可区分, 判别证据只在
E* = (时间窗 × 空间框) 里. 三条覆盖三种机制:
  s1_occlusion_color   遮挡解除: 卡片盖住的圆片只在两个粗采样点之间的短窗内露出 (时间为主)
  s2_tiny_tag          局部小字: 标签全程可见但在粗分辨率下不可读 (空间为主)
  s3_roll_direction    短时窗动作: 小球只在短窗内滚过, 方向需要≥2帧的时序 (时空+顺序)

用法: python make_samples.py [--version v1]   →  videos/*.mp4 + samples.json
所有可调参数集中在 PARAMS, 迭代时只改这里并升 version.
"""
import argparse, json, os, subprocess
import numpy as np
import decord
from PIL import Image, ImageDraw, ImageFont

VIDEONET = '/mnt/afs/panzhengyuan/VideoNet/videos'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_VID = os.path.join(HERE, 'videos')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
N_COARSE = 8

COLORS = {'red': (220, 30, 30), 'blue': (30, 60, 220), 'green': (30, 170, 60), 'yellow': (240, 210, 30)}

PARAMS = {
    's1_occlusion_color': dict(
        base='f103252e-1db7-4fb2-b65d-4924c410a4fe.mp4',      # 1920x1080, 432f @24fps
        # 圆片中心/半径; 卡片矩形; 窗口 = 卡片滑开露出圆片的帧区间 (在粗采样点 246 与 308 之间)
        disc_xy=(300, 850), disc_r=45,
        card=(190, 740, 410, 960), card_shift=(320, 0),
        window=(262, 292),
        twins={'A': 'red', 'B': 'blue'},
        options=['red', 'blue', 'green', 'yellow'],
        question=('On the counter at the lower left there is a small round object that is covered by a '
                  'dark card for most of the video. At one brief moment the card slides aside and the object '
                  'is visible. What color is the round object?'),
        question_zh='画面左下方的台面上有一个小圆形物体，视频里大部分时间它都被一张深色卡片盖着，只有很短的一瞬卡片滑开、物体露了出来。这个圆形物体是什么颜色？',
        options_zh=['红色', '蓝色', '绿色', '黄色'],
        # E*: 时间窗 + 空间框 (原图坐标); decoy = 同框但取窗口外的帧 (只看到卡片)
        evidence_bbox=(190, 740, 410, 960),
    ),
    's2_tiny_tag': dict(
        base='905ce3c1-63bf-455b-abc9-c1b61352f86e.mp4',      # 1280x720, 96f @24fps
        # v1: text 24px/tag 66x32 -> 粗采样下 p=0.99 直接读出 (C1 挂); v2 缩到 15px/44x22
        # v2: 15px/44x22 -> 粗采样 (缩放 0.46, 字高 ~7px) 仍 p=0.99 读出; v3: 12px/36x18
        # v3: 12px/36x18 (粗采样 ~5.5px) 仍 p=0.99; v4: 9px/28x14 (~4.1px)
        tag_xy=(1085, 620), tag_wh=(28, 14), text_px=9,     # 标签左上角/尺寸/字高 (原图像素)
        window=None,                                        # 全程可见
        twins={'A': '358', 'B': '386'},
        options=['358', '356', '338', '386'],
        question=('There is a small white tag with a number printed on it, lying on the bar counter near the '
                  'lower right of the frame. What number is printed on the tag?'),
        question_zh='画面右下方的吧台上放着一张白色小标签，上面印着一个数字。标签上印的数字是多少？',
        options_zh=['358', '356', '338', '386'],
        evidence_bbox=(1077, 612, 1121, 642),
        decoy_bbox=(880, 612, 924, 642),                    # 同尺寸、同高度的另一块台面 (无标签)
    ),
    's2b_tiny_tag_1080p': dict(
        # 同一机制换 1080p 底片: 粗采样缩放比 0.31, 15px 字 -> ~4.7px; 证据裁剪保留原生 15px
        base='f103252e-1db7-4fb2-b65d-4924c410a4fe.mp4',
        # v3: 15px/44x22 (粗采样 ~4.7px) A 仍 0.87 读出 "358", B 误读 356 -> 半可读; v4: 12px/36x18 (~3.7px)
        tag_xy=(620, 940), tag_wh=(36, 18), text_px=12,
        window=None,
        twins={'A': '358', 'B': '386'},
        options=['358', '356', '338', '386'],
        question=('There is a small white tag with a number printed on it, lying on the counter near the '
                  'bottom center of the frame. What number is printed on the tag?'),
        question_zh='画面底部中央的台面上放着一张白色小标签，上面印着一个数字。标签上印的数字是多少？',
        options_zh=['358', '356', '338', '386'],
        evidence_bbox=(610, 930, 666, 968),
        decoy_bbox=(400, 930, 456, 968),
    ),
    's3b_flash_symbol': dict(
        base='1c47ea6a-274d-450f-98bc-26260727428c.mp4',
        card_xy=(1290, 870), card_wh=(120, 120), window=(160, 187),
        twins={'A': 'triangle', 'B': 'circle'},     # v2 用 cross, 纯文本先验 0.52 (C3 挂); v3 换 circle
        options=['triangle', 'square', 'circle', 'cross'],
        question=('At one brief moment a small white card with a black symbol on it is shown on the table '
                  'near the lower right. What symbol is on the card?'),
        question_zh='视频中有一瞬间，画面右下方的桌面上出现一张印着黑色图案的白色卡片。卡片上的图案是什么？',
        options_zh=['三角形', '正方形', '圆形', '十字'],
        evidence_bbox=(1260, 840, 1440, 1020),
    ),
    's3_roll_direction': dict(
        base='1c47ea6a-274d-450f-98bc-26260727428c.mp4',      # 1920x1080, 270f @30fps
        # v1: r=30, 选项 "from left to right" 纯文本先验 0.82, 真证据也推不出 (C1-C4 全挂)
        # v2: 球放大, 选项改成对称措辞, 证据框加高让球在裁剪图里更显眼
        ball_r=48, ball_color=(245, 140, 20), y=960, x_range=(260, 1660),
        window=(160, 187),                                  # 粗采样点 153.7 与 192.1 之间
        twins={'A': 'toward the right side', 'B': 'toward the left side'},
        options=['toward the right side', 'toward the left side', 'toward the top', 'toward the bottom'],
        question=('At one brief moment a small orange ball rolls across the table near the bottom of the '
                  'frame. Toward which side of the frame does the ball move as time goes on?'),
        question_zh='视频中有一瞬间，一个橙色小球从画面底部的桌面上滚过。随着时间推移，小球朝画面的哪一侧运动？',
        options_zh=['朝右侧', '朝左侧', '朝上方', '朝下方'],
        evidence_bbox=(200, 840, 1720, 1080),
    ),
}


def _read(base):
    vr = decord.VideoReader(os.path.join(VIDEONET, base))
    return vr, float(vr.get_avg_fps()), len(vr)


def _write(path, frames, fps):
    h, w, _ = frames[0].shape
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', f'{fps}', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
           '-pix_fmt', 'yuv420p', path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()
    assert p.returncode == 0, f'ffmpeg failed for {path}'


def _coarse_idx(n):
    return np.unique(np.linspace(0, n - 1, N_COARSE).astype(int)).tolist()


def render_s1(p, variant):
    vr, fps, n = _read(p['base'])
    cx, cy = p['disc_xy']; r = p['disc_r']
    l, t, rr, b = p['card']; dx, dy = p['card_shift']
    w0, w1 = p['window']
    col = COLORS[p['twins'][variant]]
    out = []
    for i in range(n):
        im = Image.fromarray(vr[i].asnumpy()); d = ImageDraw.Draw(im)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col, outline=(20, 20, 20), width=3)
        if w0 <= i < w1:                              # 窗口内: 卡片滑到一旁
            box = (l + dx, t + dy, rr + dx, b + dy)
        else:
            box = (l, t, rr, b)
        d.rounded_rectangle(box, radius=18, fill=(48, 48, 52), outline=(15, 15, 15), width=3)
        out.append(np.asarray(im))
    return out, fps, n


def render_s2(p, variant):
    vr, fps, n = _read(p['base'])
    x, y = p['tag_xy']; w, h = p['tag_wh']
    font = ImageFont.truetype(FONT, p['text_px'])
    txt = p['twins'][variant]
    out = []
    for i in range(n):
        im = Image.fromarray(vr[i].asnumpy()); d = ImageDraw.Draw(im)
        d.rounded_rectangle((x, y, x + w, y + h), radius=4, fill=(245, 245, 240), outline=(70, 70, 70), width=1)
        tw = d.textlength(txt, font=font)
        d.text((x + (w - tw) / 2, y + (h - p['text_px']) / 2 - 2), txt, fill=(15, 15, 15), font=font)
        out.append(np.asarray(im))
    return out, fps, n


def render_s3(p, variant):
    vr, fps, n = _read(p['base'])
    w0, w1 = p['window']; x0, x1 = p['x_range']; y = p['y']; r = p['ball_r']
    if variant == 'B':
        x0, x1 = x1, x0
    out = []
    for i in range(n):
        im = Image.fromarray(vr[i].asnumpy())
        if w0 <= i < w1:
            a = (i - w0) / max(1, (w1 - 1 - w0))
            x = x0 + a * (x1 - x0)
            d = ImageDraw.Draw(im)
            d.ellipse((x - r, y - r, x + r, y + r), fill=p['ball_color'], outline=(60, 30, 0), width=3)
            # 一条高光让它像球而不是圆片
            d.ellipse((x - r * 0.45, y - r * 0.55, x - r * 0.05, y - r * 0.15), fill=(255, 235, 200))
        out.append(np.asarray(im))
    return out, fps, n


def _draw_symbol(d, sym, cx, cy, s):
    """在 (cx,cy) 画边长/直径约 s 的黑色符号."""
    k = s / 2
    if sym == 'triangle':
        d.polygon([(cx, cy - k), (cx - k, cy + k * 0.8), (cx + k, cy + k * 0.8)], fill=(15, 15, 15))
    elif sym == 'square':
        d.rectangle((cx - k * 0.85, cy - k * 0.85, cx + k * 0.85, cy + k * 0.85), fill=(15, 15, 15))
    elif sym == 'circle':
        d.ellipse((cx - k * 0.9, cy - k * 0.9, cx + k * 0.9, cy + k * 0.9), fill=(15, 15, 15))
    elif sym == 'cross':
        w = s * 0.14
        d.rectangle((cx - k, cy - w, cx + k, cy + w), fill=(15, 15, 15))
        d.rectangle((cx - w, cy - k, cx + w, cy + k), fill=(15, 15, 15))


def render_s3b(p, variant):
    vr, fps, n = _read(p['base'])
    x, y = p['card_xy']; w, h = p['card_wh']; w0, w1 = p['window']
    sym = p['twins'][variant]
    out = []
    for i in range(n):
        im = Image.fromarray(vr[i].asnumpy())
        if w0 <= i < w1:
            d = ImageDraw.Draw(im)
            d.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=(245, 245, 240), outline=(60, 60, 60), width=2)
            _draw_symbol(d, sym, x + w / 2, y + h / 2, min(w, h) * 0.6)
        out.append(np.asarray(im))
    return out, fps, n


RENDER = {'s1_occlusion_color': render_s1, 's2_tiny_tag': render_s2, 's2b_tiny_tag_1080p': render_s2,
          's3_roll_direction': render_s3, 's3b_flash_symbol': render_s3b}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--version', default='v1')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--meta-only', action='store_true', help='不重渲染视频, 只按 PARAMS 重写 samples.json')
    args = ap.parse_args()
    os.makedirs(OUT_VID, exist_ok=True)
    samples = []
    for sid, p in PARAMS.items():
        if args.only and sid not in args.only:
            continue
        rec = dict(id=sid, version=args.version, base=p['base'], question=p['question'],
                   options=p['options'], question_zh=p['question_zh'], options_zh=p['options_zh'],
                   twins={}, evidence_bbox=list(p['evidence_bbox']),
                   decoy_bbox=list(p.get('decoy_bbox') or p['evidence_bbox']))
        for variant in ('A', 'B'):
            fn = f'{sid}_{variant}.mp4'
            if args.meta_only:
                _, fps, n = _read(p['base'])
            else:
                frames, fps, n = RENDER[sid](p, variant)
                _write(os.path.join(OUT_VID, fn), frames, fps)
            ans = p['twins'][variant]; ai = p['options'].index(ans)
            rec['twins'][variant] = dict(video=fn, answer=ans, answer_idx=ai, answer_letter='ABCD'[ai],
                                         answer_zh=p['options_zh'][ai])
            rec['fps'] = fps; rec['n_frames'] = n
            print(f'{fn}: {n} frames @ {fps:.1f}fps, answer={ans}')
        if p.get('window'):
            w0, w1 = p['window']
            rec['evidence_window_frames'] = [w0, w1]
            rec['evidence_window_sec'] = [round(w0 / fps, 2), round(w1 / fps, 2)]
            ci = _coarse_idx(n)
            inside = [i for i in ci if w0 <= i < w1]
            assert not inside, f'{sid}: coarse frames {inside} fall inside window!'
            # decoy 帧: 窗口外、离窗口最远的一段 (保证与 E* 同框不同时)
            rec['decoy_window_frames'] = [0, min(n, w0 - 10)] if w0 > n - w1 else [w1 + 10, n]
        else:
            rec['evidence_window_frames'] = None
            rec['decoy_window_frames'] = None
        rec['coarse_frame_idx'] = _coarse_idx(n)
        rec['params'] = {k: v for k, v in p.items() if k not in ('question', 'options', 'twins')}
        samples.append(rec)
    out = os.path.join(HERE, 'samples.json')
    prev = json.load(open(out)) if os.path.exists(out) and args.only else []
    if args.meta_only:   # 保留各样本原有的 version 标记
        old = {s['id']: s.get('version') for s in (json.load(open(out)) if os.path.exists(out) else [])}
        for s in samples:
            s['version'] = old.get(s['id'], s['version'])
    keep = [s for s in prev if s['id'] not in {x['id'] for x in samples}]
    json.dump(keep + samples, open(out, 'w'), indent=1, ensure_ascii=False)
    print(f'-> {out} ({len(keep) + len(samples)} samples)')


if __name__ == '__main__':
    main()
