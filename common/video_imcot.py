"""视频版 DeepEyes iMCoT 探针 — 全部辅助代码.

notebook 4 个 cell 依次调用:
  1) load_model()      加载 DeepEyes-7B
  2) load_items() + show_item()   加载 VideoNet MCQ 数据
  3) run_all(items, ...)          测评 (所有变量可在调用处覆盖)
  4) summarize(records)           统计

协议实现与 autoresearch 实验代码 (autoresearch/orchestrator-260723-1900/eval_variants.py)
逐一对齐: replay 模式 == 实验中的 scaffold_revise (全池 274 条上最佳: 43.4% vs direct 43.1%).
"""
import os, re, json, math, time
import numpy as np
import torch
import decord
from PIL import Image
from IPython.display import display

# ════════════════════════════════════════════════════════════════════
# 配置区: 所有可调变量集中在这里. notebook 里既可改这里, 也可在
# run_all(...) 调用时逐参数覆盖 (调用参数优先).
# ════════════════════════════════════════════════════════════════════
MODEL_PATH = '/mnt/afs/panzhengyuan/llm_training/models/DeepEyes-7B'
VIDEONET   = '/mnt/afs/panzhengyuan/VideoNet'
OUT_DIR    = '/mnt/afs/panzhengyuan/llm_training/project/video-agent/results'

# ---- 帧截取变量 ----
N_COARSE       = 8              # 初始均匀采样帧数 (decord 等时间间隔截取; 基线对比时保持一致)
REPLAY_FRAMES  = 6              # replay 模式回放段补采帧数 (= 对该段的局部帧率提升)
SEEK_MAX       = 12             # 单次回放/seek 补采帧数硬上限 (防止一次调用塞爆上下文)
COARSE_PIXELS  = 256 * 28 * 28  # 初始帧分辨率上限, 像素数 (≈450x450; Qwen-VL 按 28px patch)
DETAIL_PIXELS  = 768 * 28 * 28  # 回放帧 / zoom 裁剪图分辨率上限 (实验试过 1280*28*28, 无提升)
FRAME_HEADROOM = 18             # 上下文帧总数上限 = n_coarse + FRAME_HEADROOM (工具还能再添多少帧)

# ---- 推理变量 ----
MAX_TURNS             = 4       # zoom/full 模式最多推理轮数
MAX_NEW_TOKENS        = 512     # agent 每轮生成 token 上限
DIRECT_MAX_NEW_TOKENS = 64      # direct 直答生成上限 (只需字母)

# 本次实验专门下载的 10 条细粒度域视频 (load_items 优先取这些)
DL_VIDS = [
    'dffb7271-5fce-44f7-9e78-65489c59e518.mp4',
    '34bd7239-46e9-4769-8896-ae3a1d8847ac.mp4',
    '1c47ea6a-274d-450f-98bc-26260727428c.mp4',
    'f103252e-1db7-4fb2-b65d-4924c410a4fe.mp4',
    '5cc0e669-2cf7-4dbd-b991-128fad82b846.mp4',
    '79b1d7d7-b0bb-42d8-a526-0eaa566e1abf.mp4',
    '32f317ad-2804-4925-ad5e-c3d866f951ae.mp4',
    'd41f172a-d38a-4ba6-93fa-4311a3349111.mp4',
    '905ce3c1-63bf-455b-abc9-c1b61352f86e.mp4',
    '8c842293-6717-4988-a91c-6615251f2f66.mp4',
]

model = None
processor = None

# 四种测评模式 (实际帧数等以 run_all 打印的方括号实时值为准)
MODE_DESC = {
    'direct': 'direct = baseline: 模型看均匀采样的粗采帧后直接回答, 不给任何工具',
    'zoom':   'zoom = 给模型 image_zoom_in_tool: 推理中可指定"把第 i 帧的 bbox 区域放大", '
              '裁剪图以更高分辨率塞回对话继续推理 (DeepEyes 原生工具 + frame_index 扩展)',
    'full':   'full = zoom 基础上再加 video_seek_tool: 模型可指定时间段 [t1,t2] 密集补采帧'
              ' ("回看这一段"), 测时间维工具的零样本泛化',
    'replay': 'replay = autoresearch 胜出的保守回放协议 (scaffold_revise): 先直答记初始答案 -> '
              '模型只做时间定位(<segment>) -> 程序化密集回放该段 -> 仅当回放明确矛盾时才改答案',
}

# ════════════════════════════════════════════════════════════════════ 1. 模型

def load_model():
    """加载 DeepEyes-7B 到 cuda:0 (bf16 + flash-attn)."""
    global model, processor
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map={'': 0})
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print(f'DeepEyes-7B 加载完成: {time.time()-t0:.0f}s, device={model.device}')
    return model, processor

# ════════════════════════════════════════════════════════════════════ 2. 数据

def _domain_of(item):
    q = item['question'][0]['text']
    return q.split('Which of the following ')[1].split(' actions')[0] \
        if 'Which of the following ' in q else '?'

def load_items(n=10, domains=None):
    """从 mcq_test.jsonl 选出本地已有视频的 MCQ, 每视频 1 条.

    n       条数上限 (本地视频池约 274 条可用; autoresearch 用过 50/100/274)
    domains 域过滤, 如 ['Cooking', 'Figure Skating']; None = 不过滤"""
    have = {f for f in os.listdir(f'{VIDEONET}/videos') if f.endswith('.mp4')}
    all_items = [json.loads(l) for l in open(f'{VIDEONET}/benchmarks/mcq_test.jsonl')]
    by_vid = {}
    for it in all_items:
        vid = it['question'][1]['video']
        if vid in have and vid not in by_vid:
            by_vid[vid] = it
    ordered = [by_vid[v] for v in DL_VIDS if v in by_vid]
    ordered += [it for v, it in by_vid.items() if v not in set(DL_VIDS)]
    if domains:
        ordered = [it for it in ordered
                   if any(d.lower() in _domain_of(it).lower() for d in domains)]
    items = ordered[:n]
    print(f'选出 {len(items)} 条测试样本 (本地视频池共 {len(by_vid)} 条可用'
          + (f', 域过滤 {domains}' if domains else '') + '):')
    for it in items[:30]:
        print(f"  [{it['answer']}] {_domain_of(it):22s} {it['question'][1]['video'][:8]}")
    if len(items) > 30:
        print(f'  ... 其余 {len(items) - 30} 条略')
    return items

def show_item(item, n_coarse=None):
    """打印一条样本的完整题面/答案, 并显示其均匀截取的粗采帧."""
    n_coarse = n_coarse or N_COARSE
    print('question:')
    print(item['question'][0]['text'])
    print('\nvideo :', item['question'][1]['video'])
    print('answer:', item['answer'])
    sess = VideoSession(f"{VIDEONET}/videos/{item['question'][1]['video']}", n_coarse=n_coarse)
    print(f'视频时长 {sess.duration:.1f}s; 下面是用 decord 按等时间间隔截取的 {len(sess.frames)} 帧'
          f' (n_coarse={n_coarse}, 每帧 ≤{sess.coarse_pixels} 像素):')
    display(_grid([f['pil_disp'] for f in sess.frames], thumb=170))

def _grid(pils, thumb=170, cols=4):
    thumbs = []
    for p in pils:
        t = p.copy(); t.thumbnail((thumb, thumb)); thumbs.append(t)
    W = max(t.width for t in thumbs); H = max(t.height for t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new('RGB', (W * cols, H * rows), 'white')
    for i, t in enumerate(thumbs):
        canvas.paste(t, ((i % cols) * W, (i // cols) * H))
    return canvas

# ════════════════════════════════════════════════════════════════════ 3. 视频会话 + 工具

def smart_resize(height, width, factor=28, min_pixels=4*28*28, max_pixels=16384*28*28):
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

class VideoSession:
    """一个视频 episode 的帧注册表: 模型用 frame index 引用帧, 工具在原始分辨率上操作.

    帧变量: n_coarse 初始均匀帧数 / coarse_pixels 初始帧分辨率上限 /
            detail_pixels 工具产生帧(回放/裁剪)分辨率上限"""

    def __init__(self, video_path, n_coarse=None, coarse_pixels=None, detail_pixels=None):
        self.vr = decord.VideoReader(video_path)
        self.fps = self.vr.get_avg_fps()
        self.duration = len(self.vr) / self.fps
        self.coarse_pixels = coarse_pixels or COARSE_PIXELS
        self.detail_pixels = detail_pixels or DETAIL_PIXELS
        self.frames = []   # dict(pil_orig, pil_disp, t, kind, note)
        n_coarse = n_coarse or N_COARSE
        for fi in np.unique(np.linspace(0, len(self.vr) - 1, n_coarse).astype(int)):
            self._register(fi / self.fps, self.vr[int(fi)].asnumpy(), 'coarse', self.coarse_pixels)

    def _register(self, t, arr_or_pil, kind, pixel_cap, note=''):
        pil = Image.fromarray(arr_or_pil) if isinstance(arr_or_pil, np.ndarray) else arr_or_pil
        dh, dw = smart_resize(pil.height, pil.width, max_pixels=pixel_cap)
        self.frames.append(dict(pil_orig=pil, pil_disp=pil.resize((dw, dh), Image.BICUBIC),
                                t=t, kind=kind, note=note))
        return len(self.frames) - 1

    def frame_label(self, i):
        f = self.frames[i]
        return f"[Frame {i}, t={f['t']:.1f}s{', ' + f['note'] if f['note'] else ''}]"

    def tool_zoom(self, args):
        """image_zoom_in_tool: 显示坐标 -> 原图坐标 -> 裁剪, 返回新帧索引列表."""
        fi = int(args['frame_index'])
        if not (0 <= fi < len(self.frames)):
            raise ValueError(f'frame_index {fi} out of range (0-{len(self.frames)-1})')
        l, t_, r, b = [float(v) for v in args['bbox_2d']]
        f = self.frames[fi]
        disp, orig = f['pil_disp'], f['pil_orig']
        sx, sy = orig.width / disp.width, orig.height / disp.height
        l, t_ = max(0, l) * sx, max(0, t_) * sy
        r, b = min(disp.width, r) * sx, min(disp.height, b) * sy
        if not (l < r and t_ < b):
            raise ValueError(f'invalid bbox {args["bbox_2d"]}')
        w, h = r - l, b - t_
        if max(w, h) / max(min(w, h), 1) > 100 or min(w, h) < 8:
            raise ValueError(f'bbox too thin/small: {args["bbox_2d"]}')
        if min(w, h) < 28:   # DeepEyes 逻辑: 最小边放大到 28px
            cx, cy, ratio = (l + r) / 2, (t_ + b) / 2, 28 / min(w, h)
            w, h = w * ratio, h * ratio
            l, t_, r, b = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        crop = orig.crop((int(l), int(t_), int(r), int(b)))
        note = f"zoom of Frame {fi}" + (f" ({args.get('label','')})" if args.get('label') else '')
        return [self._register(f['t'], crop, 'zoom', self.detail_pixels, note=note)]

    def tool_seek(self, args):
        """video_seek_tool: 时间段密集补采帧 (局部帧率提升), 返回新帧索引列表.
        num_frames 由调用方指定, 硬上限 SEEK_MAX (模型自由调用按工具 spec 限 2-6)."""
        t1 = max(0.0, float(args['start_time']))
        t2 = min(self.duration, float(args['end_time']))
        if t2 - t1 < 1e-3:
            raise ValueError(f'invalid time range [{t1:.1f}, {t2:.1f}] (video is {self.duration:.1f}s)')
        n = int(min(max(int(args.get('num_frames', REPLAY_FRAMES)), 2), SEEK_MAX))
        out = []
        for tt in np.linspace(t1, t2, n):
            fi = int(min(round(tt * self.fps), len(self.vr) - 1))
            out.append(self._register(tt, self.vr[fi].asnumpy(), 'seek', self.detail_pixels,
                                      note='dense replay'))
        return out

# ════════════════════════════════════════════════════════════════════ 4. 提示词 (DeepEyes V2 风格)

ZOOM_TOOL_JSON = '{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of one video frame by cropping it based on a bounding box (bbox_2d), to see fine details such as hands, objects, or body parts. Refer to the frame by the index shown in its label.","parameters":{"type":"object","properties":{"frame_index":{"type":"integer","description":"Index of the frame to crop, as shown in the frame label, e.g. 3 for [Frame 3]."},"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["frame_index","bbox_2d"]}}}'

SEEK_TOOL_JSON = '{"type":"function","function":{"name":"video_seek_tool","description":"Replay a time segment of the video with densely sampled frames, to inspect a fast or brief motion more closely. Use this when the key moment falls between the frames you have seen.","parameters":{"type":"object","properties":{"start_time":{"type":"number","description":"Start of the segment in seconds."},"end_time":{"type":"number","description":"End of the segment in seconds."},"num_frames":{"type":"integer","description":"Number of frames to sample in the segment (2-6, default 6)."}},"required":["start_time","end_time"]}}}'

SYS_TMPL = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

**Example**:
<tool_call>
{{"name": "image_zoom_in_tool", "arguments": {{"frame_index": 2, "bbox_2d": [10, 20, 100, 200], "label": "the right hand"}}}}
</tool_call>"""

SYS_ZOOM = SYS_TMPL.format(tools=ZOOM_TOOL_JSON)
SYS_FULL = SYS_TMPL.format(tools=ZOOM_TOOL_JSON + '\n' + SEEK_TOOL_JSON)

INSTR_FULL = ("\nThink first, call **image_zoom_in_tool** or **video_seek_tool** if needed, then answer. "
              "Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) "
              "<answer>...</answer> (answer with the option letter)")
INSTR_ZOOM = ("\nThink first, call **image_zoom_in_tool** if needed, then answer. "
              "Format strictly as: <think>...</think> <tool_call>...</tool_call> (if tools needed) "
              "<answer>...</answer> (answer with the option letter)")
CONT_PROMPT = ("Above are the frames returned by your tool call."
               " Think first, continue to call tools if needed, otherwise give your final answer."
               " Format strictly as: <think>...</think> <tool_call>...</tool_call> OR <answer>...</answer>")

# replay 模式的时间定位提示词 (与 autoresearch eval_variants.py 的 SEG_PROMPT 一致)
SEG_PROMPT = ("\nQuestion: {q}\n"
              "Do NOT answer yet. First decide which time segment of the video most likely contains "
              "the decisive visual evidence for distinguishing these candidate actions "
              "(e.g. the moment of contact, the hand motion, the stroke). "
              "Reply with ONLY the segment in this exact format: <segment>start_seconds,end_seconds</segment>")

# ════════════════════════════════════════════════════════════════════ 5. 生成 + episode

@torch.inference_mode()
def _generate(messages, max_new_tokens=None):
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors='pt').to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens or MAX_NEW_TOKENS, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

def _frames_content(sess, idx_list):
    content = []
    for i in idx_list:
        content.append({'type': 'text', 'text': sess.frame_label(i)})
        content.append({'type': 'image', 'image': sess.frames[i]['pil_disp'],
                        'min_pixels': 4*28*28, 'max_pixels': 16384*28*28})
    return content

def _extract_letter(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    seg = m.group(1) if m else text
    m2 = re.search(r'\b([ABCD])\b', seg)
    return m2.group(1) if m2 else None

def _parse_segment(text, duration):
    m = re.search(r'<segment>\s*([\d.]+)\s*,\s*([\d.]+)\s*</segment>', text)
    if not m:
        m = re.search(r'([\d.]+)\s*,\s*([\d.]+)', text)
    if m:
        t1, t2 = float(m.group(1)), float(m.group(2))
        t1, t2 = max(0.0, min(t1, duration)), max(0.0, min(t2, duration))
        if t2 - t1 >= 0.2:
            return t1, t2, True
    return duration / 3, duration * 2 / 3, False   # 解析失败回退中段

def _short(text, n=500):
    text = text.strip()
    return text if len(text) <= n else text[:n] + f' ...[截断,共{len(text)}字符]'

def _header(sess):
    return (f'The following are {sum(1 for f in sess.frames if f["kind"]=="coarse")} frames '
            f'uniformly sampled from a {sess.duration:.1f}s video. '
            f'Each frame is labeled with its index and timestamp.\n')

def run_episode(item, mode, n_coarse=None, max_turns=None,
                coarse_pixels=None, detail_pixels=None,
                max_new_tokens=None, max_frames=None,
                system_prompt=None, instruction=None, cont_prompt=None,
                verbose=False):
    """direct / zoom / full 模式单条评测. (replay 模式见 run_replay_episode)

    可调变量 (None = 用配置区默认值):
      n_coarse       初始均匀采样帧数 (视频总帧不足自动去重减少)
      max_turns      zoom/full 最多推理轮数
      coarse_pixels  初始帧分辨率上限 (像素数)
      detail_pixels  工具产生帧分辨率上限
      max_new_tokens agent 每轮生成上限 (direct 固定 DIRECT_MAX_NEW_TOKENS)
      max_frames     上下文帧总数上限; None = n_coarse + FRAME_HEADROOM (随 n_coarse 自动放大)
      system_prompt / instruction / cont_prompt  覆盖默认提示词
      verbose        打印每轮输出与工具调用, 内联显示新帧"""
    n_coarse = n_coarse or N_COARSE
    max_turns = max_turns or MAX_TURNS
    max_frames = max_frames or (n_coarse + FRAME_HEADROOM)
    video = item['question'][1]['video']
    q_full = item['question'][0]['text']
    q_body = q_full.replace('Please respond with only the letter of the correct answer.', '').strip()
    sess = VideoSession(f'{VIDEONET}/videos/{video}', n_coarse=n_coarse,
                        coarse_pixels=coarse_pixels, detail_pixels=detail_pixels)
    coarse_idx = list(range(len(sess.frames)))
    traj = dict(video=video, domain=_domain_of(item), mode=mode, gt=item['answer'],
                duration=sess.duration, turns=[], tool_calls=[], pred=None)
    t0 = time.time()

    if mode == 'direct':
        messages = [{'role': 'user', 'content':
                     [{'type': 'text', 'text': _header(sess)}] + _frames_content(sess, coarse_idx) +
                     [{'type': 'text', 'text': 'Question: ' + q_full}]}]
        resp = _generate(messages, max_new_tokens=DIRECT_MAX_NEW_TOKENS)
        traj['turns'].append(resp)
        m = re.search(r'\b([ABCD])\b', resp)
        traj['pred'] = m.group(1) if m else None
        if verbose:
            print(f'  模型输出: {_short(resp, 200)}')
    elif mode in ('zoom', 'full'):
        system = system_prompt or (SYS_ZOOM if mode == 'zoom' else SYS_FULL)
        instr = instruction or (INSTR_ZOOM if mode == 'zoom' else INSTR_FULL)
        cont = cont_prompt or CONT_PROMPT
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content':
                     [{'type': 'text', 'text': _header(sess)}] + _frames_content(sess, coarse_idx) +
                     [{'type': 'text', 'text': 'Question: ' + q_body + instr}]}]
        for turn in range(max_turns):
            resp = _generate(messages, max_new_tokens=max_new_tokens)
            traj['turns'].append(resp)
            if verbose:
                print(f'  ── 第 {turn} 轮模型输出 ──')
                print('  ' + _short(resp).replace('\n', '\n  '))
            messages.append({'role': 'assistant', 'content': resp})
            if '<answer>' in resp:
                traj['pred'] = _extract_letter(resp)
                break
            calls = re.findall(r'<tool_call>(.*?)</tool_call>', resp, re.DOTALL)
            if not calls:
                traj['pred'] = _extract_letter(resp)
                break
            obs_content = []
            for c in calls[:3]:
                rec = dict(turn=turn, raw=c.strip()[:400], ok=False, err=None, name=None)
                try:
                    call = json.loads(c.strip())
                    rec['name'] = call.get('name')
                    if len(sess.frames) >= max_frames:
                        raise ValueError('too many frames in context, please answer now')
                    if call['name'] == 'image_zoom_in_tool':
                        new_idx = sess.tool_zoom(call['arguments'])
                    elif call['name'] == 'video_seek_tool' and mode == 'full':
                        new_idx = sess.tool_seek(call['arguments'])
                    else:
                        raise ValueError(f"unknown tool {call.get('name')}")
                    rec['ok'] = True
                    obs_content += _frames_content(sess, new_idx)
                    if verbose:
                        print(f"  ✔ 工具执行成功: {call['name']} args={call['arguments']}")
                        print(f"    返回 {len(new_idx)} 帧:")
                        display(_grid([sess.frames[i]['pil_disp'] for i in new_idx], thumb=220))
                except Exception as e:
                    rec['err'] = str(e)[:200]
                    obs_content.append({'type': 'text', 'text': f'Error: {e}'})
                    if verbose:
                        print(f'  ✘ 工具执行失败: {rec["err"]}  原始调用: {rec["raw"][:120]}')
                traj['tool_calls'].append(rec)
            obs_content.append({'type': 'text', 'text': cont})
            messages.append({'role': 'user', 'content': obs_content})
        if traj['pred'] is None and traj['turns']:
            traj['pred'] = _extract_letter(traj['turns'][-1])
    else:
        raise ValueError(f'unknown mode {mode!r} (direct/zoom/full; replay 用 run_replay_episode)')

    traj['n_frames_final'] = len(sess.frames)
    traj['correct'] = (traj['pred'] == traj['gt'])
    traj['wall_time'] = round(time.time() - t0, 1)
    return traj, sess

def run_replay_episode(item, n_coarse=None, replay_frames=None,
                       coarse_pixels=None, detail_pixels=None, verbose=False):
    """replay 模式 = autoresearch 胜出协议 scaffold_revise (保守修正):
    1) 粗采帧直答记录初始答案 A1;
    2) 模型只做时间定位 (<segment>t1,t2</segment>) -> 程序化在该段密集补采 replay_frames 帧;
    3) 复核: 仅当回放证据明确矛盾时才允许改答案 (防"答案翻动").

    帧变量 (None = 配置区默认): n_coarse / replay_frames (≤SEEK_MAX) / coarse_pixels / detail_pixels"""
    n_coarse = n_coarse or N_COARSE
    replay_frames = replay_frames or REPLAY_FRAMES
    video = item['question'][1]['video']
    q_full = item['question'][0]['text']
    q_body = q_full.replace('Please respond with only the letter of the correct answer.', '').strip()
    sess = VideoSession(f'{VIDEONET}/videos/{video}', n_coarse=n_coarse,
                        coarse_pixels=coarse_pixels, detail_pixels=detail_pixels)
    coarse_idx = list(range(len(sess.frames)))
    header = _header(sess)
    traj = dict(video=video, domain=_domain_of(item), mode='replay', gt=item['answer'],
                duration=sess.duration, turns=[], tool_calls=[], pred=None)
    t0 = time.time()

    # 1) 初始直答
    resp = _generate([{'role': 'user', 'content':
                       [{'type': 'text', 'text': header}] + _frames_content(sess, coarse_idx) +
                       [{'type': 'text', 'text': 'Question: ' + q_full}]}],
                     max_new_tokens=DIRECT_MAX_NEW_TOKENS)
    traj['turns'].append(resp)
    m = re.search(r'\b([ABCD])\b', resp)
    a1 = m.group(1) if m else None

    # 2) 时间定位 + 程序化回放
    seg_resp = _generate([{'role': 'user', 'content':
                           [{'type': 'text', 'text': header}] + _frames_content(sess, coarse_idx) +
                           [{'type': 'text', 'text': SEG_PROMPT.format(q=q_body)}]}],
                         max_new_tokens=DIRECT_MAX_NEW_TOKENS)
    traj['turns'].append(seg_resp)
    t1, t2, parsed = _parse_segment(seg_resp, sess.duration)
    traj['tool_calls'].append(dict(turn=1, raw=f'replay [{t1:.1f},{t2:.1f}] parsed={parsed}',
                                   ok=True, err=None, name='video_seek_tool'))
    new_idx = sess.tool_seek({'start_time': t1, 'end_time': t2, 'num_frames': replay_frames})
    if verbose:
        print(f'  初始答案 A1={a1}; 定位段 [{t1:.1f},{t2:.1f}]s (parsed={parsed})'
              f' -> 回放 {len(new_idx)} 帧:')
        display(_grid([sess.frames[i]['pil_disp'] for i in new_idx], thumb=170))

    # 3) 保守复核
    instr = (f"\nYour preliminary answer from the sparse frames was: {a1 or 'unknown'}. "
             "Now verify it against the dense replay above. KEEP your preliminary answer unless the "
             "replay clearly shows evidence that contradicts it and clearly supports a different option. "
             "Think first, then answer. Format strictly as: <think>...</think> <answer>letter</answer>")
    final = _generate([{'role': 'user', 'content':
                        [{'type': 'text', 'text': header}] + _frames_content(sess, coarse_idx) +
                        [{'type': 'text', 'text': f'Dense replay of the key segment [{t1:.1f}s - {t2:.1f}s]:'}] +
                        _frames_content(sess, new_idx) +
                        [{'type': 'text', 'text': 'Question: ' + q_body + instr}]}])
    traj['turns'].append(final)
    if verbose:
        print('  复核输出: ' + _short(final, 300))
    traj['pred'] = _extract_letter(final) or a1
    traj['n_frames_final'] = len(sess.frames)
    traj['correct'] = (traj['pred'] == traj['gt'])
    traj['wall_time'] = round(time.time() - t0, 1)
    return traj, sess

# run_episode 与 run_replay_episode 各自接受的参数名 (run_all 按模式分发)
_EPISODE_KEYS = {'n_coarse', 'max_turns', 'coarse_pixels', 'detail_pixels',
                 'max_new_tokens', 'max_frames', 'system_prompt', 'instruction', 'cont_prompt'}
_REPLAY_KEYS = {'n_coarse', 'replay_frames', 'coarse_pixels', 'detail_pixels'}

def run_all(items, modes=('direct', 'zoom', 'full', 'replay'),
            verbose_agent=True, verbose_first=3, **episode_kwargs):
    """跑全部模式 x 全部样本, 返回 records 列表并存 results/probe_records.json.

    modes           要跑的模式子集
    verbose_agent   agent 模式是否打印详细轨迹
    verbose_first   每模式只对前 N 条打完整轨迹+帧图, 其余一行摘要 (样本多时防 notebook 膨胀)
    **episode_kwargs 变量透传 (按模式自动分发, 各变量含义见 run_episode / run_replay_episode):
        n_coarse / replay_frames / coarse_pixels / detail_pixels /
        max_turns / max_new_tokens / max_frames / system_prompt / instruction / cont_prompt"""
    unknown = set(episode_kwargs) - _EPISODE_KEYS - _REPLAY_KEYS
    if unknown:
        raise TypeError(f'未知参数: {unknown}; 可用: {sorted(_EPISODE_KEYS | _REPLAY_KEYS)}')
    records = []
    n_coarse = episode_kwargs.get('n_coarse') or N_COARSE
    replay_frames = episode_kwargs.get('replay_frames') or REPLAY_FRAMES
    for mode in modes:
        knobs = f'[初始均匀采样 {n_coarse} 帧' + (f', 回放补采 {replay_frames} 帧' if mode == 'replay' else '') + ']'
        print(f'\n{"="*70}\n{MODE_DESC.get(mode, mode)}  {knobs}\n{"="*70}')
        for k, item in enumerate(items):
            verbose = verbose_agent and mode != 'direct' and k < verbose_first
            print(f'\n[{k+1}/{len(items)}] {_domain_of(item)} | {item["question"][1]["video"][:8]} '
                  f'| gt={item["answer"]}')
            if mode == 'replay':
                kw = {k2: v for k2, v in episode_kwargs.items() if k2 in _REPLAY_KEYS}
                traj, _ = run_replay_episode(item, verbose=verbose, **kw)
            else:
                kw = {k2: v for k2, v in episode_kwargs.items() if k2 in _EPISODE_KEYS}
                traj, _ = run_episode(item, mode, verbose=verbose, **kw)
            records.append(traj)
            mark = '✓' if traj['correct'] else '✗'
            n_ok = sum(1 for c in traj['tool_calls'] if c['ok'])
            print(f'  => pred={traj["pred"]} {mark} | 轮数={len(traj["turns"])} '
                  f'工具调用={len(traj["tool_calls"])}(成功{n_ok}) | {traj["wall_time"]}s')
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f'{OUT_DIR}/probe_records.json', 'w') as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    print(f'\n轨迹已保存: {OUT_DIR}/probe_records.json')
    return records

# ════════════════════════════════════════════════════════════════════ 6. 统计

def summarize(records):
    import pandas as pd
    df = pd.DataFrame(records)
    modes = list(dict.fromkeys(df['mode']))

    def stats(sub):
        calls = [c for r in sub.itertuples() for c in r.tool_calls]
        zoom = [c for c in calls if c['name'] == 'image_zoom_in_tool']
        seek = [c for c in calls if c['name'] == 'video_seek_tool']
        return pd.Series({
            '准确率': f"{sub['correct'].mean():.0%} ({sub['correct'].sum()}/{len(sub)})",
            '平均轮数': round(sub['turns'].str.len().mean(), 1),
            '调工具episode数': int((sub['tool_calls'].str.len() > 0).sum()),
            'zoom调用(成功)': f"{len(zoom)}({sum(c['ok'] for c in zoom)})",
            'seek/回放(成功)': f"{len(seek)}({sum(c['ok'] for c in seek)})",
            '平均耗时s': round(sub['wall_time'].mean(), 1),
        })

    print('════════ 1) 总表 ════════')
    summary = df.groupby('mode', sort=False).apply(stats, include_groups=False).reindex(modes)
    print(summary.to_string())

    print('\n════════ 2) 逐条对比 (哪些题被工具翻对/翻错) ════════')
    by = {}
    for r in records:
        by.setdefault((r['video'][:8], r['domain'], r['gt']), {})[r['mode']] = r
    print(f"{'video':10s} {'domain':20s} gt  " + '  '.join(f'{m:7s}' for m in modes))
    for (v, dom, gt), d in by.items():
        row = '  '.join((f"{d[m]['pred'] or '-'}{'✓' if d[m]['correct'] else '✗':6s}") if m in d else '-      '
                        for m in modes)
        print(f'{v:10s} {dom:20s} {gt}   {row}')

    agent = [r for r in records if r['mode'] not in ('direct',)]
    print('\n════════ 3) 弱点信号 ════════')
    fi = {}
    for r in agent:
        for c in r['tool_calls']:
            m = re.search(r'"frame_index":\s*(\d+)', c['raw'])
            if m:
                fi[m.group(1)] = fi.get(m.group(1), 0) + 1
    print(f'a. zoom 的 frame_index 分布: {fi}   <- 若塌缩在中间帧,说明没有真正的时间定位')
    pre = sum(1 for r in agent
              if r['turns'] and re.search(r'answer is|correct answer', r['turns'][0].split('</think>')[0]))
    print(f'b. 首轮 <think> 里先下结论再调工具: {pre}/{len(agent)} 个 episode   <- 仪式性调用信号')
    deg = sum(1 for r in agent for t in r['turns'] if 'addCriterion' in t)
    print(f'c. 出现退化输出 (addCriterion 等) 的轮数: {deg}   <- 多发生在尝试 seek/多调用时')
    errs = {}
    for r in agent:
        for c in r['tool_calls']:
            if c['err']:
                k = c['err'][:60]
                errs[k] = errs.get(k, 0) + 1
    print(f'd. 工具执行错误: {errs if errs else "无"}')
    labels = [m.group(1) for r in agent for c in r['tool_calls']
              if (m := re.search(r'"label":\s*"([^"]+)"', c['raw']))]
    print(f'e. zoom 的 label (动作假设名 vs 可 ground 的物体): {labels[:24]}')
    return summary
