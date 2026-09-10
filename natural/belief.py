"""与模型无关的 MCQ 信念读出.

信念 b(option) = 四个选项字母首 token 的 softmax, 对选项顺序做 n_perm 个循环置换取平均 (去字母位置偏置).
judge-free、确定性、比 acc 灵敏 —— 这是整条流水线的测量原语.

逻辑照搬自 data/evidencegap_v_pilot/eval_deepeyes.py (belief_and_greedy / _letter_ids), 但抽成
与模型无关, 且**不** import 那个模块 —— 它在 import 时会改写 video_imcot 的全局 (VIDEONET / _header /
frame_label / _generate), 会让本流水线读错目录.

后端:
  deepeyes  DeepEyes-7B (Qwen2.5-VL-7B)          conda env: ms_swift
  qwen38    Qwen3.8-27B (Qwen3_5, 强模型认证者)   conda env: verl_qwen35
  qwen36    Qwen3.6-27B                          conda env: verl_qwen35
"""
import logging, os, time
import numpy as np
import torch
import transformers

transformers.logging.set_verbosity_error()          # 关掉每次 generate 都刷的 pad_token_id 提示
transformers.utils.logging.disable_progress_bar()   # 关掉权重加载进度条
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

LETTERS = 'ABCD'

MODELS = {
    'deepeyes': '/mnt/afs/panzhengyuan/llm_training/models/DeepEyes-7B',
    'qwen38':   '/mnt/afs/panzhengyuan/llm_training/models/pretrained/Qwen/Qwen3.8-27B',
    'qwen36':   '/mnt/afs/panzhengyuan/llm_training/models/pretrained/Qwen/Qwen3.6-27B',
    'qwen35':   '/mnt/afs/panzhengyuan/llm_training/models/pretrained/Qwen/Qwen3.5-9B',
    'zoom_s1':  '/mnt/afs/panzhengyuan/llm_training/models/merged/qwen35-9b-zoom-s1',
}

ZH = dict(
    header=lambda n, d: f'以下是从一段 {d:.1f} 秒的视频中均匀抽取的 {n} 帧画面，每帧都标注了序号和时间戳。\n',
    label=lambda i, t, note: f'[第{i}帧, t={t:.1f}s{("，" + note) if note else ""}]',
    question='问题：', tail='\n请只回答正确选项的字母。',
    evidence=lambda a, b, extra: f'以下是视频检视工具返回的补充画面（时间窗 {a:.1f}s–{b:.1f}s{extra}）：',
)
EN = dict(
    header=lambda n, d: f'The following are {n} frames uniformly sampled from a {d:.1f}s video. '
                        'Each frame is labeled with its index and timestamp.\n',
    label=lambda i, t, note: f'[Frame {i}, t={t:.1f}s{(", " + note) if note else ""}]',
    question='Question: ', tail='\nPlease respond with only the letter of the correct answer.',
    evidence=lambda a, b, extra: f'Additional frames returned by the video inspection tool '
                                 f'(time window {a:.1f}s-{b:.1f}s{extra}):',
)
LANG = {'zh': ZH, 'en': EN}


class Backend:
    """加载一个 VLM 并提供 belief() / generate()."""

    def __init__(self, name, device=0, dtype=torch.bfloat16, attn='flash_attention_2'):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        path = MODELS[name]
        t0 = time.time()
        kw = dict(dtype=dtype, device_map={'': device})
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(path, attn_implementation=attn, **kw)
        except Exception as e:
            print(f'[{name}] attn={attn} 失败 ({type(e).__name__}: {str(e)[:120]}), 退回默认 attention')
            self.model = AutoModelForImageTextToText.from_pretrained(path, **kw)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(path)
        self.name = name
        self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id
        self.letter_ids = self._letter_ids()
        # Qwen3.5 系模板预开 <think>: 第一个生成位置是思考内容不是答案字母, 信念读出会变成均匀分布 (H=ln4).
        # 读信念时用 enable_thinking=False 把 think 块关上 (渲染成 <think>\n\n</think>\n\n), 答案位置才对.
        probe = self.processor.apply_chat_template([{'role': 'user', 'content': 'x'}],
                                                   tokenize=False, add_generation_prompt=True)
        self.thinking_template = probe.rstrip().endswith('<think>')
        print(f'[{name}] 加载完成 {time.time()-t0:.0f}s  device={self.model.device}  '
              f'thinking_template={self.thinking_template}  '
              f'letter_ids={ {k: v for k, v in self.letter_ids.items()} }')

    def _letter_ids(self):
        tok = self.processor.tokenizer
        ids = {}
        for L in LETTERS:
            s = set()
            for v in (L, ' ' + L):
                e = tok.encode(v, add_special_tokens=False)
                if len(e) == 1:
                    s.add(e[0])
            if not s:
                raise RuntimeError(f'{self.name}: 选项字母 {L!r} 不是单 token, 信念读出不适用')
            ids[L] = sorted(s)
        return ids

    # ---------- 底层 ----------

    def _prep(self, content, no_think=True):
        """content: [{'type':'text','text':...} | {'type':'image','image':PIL}] -> model inputs.
        no_think=True 时对预开 <think> 的模板关掉思考 (信念读出用); agentic 生成走 pipeline._gen 不经过这里."""
        msgs = [{'role': 'user', 'content': content}]
        kw = {'enable_thinking': False} if (no_think and self.thinking_template) else {}
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        images = [c['image'] for c in content if c['type'] == 'image']
        return self.processor(text=[text], images=images or None, padding=True,
                              return_tensors='pt').to(self.model.device)

    @torch.inference_mode()
    def _first_token_logits(self, content):
        inputs = self._prep(content)
        out = self.model(**inputs)
        return out.logits[0, -1].float()

    @torch.inference_mode()
    def generate(self, content, max_new_tokens=512, stop=None):
        inputs = self._prep(content)
        kw = {}
        if stop:
            kw = dict(stop_strings=list(stop), tokenizer=self.processor.tokenizer)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **kw)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                           skip_special_tokens=True)[0]

    # ---------- 信念 ----------

    def belief(self, prefix, stem, options, n_perm=4, lang='zh'):
        """prefix: content 列表 (帧和说明文字); 返回 dict(p=[4], pred_idx, entropy, per_perm)

        p[i] = 原始顺序下第 i 个选项的概率, 已对 n_perm 个循环置换取平均.
        """
        L = LANG[lang]
        per, greedy = [], []
        for shift in range(n_perm):
            perm = [(j + shift) % 4 for j in range(4)]      # 展示位置 j 上放原选项 perm[j]
            q = stem + '\n' + '\n'.join(f'{c}. {options[i]}' for c, i in zip(LETTERS, perm)) + L['tail']
            content = list(prefix) + [{'type': 'text', 'text': L['question'] + q}]
            logits = self._first_token_logits(content)
            lp = torch.log_softmax(logits, -1)
            z = np.array([torch.logsumexp(lp[self.letter_ids[c]], 0).item() for c in LETTERS])
            z = np.exp(z - z.max()); z /= z.sum()
            b = np.zeros(4)
            for j, i in enumerate(perm):
                b[i] = z[j]
            per.append(b)
            greedy.append(int(perm[int(np.argmax(z))]))
        p = np.mean(per, 0)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        return dict(p=[round(float(v), 4) for v in p], pred_idx=int(np.argmax(p)),
                    entropy=round(ent, 4),
                    perm_std=round(float(np.std([b for b in per], axis=0).max()), 4),
                    greedy=greedy)


# ---------- prefix 构造 ----------

def frames_prefix(pils, indices, fps, duration, lang='zh', note='', start_at=0):
    """把帧列表变成带标签的 content. indices 用于时间戳, note 标在标签里 (如 '密集回放')."""
    L = LANG[lang]
    out = [{'type': 'text', 'text': L['header'](len(pils), duration)}]
    for j, (pil, fi) in enumerate(zip(pils, indices)):
        out.append({'type': 'text', 'text': L['label'](start_at + j, fi / fps, note)})
        out.append({'type': 'image', 'image': pil})
    return out


def evidence_prefix(pils, indices, fps, lang='zh', bbox=None, start_at=0):
    """工具返回帧的 content 片段 (接在 coarse 之后)."""
    L = LANG[lang]
    t0, t1 = indices[0] / fps, indices[-1] / fps
    extra = f'，裁剪区域 {list(bbox)}' if (bbox is not None and lang == 'zh') else \
            (f', cropped to region {list(bbox)}' if bbox is not None else '')
    out = [{'type': 'text', 'text': L['evidence'](t0, t1, extra)}]
    note = '工具返回' if lang == 'zh' else 'tool-returned'
    for j, (pil, fi) in enumerate(zip(pils, indices)):
        out.append({'type': 'text', 'text': L['label'](start_at + j, fi / fps, note)})
        out.append({'type': 'image', 'image': pil})
    return out
