"""agentic 的中文提示词 (工具说明 / 指令 / 续轮), 与 pilot 里那套一致."""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
import video_imcot as vi


def _tools_zh():
    zoom = json.loads(vi.ZOOM_TOOL_JSON)
    seek = json.loads(vi.SEEK_TOOL_JSON)
    zf, sf = zoom['function'], seek['function']
    zf['description'] = ('通过边界框 (bbox_2d) 裁剪并放大某一视频帧的指定区域，用于查看手部、物体或身体部位等细节。'
                         '用帧标签中显示的序号来指定帧。')
    zp = zf['parameters']['properties']
    zp['frame_index']['description'] = '要裁剪的帧的序号，即帧标签中显示的数字，例如 [第3帧] 对应 3。'
    zp['bbox_2d']['description'] = '要放大的区域的边界框 [x1, y1, x2, y2]，(x1, y1) 为左上角，(x2, y2) 为右下角。'
    zp['label']['description'] = '该区域内物体的名称或标签（可选）。'
    sf['description'] = '以更密集的采样帧回放视频中的一个时间段，用于仔细查看快速或短暂的动作。当关键时刻落在你已看到的帧之间时使用。'
    sp = sf['parameters']['properties']
    sp['start_time']['description'] = '时间段起点（秒）。'
    sp['end_time']['description'] = '时间段终点（秒）。'
    sp['num_frames']['description'] = '在该时间段内采样的帧数（2-6，默认 6）。'
    return json.dumps(zoom, ensure_ascii=False) + '\n' + json.dumps(seek, ensure_ascii=False)


SYS_ZH = """你是一个乐于助人的助手。

# 工具
你可以调用一个或多个函数来协助回答用户的问题。
可用的函数签名在 <tools></tools> XML 标签内：
<tools>
%s
</tools>

# 如何调用工具
在 <tool_call></tool_call> XML 标签内返回一个包含函数名和参数的 json 对象：
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**示例**：
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"frame_index": 2, "bbox_2d": [10, 20, 100, 200], "label": "右手"}}
</tool_call>""" % _tools_zh()

INSTR_ZH = ('\n请先用中文思考；如有需要，调用 **image_zoom_in_tool**（放大某一帧的某个区域）或 '
            '**video_seek_tool**（密集回看某个时间段），然后作答。'
            '格式严格为：<think>...</think> <tool_call>...</tool_call>（需要工具时）<answer>...</answer>（只填选项字母）')

CONT_ZH = ('以上是你调用工具后返回的画面。请继续用中文思考；如有需要可继续调用工具，否则给出最终答案。'
           '格式严格为：<think>...</think> <tool_call>...</tool_call> 或 <answer>...</answer>')

PRE_ANSWER_RE = (r'answer is|correct answer|答案是|答案应该是|正确答案|应该选|应选|'
                 r'选项\s*[ABCD]\b|是\s*[ABCD]\b')
