"""阿里云百炼语音识别（支持 Fun-ASR / Qwen-Audio-3.1-ASR-Flash 等模型，含说话人分离）。

流程：
1. 将本地 MP3 上传到百炼临时存储，获得 48 小时有效、可被百炼访问的 oss:// URL。
2. 根据模型类型调用对应接口：
   - 同步多模态识别（如 qwen-audio-3.1-asr-flash）：调用 multimodal-generation/generation，
     开启 speaker_diarization_enabled=True，解析 output.sentences / output.sentence。
   - 异步文件转写（如 fun-asr-*、qwen-audio-3.1-asr-flash-filetrans）：调用 audio/asr/transcription，
     开启 diarization_enabled=True，轮询任务状态并下载转写 JSON。
3. 按说话人(speaker_id)整理为对话片段。

API Key 通过环境变量 DASHSCOPE_API_KEY 读取。
"""
import base64
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import imageio_ffmpeg
import requests

MODEL = "fun-asr-2025-11-07"
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

# Flash 同步模型单次请求限制：时长 ≤ 5 分钟、Base64 体积 ≤ 10MB
# 这里按 240 秒（4 分钟）切片并转码为 16kHz 单声道 48kbps MP3（每片约 1.4MB）
FLASH_CHUNK_SECONDS = 240


def _normalize_base_url(url: str) -> str:
    raw = (url or "https://dashscope.aliyuncs.com").strip().rstrip("/")
    for marker in ("/api/v1", "/compatible-mode/v1", "/compatible-mode"):
        idx = raw.find(marker)
        if idx != -1:
            raw = raw[:idx]
    parsed = urlsplit(raw)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return raw or "https://dashscope.aliyuncs.com"


def _base_url() -> str:
    return _normalize_base_url(
        os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("BASE_URL")
        or "https://dashscope.aliyuncs.com"
    )


DASHSCOPE_BASE = _base_url()


def _is_sync_flash_model(model: str) -> bool:
    """判断是否为走 multimodal-generation 同步接口的 Flash ASR 模型（不含 -filetrans）。"""
    m = (model or "").strip().lower()
    if m.endswith("-filetrans"):
        return False
    return (
        m.startswith("qwen-audio-") and "asr-flash" in m
    ) or m.startswith("fun-asr-flash") or m.startswith("qwen3-asr-flash")


class ASRError(Exception):
    pass


def _api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise ASRError("未配置 DASHSCOPE_API_KEY 环境变量")
    return key


def _upload_endpoints() -> list[str]:
    """返回候选的 /api/v1/uploads 地址列表（兼容 MaaS 专属域名无 /api/v1/uploads 的情况）。"""
    base = _base_url()
    candidates = [f"{base}/api/v1/uploads"]
    fallback_host = (
        "https://dashscope-intl.aliyuncs.com"
        if ("ap-southeast" in base or "intl" in base)
        else "https://dashscope.aliyuncs.com"
    )
    fallback_url = f"{fallback_host}/api/v1/uploads"
    if fallback_url not in candidates:
        candidates.append(fallback_url)
    return candidates


def upload_to_dashscope(file_path: str, api_key: str | None = None, model: str = MODEL) -> str:
    """上传文件到百炼临时存储，返回 oss:// 临时 URL（48 小时有效）。

    注意：上传凭证与模型绑定；若自定义 MaaS 域名无 /api/v1/uploads（返回 404），
    自动回退到 dashscope.aliyuncs.com 的上传网关。
    """
    api_key = api_key or _api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # 步骤1：获取上传凭证（兼容自定义域名 404 回退及新模型名回退）
    resp = None
    for upload_url in _upload_endpoints():
        for policy_model in (model, MODEL):
            resp = requests.get(
                upload_url,
                headers=headers,
                params={"action": "getPolicy", "model": policy_model},
                timeout=30,
            )
            if resp.status_code == 200 and resp.json().get("data"):
                break
            if resp.status_code == 404:
                # 当前域名不存在 /api/v1/uploads 端点，换下一个候选域名
                break
        if resp is not None and resp.status_code == 200 and resp.json().get("data"):
            break

    if resp is None or resp.status_code != 200 or not resp.json().get("data"):
        code = resp.status_code if resp is not None else "N/A"
        body_text = (resp.text[:300] if resp is not None else "").strip()
        raise ASRError(f"获取上传凭证失败: {code} {body_text}")
    policy = resp.json()["data"]

    # 步骤2：上传文件到 OSS 临时存储
    file_name = Path(file_path).name
    key = f"{policy['upload_dir']}/{file_name}"
    with open(file_path, "rb") as fh:
        files = {
            "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
            "Signature": (None, policy["signature"]),
            "policy": (None, policy["policy"]),
            "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
            "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]),
            "key": (None, key),
            "success_action_status": (None, "200"),
            "file": (file_name, fh),
        }
        up = requests.post(policy["upload_host"], files=files, timeout=300)
    if up.status_code != 200:
        raise ASRError(f"上传文件到临时存储失败: {up.status_code} {up.text[:300]}")

    # 步骤3：拼接 oss:// URL
    return f"oss://{key}"


def submit_transcription(
    audio_url: str, api_key: str | None = None, speaker_count: int | None = None,
    model: str = MODEL,
) -> str:
    """提交异步转写任务，开启说话人分离，返回 task_id。"""
    api_key = api_key or _api_key()
    submit_url = f"{_base_url()}/api/v1/services/audio/asr/transcription"

    parameters = {"diarization_enabled": True}
    if speaker_count and 2 <= speaker_count <= 100:
        parameters["speaker_count"] = speaker_count

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    # oss:// 临时 URL 需要该头，百炼才能解析内部协议
    if audio_url.startswith("oss://"):
        headers["X-DashScope-OssResourceResolve"] = "enable"

    body = {
        "model": model,
        "input": {"file_urls": [audio_url]},
        "parameters": parameters,
    }
    resp = requests.post(submit_url, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        raise ASRError(f"提交转写任务失败: {resp.status_code} {resp.text[:300]}")
    out = resp.json().get("output", {})
    task_id = out.get("task_id")
    if not task_id:
        raise ASRError(f"未返回 task_id: {resp.text[:300]}")
    return task_id


def poll_task(
    task_id: str, api_key: str | None = None, timeout: int = 300, interval: float = 3.0
) -> dict:
    """轮询任务状态直至终态，返回 output 对象。"""
    api_key = api_key or _api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{_base_url()}/api/v1/tasks/{task_id}"
    deadline = time.time() + timeout

    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise ASRError(f"查询任务失败: {resp.status_code} {resp.text[:300]}")
        out = resp.json().get("output", {})
        status = out.get("task_status")
        if status in ("SUCCEEDED", "FAILED", "CANCELED"):
            return out
        if time.time() > deadline:
            raise ASRError(f"任务轮询超时（{timeout}s），当前状态: {status}")
        time.sleep(interval)


def fetch_transcript(transcription_url: str) -> dict:
    """下载转写结果 JSON。"""
    resp = requests.get(transcription_url, timeout=60)
    if resp.status_code != 200:
        raise ASRError(f"下载转写结果失败: {resp.status_code}")
    return resp.json()


# 单个对话气泡合并阈值：限制单一说话人单气泡的最大字数、最大时长和句间停顿
MAX_SEGMENT_CHARS = 128        # 单个气泡最大字符数（128 字）
MAX_SEGMENT_DURATION_MS = 30000  # 单个气泡建议最大时长（30 秒）
MAX_PAUSE_GAP_MS = 1200        # 句间停顿超过 1.2 秒则新起一个气泡

_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，,、：:])")


def _fmt_ts(ms: int) -> str:
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _split_long_sentence_item(item: dict, max_chars: int = MAX_SEGMENT_CHARS) -> list[dict]:
    """当模型返回的单条句子本身过长时，按词级时间戳(words)或标点切分为多个短句。"""
    text = (item.get("text") or "").strip()
    if len(text) <= max_chars:
        return [item]

    spk = item.get("speaker", 0)
    begin_ms = int(item.get("begin_time") or 0)
    end_ms = max(begin_ms, int(item.get("end_time") or begin_ms))
    total_dur = max(0, end_ms - begin_ms)

    # 1. 优先使用模型返回的词级时间戳（words）切分，保证每个短句时间戳精准
    words = item.get("words")
    if isinstance(words, list) and words:
        sub_items: list[dict] = []
        buf_text = ""
        buf_begin: int | None = None
        buf_end: int = begin_ms
        for w in words:
            if not isinstance(w, dict):
                continue
            w_text = f"{w.get('text') or ''}{w.get('punctuation') or ''}"
            if not w_text:
                continue
            w_b = int(w.get("begin_time") if w.get("begin_time") is not None else buf_end)
            w_e = int(w.get("end_time") if w.get("end_time") is not None else w_b)
            if buf_begin is None:
                buf_begin = w_b
            buf_text += w_text
            buf_end = max(buf_end, w_e)

            is_sent_end = w_text.endswith(("。", "！", "？", "!", "?", "；", ";"))
            is_clause_end = w_text.endswith(("，", ",", "、", "：", ":"))
            if (len(buf_text) >= 64 and is_sent_end) or (len(buf_text) >= 100 and is_clause_end) or len(buf_text) >= max_chars:
                sub_items.append({
                    "speaker": spk,
                    "begin_time": buf_begin,
                    "end_time": buf_end,
                    "text": buf_text.strip(),
                })
                buf_text = ""
                buf_begin = None
        if buf_text.strip():
            sub_items.append({
                "speaker": spk,
                "begin_time": buf_begin if buf_begin is not None else begin_ms,
                "end_time": max(buf_end, end_ms),
                "text": buf_text.strip(),
            })
        if sub_items:
            return sub_items

    # 2. 若无词级时间戳，按句末标点（。！？；）及逗号（，、）切分，并按字数比例插值估算时间戳
    raw_parts = [p for p in _SENT_SPLIT_RE.split(text) if p]
    fine_parts: list[str] = []
    for part in raw_parts:
        if len(part) <= max_chars:
            fine_parts.append(part)
        else:
            for c in _CLAUSE_SPLIT_RE.split(part):
                if not c:
                    continue
                while len(c) > max_chars:
                    fine_parts.append(c[:max_chars])
                    c = c[max_chars:]
                if c:
                    fine_parts.append(c)

    chunks: list[str] = []
    cur = ""
    for piece in fine_parts:
        if cur and len(cur) + len(piece) > max_chars:
            chunks.append(cur.strip())
            cur = piece
        else:
            cur += piece
    if cur.strip():
        chunks.append(cur.strip())

    if len(chunks) <= 1:
        return [item]

    total_chars = max(1, sum(len(c) for c in chunks))
    result: list[dict] = []
    acc_chars = 0
    for idx, ch in enumerate(chunks):
        c_begin = begin_ms + int(round(total_dur * acc_chars / total_chars))
        acc_chars += len(ch)
        c_end = end_ms if idx == len(chunks) - 1 else (begin_ms + int(round(total_dur * acc_chars / total_chars)))
        result.append({
            "speaker": spk,
            "begin_time": c_begin,
            "end_time": max(c_begin, c_end),
            "text": ch,
        })
    return result


def _should_merge_segment(prev: dict, curr: dict) -> bool:
    """判断同一说话人的相邻两个短句是否合并进同一个气泡。"""
    if prev["speaker"] != curr["speaker"]:
        return False
    prev_text = prev["text"]
    # 若上一句未以句末标点结尾，且合并后不超长，优先合并保持语义完整
    ends_with_stop = prev_text.endswith(("。", "！", "？", "!", "?", "；", ";", "…"))
    gap_ms = curr["begin_time"] - prev["end_time"]
    merged_chars = len(prev_text) + len(curr["text"])
    merged_dur = max(prev["end_time"], curr["end_time"]) - prev["begin_time"]

    if gap_ms > MAX_PAUSE_GAP_MS:
        return False
    if merged_chars > MAX_SEGMENT_CHARS or merged_dur > MAX_SEGMENT_DURATION_MS:
        return False
    # 如果上一句已经是完整句子（带句号/问号）且已有一定长度（>= 64 字），自然分段更易读
    if ends_with_stop and len(prev_text) >= 64:
        return False
    return True


def _build_grouped_from_sentences(sentences: list[dict]) -> dict:
    """将统一格式的 [{speaker, begin_time, end_time, text, words?}, ...] 整理为适度长度的说话人对话气泡。"""
    cleaned = []
    for s in sentences:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spk = s.get("speaker")
        if spk is None:
            spk = 0
        try:
            spk = int(spk)
        except (TypeError, ValueError):
            spk = 0
        base_item = {
            "speaker": spk,
            "begin_time": int(s.get("begin_time") or 0),
            "end_time": int(s.get("end_time") or 0),
            "text": text,
            "words": s.get("words"),
        }
        cleaned.extend(_split_long_sentence_item(base_item))

    cleaned.sort(key=lambda x: x["begin_time"])

    # 合并连续同一说话人的短句，遇到超长/长停顿/完整句段则自动切分为新段落气泡
    segments = []
    for s in cleaned:
        item = {
            "speaker": s["speaker"],
            "begin_time": s["begin_time"],
            "end_time": s["end_time"],
            "text": s["text"],
        }
        if segments and _should_merge_segment(segments[-1], item):
            if not segments[-1]["text"].endswith(("。", "！", "？", "!", "?", "，", ",", "、", "；", ";", "：", ":")):
                segments[-1]["text"] += "，"
            segments[-1]["text"] += item["text"]
            segments[-1]["end_time"] = max(segments[-1]["end_time"], item["end_time"])
        else:
            segments.append(item)

    speakers: dict[str, str] = {}
    for s in cleaned:
        speakers.setdefault(str(s["speaker"]), "")
        speakers[str(s["speaker"])] += s["text"]

    return {
        "segments": [
            {
                "speaker": seg["speaker"],
                "begin": _fmt_ts(seg["begin_time"]),
                "end": _fmt_ts(seg["end_time"]),
                "begin_ms": seg["begin_time"],
                "end_ms": seg["end_time"],
                "text": seg["text"],
            }
            for seg in segments
        ],
        "speakers": speakers,
        "speaker_count": len(speakers),
    }


def group_by_speaker(transcript: dict) -> dict:
    """把异步转写结果整理为按说话人分离的对话片段。"""
    sentences = []
    for tr in transcript.get("transcripts", []):
        for st in tr.get("sentences", []):
            sentences.append(
                {
                    "speaker": st.get("speaker_id", 0),
                    "begin_time": st.get("begin_time", 0),
                    "end_time": st.get("end_time", 0),
                    "text": (st.get("text") or "").strip(),
                    "words": st.get("words"),
                }
            )
    return _build_grouped_from_sentences(sentences)


def _get_audio_duration_seconds(file_path: str) -> float:
    """使用 ffmpeg 探测音频时长（秒）。"""
    try:
        res = subprocess.run(
            [FFMPEG_BIN, "-i", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr or "")
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return 0.0


def _call_multimodal_chunk(
    data_uri_or_url: str,
    api_key: str,
    model: str = "qwen-audio-3.1-asr-flash",
    audio_format: str = "mp3",
    offset_ms: int = 0,
) -> tuple[list[dict], str]:
    """调用单段音频的 multimodal-generation 接口并解析句子列表。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "disable",
    }
    if data_uri_or_url.startswith("oss://"):
        headers["X-DashScope-OssResourceResolve"] = "enable"

    parameters: dict = {
        "format": audio_format,
        "sample_rate": "16000",
        "speaker_diarization_enabled": True,
    }

    body = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data_uri_or_url},
                        }
                    ],
                }
            ]
        },
        "parameters": parameters,
    }

    base = _base_url()
    candidate_bases = [base]
    for fallback in ("https://maas.qianwenaiapi.com", "https://dashscope.aliyuncs.com"):
        if fallback not in candidate_bases:
            candidate_bases.append(fallback)

    resp = None
    for b in candidate_bases:
        url = f"{b}/api/v1/services/aigc/multimodal-generation/generation"
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=300)
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            break
        if resp.status_code not in (404, 400):
            break

    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "N/A"
        txt = (resp.text[:300] if resp is not None else "").strip()
        raise ASRError(f"调用 {model} 识别失败: {code} {txt}")

    data = resp.json()
    raw_out = data.get("output") or {}
    # 兼容 output 下可能再嵌套一层 output 的情况
    inner_out = raw_out.get("output") if isinstance(raw_out.get("output"), dict) else raw_out

    raw_sentences = (
        inner_out.get("sentences")
        or raw_out.get("sentences")
        or []
    )
    if not raw_sentences:
        single = inner_out.get("sentence") or raw_out.get("sentence")
        if isinstance(single, dict):
            raw_sentences = [single]

    sentences = []
    for st in raw_sentences:
        if not isinstance(st, dict):
            continue
        spk = st.get("speaker_id")
        if spk is None and isinstance(st.get("words"), list):
            for w in st["words"]:
                if isinstance(w, dict) and w.get("speaker_id") is not None:
                    spk = w.get("speaker_id")
                    break
        begin_ms = int(st.get("begin_time") or 0) + offset_ms
        end_ms = int(st.get("end_time") or 0) + offset_ms
        shifted_words = None
        if isinstance(st.get("words"), list):
            shifted_words = []
            for w in st["words"]:
                if isinstance(w, dict):
                    wc = dict(w)
                    if wc.get("begin_time") is not None:
                        wc["begin_time"] = int(wc["begin_time"]) + offset_ms
                    if wc.get("end_time") is not None:
                        wc["end_time"] = int(wc["end_time"]) + offset_ms
                    shifted_words.append(wc)
        sentences.append(
            {
                "speaker": 0 if spk is None else spk,
                "begin_time": begin_ms,
                "end_time": end_ms,
                "text": (st.get("text") or "").strip(),
                "words": shifted_words,
            }
        )

    # 兜底：若仅返回顶层 output.text 而无 sentences
    if not sentences:
        full_text = (inner_out.get("text") or raw_out.get("text") or "").strip()
        if full_text:
            sentences.append({"speaker": 0, "begin_time": offset_ms, "end_time": offset_ms, "text": full_text})

    return sentences, data.get("request_id", "")


def run_sync_flash_asr(
    file_path: str,
    api_key: str,
    model: str = "qwen-audio-3.1-asr-flash",
) -> dict:
    """使用 Base64 Data URI 调用 qwen-audio-3.1-asr-flash（无需 /api/v1/uploads 上传凭证，超 4 分钟自动切片）。"""
    duration = _get_audio_duration_seconds(file_path)
    num_chunks = max(1, math.ceil(duration / FLASH_CHUNK_SECONDS)) if duration > 0 else 1

    all_sentences: list[dict] = []
    request_ids: list[str] = []

    with tempfile.TemporaryDirectory(prefix="asr_flash_") as tmp_dir:
        for idx in range(num_chunks):
            start_sec = idx * FLASH_CHUNK_SECONDS
            chunk_path = os.path.join(tmp_dir, f"chunk_{idx:03d}.mp3")
            cmd = [
                FFMPEG_BIN, "-y",
                "-ss", str(start_sec),
                "-t", str(FLASH_CHUNK_SECONDS),
                "-i", file_path,
                "-vn", "-ac", "1", "-ar", "16000",
                "-acodec", "libmp3lame", "-ab", "48k",
                chunk_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if res.returncode != 0 or not os.path.exists(chunk_path) or os.path.getsize(chunk_path) == 0:
                if idx == 0:
                    # 兜底：直接使用原文件
                    chunk_path = file_path
                else:
                    continue

            b64_str = base64.b64encode(Path(chunk_path).read_bytes()).decode("ascii")
            data_uri = f"data:audio/mpeg;base64,{b64_str}"
            chunk_sentences, req_id = _call_multimodal_chunk(
                data_uri,
                api_key=api_key,
                model=model,
                audio_format="mp3",
                offset_ms=start_sec * 1000,
            )
            all_sentences.extend(chunk_sentences)
            if req_id:
                request_ids.append(req_id)

    grouped = _build_grouped_from_sentences(all_sentences)
    grouped["audio_url"] = "data:audio/mpeg;base64"
    grouped["task_id"] = ",".join(request_ids)
    grouped["model"] = model
    return grouped


def run_asr(file_path: str, speaker_count: int | None = None, model: str = MODEL) -> dict:
    """完整执行：上传/编码 -> 提交/调用 -> 解析，返回按说话人整理的结果。"""
    api_key = _api_key()

    # Flash 同步多模态模型直接走 Base64 Data URI，无需调用 /api/v1/uploads 获取上传凭证
    if _is_sync_flash_model(model):
        return run_sync_flash_asr(file_path, api_key, model=model)

    audio_url = upload_to_dashscope(file_path, api_key, model=model)
    task_id = submit_transcription(audio_url, api_key, speaker_count, model=model)
    out = poll_task(task_id, api_key)

    if out.get("task_status") != "SUCCEEDED":
        raise ASRError(f"任务失败，状态: {out.get('task_status')}")

    results = out.get("results", [])
    if not results:
        raise ASRError("任务无结果")

    sub = results[0]
    if sub.get("subtask_status") != "SUCCEEDED":
        raise ASRError(f"子任务失败: {sub.get('code')} {sub.get('message')}")

    transcript = fetch_transcript(sub["transcription_url"])
    grouped = group_by_speaker(transcript)
    grouped["audio_url"] = audio_url
    grouped["task_id"] = task_id
    grouped["model"] = model
    return grouped
