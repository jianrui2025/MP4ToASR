"""阿里云百炼 Fun-ASR 录音文件识别（含说话人分离）。

流程：
1. 将本地 MP3 上传到百炼临时存储，获得 48 小时有效、可被百炼访问的 oss:// URL。
2. 以异步方式提交 fun-asr-2025-11-07 转写任务（开启 diarization_enabled 说话人分离）。
3. 轮询任务状态直至完成。
4. 下载转写结果 JSON，并按说话人(speaker_id)整理为对话片段。

API Key 通过环境变量 DASHSCOPE_API_KEY 读取。
"""
import os
import time
from pathlib import Path

import requests

MODEL = "fun-asr-2025-11-07"
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com"
UPLOAD_URL = f"{DASHSCOPE_BASE}/api/v1/uploads"
SUBMIT_URL = f"{DASHSCOPE_BASE}/api/v1/services/audio/asr/transcription"
TASK_URL = f"{DASHSCOPE_BASE}/api/v1/tasks/{{task_id}}"


class ASRError(Exception):
    pass


def _api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise ASRError("未配置 DASHSCOPE_API_KEY 环境变量")
    return key


def upload_to_dashscope(file_path: str, api_key: str | None = None) -> str:
    """上传文件到百炼临时存储，返回 oss:// 临时 URL（48 小时有效）。"""
    api_key = api_key or _api_key()

    # 步骤1：获取上传凭证
    resp = requests.get(
        UPLOAD_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        params={"action": "getPolicy", "model": MODEL},
        timeout=30,
    )
    if resp.status_code != 200:
        raise ASRError(f"获取上传凭证失败: {resp.status_code} {resp.text[:300]}")
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
    audio_url: str, api_key: str | None = None, speaker_count: int | None = None
) -> str:
    """提交异步转写任务，开启说话人分离，返回 task_id。"""
    api_key = api_key or _api_key()

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
        "model": MODEL,
        "input": {"file_urls": [audio_url]},
        "parameters": parameters,
    }
    resp = requests.post(SUBMIT_URL, headers=headers, json=body, timeout=30)
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
    url = TASK_URL.format(task_id=task_id)
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


def _fmt_ts(ms: int) -> str:
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def group_by_speaker(transcript: dict) -> dict:
    """把转写结果整理为按说话人分离的对话片段。

    返回:
    {
      "segments": [ {"speaker": int, "begin": "00:00", "end": "00:03", "text": "..."}, ... ],
      "speakers": { "0": "合并后的全部文本", ... },
      "speaker_count": n
    }
    """
    sentences = []
    for tr in transcript.get("transcripts", []):
        for st in tr.get("sentences", []):
            sentences.append(
                {
                    "speaker": st.get("speaker_id", 0),
                    "begin_time": st.get("begin_time", 0),
                    "end_time": st.get("end_time", 0),
                    "text": (st.get("text") or "").strip(),
                }
            )

    sentences.sort(key=lambda x: x["begin_time"])

    # 合并连续同一说话人的句子为一个气泡
    segments = []
    for s in sentences:
        if segments and segments[-1]["speaker"] == s["speaker"]:
            segments[-1]["text"] += s["text"]
            segments[-1]["end_time"] = s["end_time"]
        else:
            segments.append(dict(s))

    speakers: dict[str, str] = {}
    for s in sentences:
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


def run_asr(file_path: str, speaker_count: int | None = None) -> dict:
    """完整执行：上传 -> 提交 -> 轮询 -> 解析，返回按说话人整理的结果。"""
    api_key = _api_key()
    audio_url = upload_to_dashscope(file_path, api_key)
    task_id = submit_transcription(audio_url, api_key, speaker_count)
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
    return grouped
