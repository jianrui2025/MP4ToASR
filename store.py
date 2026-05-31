"""任务持久化存储：每个任务保存为 tasks/<id>.json，全部保留在项目目录中。"""
import json
import os
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(BASE_DIR, "tasks")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_FILE = os.path.join(BASE_DIR, "models.json")
os.makedirs(TASKS_DIR, exist_ok=True)

# 默认可选的 ASR 模型
DEFAULT_MODELS = [
    "fun-asr-2025-11-07",
    "fun-asr",
    "fun-asr-2025-08-25",
    "fun-asr-mtl",
    "fun-asr-mtl-2025-08-25",
]

_lock = threading.Lock()
_models_lock = threading.Lock()


def list_models() -> list[str]:
    with _models_lock:
        if not os.path.exists(MODELS_FILE):
            return list(DEFAULT_MODELS)
        try:
            with open(MODELS_FILE, "r", encoding="utf-8") as fh:
                models = json.load(fh)
            if isinstance(models, list) and models:
                return models
        except (OSError, json.JSONDecodeError):
            pass
        return list(DEFAULT_MODELS)


def add_model(name: str) -> list[str]:
    name = (name or "").strip()
    if not name:
        return list_models()
    with _models_lock:
        models = []
        if os.path.exists(MODELS_FILE):
            try:
                with open(MODELS_FILE, "r", encoding="utf-8") as fh:
                    models = json.load(fh)
            except (OSError, json.JSONDecodeError):
                models = []
        if not models:
            models = list(DEFAULT_MODELS)
        if name not in models:
            models.append(name)
        with open(MODELS_FILE, "w", encoding="utf-8") as fh:
            json.dump(models, fh, ensure_ascii=False, indent=2)
        return models


def _path(task_id: str) -> str:
    return os.path.join(TASKS_DIR, f"{task_id}.json")


def create_task(task_id: str, title: str, description: str) -> dict:
    task = {
        "id": task_id,
        "title": title,
        "description": description,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "created",        # created -> queued -> converting -> transcribing -> done / failed
        "source": None,             # uploads/ 下的原始视频文件名
        "mp3": None,                # outputs/ 下的 mp3 文件名
        "mp3_size_mb": None,
        "bitrate": None,
        "speaker_count": None,
        "model": None,
        "asr": None,                # {segments, speakers, speaker_count, audio_url}
        "error": None,
    }
    save_task(task)
    return task


def save_task(task: dict) -> None:
    with _lock:
        with open(_path(task["id"]), "w", encoding="utf-8") as fh:
            json.dump(task, fh, ensure_ascii=False, indent=2)


def load_task(task_id: str) -> dict | None:
    p = _path(task_id)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def update_task(task_id: str, **fields) -> dict | None:
    task = load_task(task_id)
    if task is None:
        return None
    task.update(fields)
    save_task(task)
    return task


def delete_task(task_id: str) -> bool:
    """删除任务记录，并连带清理其原始视频与 MP3 中间文件。"""
    task = load_task(task_id)
    if task is None:
        return False
    with _lock:
        # 删除关联的中间文件
        if task.get("source"):
            try:
                os.remove(os.path.join(UPLOAD_DIR, task["source"]))
            except OSError:
                pass
        if task.get("mp3"):
            try:
                os.remove(os.path.join(OUTPUT_DIR, task["mp3"]))
            except OSError:
                pass
        try:
            os.remove(_path(task_id))
        except OSError:
            return False
    return True


def list_tasks() -> list[dict]:
    """返回任务摘要列表，按创建时间倒序。"""
    items = []
    for name in os.listdir(TASKS_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(TASKS_DIR, name), "r", encoding="utf-8") as fh:
                t = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        items.append(
            {
                "id": t.get("id"),
                "title": t.get("title"),
                "description": t.get("description"),
                "created_at": t.get("created_at"),
                "status": t.get("status"),
                "speaker_count": (t.get("asr") or {}).get("speaker_count"),
            }
        )
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return items
