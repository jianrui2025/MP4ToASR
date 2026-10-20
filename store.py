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
FOLDERS_FILE = os.path.join(BASE_DIR, "folders.json")
os.makedirs(TASKS_DIR, exist_ok=True)

DEFAULT_FOLDER = "默认目录"
DEFAULT_FOLDERS = [DEFAULT_FOLDER]

# 默认可选的 ASR 模型
DEFAULT_MODELS = [
    "fun-asr-2025-11-07",
    "qwen-audio-3.1-asr-flash",
    "qwen-audio-3.1-asr-flash-filetrans",
    "fun-asr",
    "fun-asr-2025-08-25",
    "fun-asr-mtl",
    "fun-asr-mtl-2025-08-25",
]

_lock = threading.Lock()
_models_lock = threading.Lock()
_folders_lock = threading.Lock()


def list_folders() -> list[str]:
    with _folders_lock:
        folders: list[str] = []
        if os.path.exists(FOLDERS_FILE):
            try:
                with open(FOLDERS_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    folders = [str(x).strip() for x in data if str(x).strip()]
            except (OSError, json.JSONDecodeError):
                folders = []
        if not folders:
            folders = list(DEFAULT_FOLDERS)
        # 确保已有任务用到的目录也在列表中
        for name in os.listdir(TASKS_DIR):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(TASKS_DIR, name), "r", encoding="utf-8") as fh:
                    t = json.load(fh)
                f = (t.get("folder") or "").strip() or DEFAULT_FOLDER
                if f not in folders:
                    folders.append(f)
            except (OSError, json.JSONDecodeError):
                continue
        return folders


def add_folder(name: str) -> list[str]:
    name = (name or "").strip()
    if not name:
        return list_folders()
    with _folders_lock:
        folders: list[str] = []
        if os.path.exists(FOLDERS_FILE):
            try:
                with open(FOLDERS_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    folders = [str(x).strip() for x in data if str(x).strip()]
            except (OSError, json.JSONDecodeError):
                folders = []
        if not folders:
            folders = list(DEFAULT_FOLDERS)
        if name not in folders:
            folders.append(name)
        with open(FOLDERS_FILE, "w", encoding="utf-8") as fh:
            json.dump(folders, fh, ensure_ascii=False, indent=2)
    return list_folders()


def delete_folder(name: str) -> tuple[bool, str]:
    name = (name or "").strip()
    if not name:
        return False, "目录名称不能为空"
    # 检查是否有任务属于该目录
    for fname in os.listdir(TASKS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(TASKS_DIR, fname), "r", encoding="utf-8") as fh:
                t = json.load(fh)
            tf = (t.get("folder") or "").strip() or DEFAULT_FOLDER
            if tf == name:
                return False, f"目录「{name}」下仍有任务，请先删除目录内的任务"
        except (OSError, json.JSONDecodeError):
            continue

    with _folders_lock:
        folders: list[str] = []
        if os.path.exists(FOLDERS_FILE):
            try:
                with open(FOLDERS_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    folders = [str(x).strip() for x in data if str(x).strip()]
            except (OSError, json.JSONDecodeError):
                folders = []
        if not folders:
            folders = list(DEFAULT_FOLDERS)
        if name not in folders:
            return False, "目录不存在"
        if len(folders) <= 1:
            return False, "至少需要保留一个任务目录"
        folders = [f for f in folders if f != name]
        with open(FOLDERS_FILE, "w", encoding="utf-8") as fh:
            json.dump(folders, fh, ensure_ascii=False, indent=2)
    return True, ""


def reorder_folders(new_order: list[str]) -> list[str]:
    """按给定顺序重排目录文件夹列表并持久化。"""
    current = list_folders()
    cleaned: list[str] = []
    for item in new_order or []:
        name = str(item).strip()
        if name and name in current and name not in cleaned:
            cleaned.append(name)
    for name in current:
        if name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        cleaned = list(DEFAULT_FOLDERS)
    with _folders_lock:
        with open(FOLDERS_FILE, "w", encoding="utf-8") as fh:
            json.dump(cleaned, fh, ensure_ascii=False, indent=2)
    return list_folders()


def move_task_to_folder(task_id: str, folder: str) -> dict | None:
    """将指定任务移动到目标目录文件夹。"""
    folder = (folder or "").strip()
    if not folder:
        return None
    task = load_task(task_id)
    if task is None:
        return None
    add_folder(folder)
    task["folder"] = folder
    save_task(task)
    return task


def list_models() -> list[str]:
    with _models_lock:
        if not os.path.exists(MODELS_FILE):
            return list(DEFAULT_MODELS)
        try:
            with open(MODELS_FILE, "r", encoding="utf-8") as fh:
                models = json.load(fh)
            if isinstance(models, list) and models:
                merged = [str(m).strip() for m in models if str(m).strip()]
                for dm in DEFAULT_MODELS:
                    if dm not in merged:
                        merged.append(dm)
                return merged
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


def create_task(task_id: str, title: str, description: str, folder: str = DEFAULT_FOLDER) -> dict:
    folder = (folder or "").strip() or DEFAULT_FOLDER
    add_folder(folder)
    task = {
        "id": task_id,
        "title": title,
        "description": description,
        "folder": folder,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "created",        # created -> queued -> converting -> transcribing -> done / failed
        "source": None,             # uploads/ 下的原始音视频文件名
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
        task = json.load(fh)
    if not (task.get("folder") or "").strip():
        task["folder"] = DEFAULT_FOLDER
    return task


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


def list_tasks(folder: str | None = None) -> list[dict]:
    """返回任务摘要列表，按创建时间倒序。可选按目录过滤。"""
    target_folder = (folder or "").strip() if folder else None
    items = []
    for name in os.listdir(TASKS_DIR):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(TASKS_DIR, name), "r", encoding="utf-8") as fh:
                t = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        task_folder = (t.get("folder") or "").strip() or DEFAULT_FOLDER
        if target_folder and task_folder != target_folder:
            continue
        items.append(
            {
                "id": t.get("id"),
                "title": t.get("title"),
                "description": t.get("description"),
                "folder": task_folder,
                "created_at": t.get("created_at"),
                "status": t.get("status"),
                "speaker_count": (t.get("asr") or {}).get("speaker_count"),
            }
        )
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return items
