import hmac
import os
import secrets
import subprocess
import threading
import time
import uuid
from datetime import datetime
from functools import wraps

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

import imageio_ffmpeg

from asr import ASRError, run_asr
import store

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ffmpeg 二进制由 imageio-ffmpeg 提供，无需系统级安装
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB

# ===== 登录控制配置 =====
# 访问密码：通过环境变量 APP_PASSWORD 配置，默认 "123456"
APP_PASSWORD = os.environ.get("APP_PASSWORD", "123456")
MAX_ATTEMPTS = 3            # 连续错误次数上限
LOCK_SECONDS = 10 * 60      # 锁定时长：10 分钟
TOKEN_TTL = 2 * 60 * 60     # 访问令牌有效期：2 小时

# 失败记录：{ip: {"count": int, "lock_until": float}}
_attempts: dict[str, dict] = {}
# 有效访问令牌：{token: expire_ts}
_tokens: dict[str, float] = {}
_auth_lock = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _lock_remaining(ip: str) -> int:
    """返回该 IP 剩余锁定秒数，0 表示未锁定。"""
    rec = _attempts.get(ip)
    if not rec:
        return 0
    remaining = int(rec.get("lock_until", 0) - time.time())
    return remaining if remaining > 0 else 0


def _issue_token() -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _auth_lock:
        # 顺手清理过期令牌
        for t in [k for k, exp in _tokens.items() if exp < now]:
            _tokens.pop(t, None)
        _tokens[token] = now + TOKEN_TTL
    return token


def _validate_token(token: str | None) -> bool:
    if not token:
        return False
    with _auth_lock:
        exp = _tokens.get(token)
        if exp is None:
            return False
        if exp < time.time():
            _tokens.pop(token, None)
            return False
        return True


def require_token(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        token = request.headers.get("X-Access-Token") or request.args.get("token")
        if not _validate_token(token):
            return jsonify({"error": "未授权或会话已失效，请重新登录", "auth": False}), 401
        return view(*args, **kwargs)

    return wrapper


def _is_allowed(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    # 每次访问/刷新都回到登录页，登录态仅存在于页面内存中，刷新即失效
    return render_template("index.html")


@app.route("/auth", methods=["POST"])
def auth():
    ip = _client_ip()

    with _auth_lock:
        rec = _attempts.setdefault(ip, {"count": 0, "lock_until": 0.0})

    remaining = _lock_remaining(ip)
    if remaining > 0:
        return (
            jsonify({"error": "登录已被锁定", "locked": True, "remaining": remaining}),
            429,
        )

    data = request.get_json(silent=True) or {}
    password = data.get("password", "")

    if hmac.compare_digest(str(password), APP_PASSWORD):
        # 登录成功：重置失败计数，签发令牌
        with _auth_lock:
            _attempts.pop(ip, None)
        return jsonify({"ok": True, "token": _issue_token()})

    # 密码错误：累加失败次数
    with _auth_lock:
        rec["count"] += 1
        count = rec["count"]
        if count >= MAX_ATTEMPTS:
            rec["lock_until"] = time.time() + LOCK_SECONDS
            rec["count"] = 0  # 锁定后重置计数，锁定到期重新计
            return (
                jsonify(
                    {
                        "error": "密码连续错误次数过多，已锁定",
                        "locked": True,
                        "remaining": LOCK_SECONDS,
                    }
                ),
                429,
            )
        left = MAX_ATTEMPTS - count
    return jsonify({"error": "密码错误", "attempts_left": left}), 401


def _process_task_bg(task_id, input_path, output_path, output_name, bitrate, speaker_count, model):
    """后台线程：完整执行 转换 -> 识别。客户端断开/关闭网页不影响本流程。"""
    # 1. 转换为 MP3
    store.update_task(task_id, status="converting", error=None)
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vn", "-acodec", "libmp3lame", "-ab", bitrate, output_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except Exception as exc:  # noqa: BLE001
        store.update_task(task_id, status="failed", error=f"转换进程启动失败: {exc}")
        return

    if result.returncode != 0 or not os.path.exists(output_path):
        stderr_tail = result.stderr.decode("utf-8", errors="ignore")[-500:]
        store.update_task(task_id, status="failed", error="转换失败: " + stderr_tail)
        return

    size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    store.update_task(task_id, status="transcribing", mp3=output_name, mp3_size_mb=size_mb)

    # 2. 语音识别（含说话人分离）
    try:
        result_asr = run_asr(output_path, speaker_count=speaker_count, model=model)
    except ASRError as exc:
        store.update_task(task_id, status="failed", error=str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        store.update_task(task_id, status="failed", error=f"ASR 识别异常: {exc}")
        return

    store.update_task(task_id, status="done", asr=result_asr)


def _start_task_thread(task_id, input_path, output_path, output_name, bitrate, speaker_count, model):
    t = threading.Thread(
        target=_process_task_bg,
        args=(task_id, input_path, output_path, output_name, bitrate, speaker_count, model),
        daemon=True,
    )
    t.start()


@app.route("/convert", methods=["POST"])
@require_token
def convert():
    if "file" not in request.files:
        return jsonify({"error": "未检测到上传的文件"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    if not _is_allowed(file.filename):
        return jsonify({"error": "不支持的文件格式，请上传 MP4 等视频文件"}), 400

    title = (request.form.get("title") or "").strip()
    description = (request.form.get("description") or "").strip()
    if not title:
        return jsonify({"error": "请填写任务标题"}), 400

    # 生成唯一标识，保留中间文件（上传的视频）于项目的 uploads 目录中
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    safe_name = secure_filename(file.filename) or "input.mp4"
    base_name = os.path.splitext(safe_name)[0]

    input_path = os.path.join(UPLOAD_DIR, f"{stamp}__{safe_name}")
    output_name = f"{stamp}__{base_name}.mp3"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    bitrate = request.form.get("bitrate", "192k")
    if bitrate not in {"128k", "192k", "256k", "320k"}:
        bitrate = "192k"

    speaker_count = request.form.get("speaker_count")
    try:
        speaker_count = int(speaker_count) if speaker_count else None
    except (TypeError, ValueError):
        speaker_count = None

    model = (request.form.get("model") or "").strip() or "fun-asr-2025-11-07"

    # 创建任务并保存上传文件（此步骤需在请求内完成，是客户端->服务器的传输）
    store.create_task(stamp, title, description)
    file.save(input_path)
    store.update_task(
        stamp,
        source=os.path.basename(input_path),
        status="queued",
        bitrate=bitrate,
        speaker_count=speaker_count,
        model=model,
    )

    # 启动后台线程处理（转换 + 识别），立即返回；关闭网页不影响后台任务
    _start_task_thread(stamp, input_path, output_path, output_name, bitrate, speaker_count, model)

    return jsonify({"ok": True, "task_id": stamp, "status": "queued"})


@app.route("/download/<path:filename>")
@require_token
def download(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.route("/asr", methods=["POST"])
@require_token
def asr():
    """对任务对应的 MP3 执行百炼 Fun-ASR 识别（含说话人分离）。"""
    data = request.get_json(silent=True) or {}
    task_id = secure_filename(data.get("task_id", ""))
    if not task_id:
        return jsonify({"error": "缺少 task_id 参数"}), 400

    task = store.load_task(task_id)
    if task is None:
        return jsonify({"error": "任务不存在"}), 404
    if not task.get("mp3"):
        return jsonify({"error": "该任务尚未完成转换"}), 400

    mp3_path = os.path.join(OUTPUT_DIR, task["mp3"])
    if not os.path.exists(mp3_path):
        return jsonify({"error": "MP3 文件不存在，请先完成转换"}), 404

    speaker_count = data.get("speaker_count")
    try:
        speaker_count = int(speaker_count) if speaker_count else None
    except (TypeError, ValueError):
        speaker_count = None

    store.update_task(task_id, status="transcribing", error=None)
    try:
        result = run_asr(mp3_path, speaker_count=speaker_count)
    except ASRError as exc:
        store.update_task(task_id, status="failed", error=str(exc))
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:  # noqa: BLE001
        store.update_task(task_id, status="failed", error=str(exc))
        return jsonify({"error": f"ASR 识别异常: {exc}"}), 500

    store.update_task(task_id, status="done", asr=result)
    return jsonify({"ok": True, "task_id": task_id, **result})


@app.route("/models", methods=["GET"])
@require_token
def models_list():
    return jsonify({"ok": True, "models": store.list_models()})


@app.route("/models", methods=["POST"])
@require_token
def models_add():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "模型名称不能为空"}), 400
    models = store.add_model(name)
    return jsonify({"ok": True, "models": models})


@app.route("/tasks", methods=["GET"])
@require_token
def tasks_list():
    return jsonify({"ok": True, "tasks": store.list_tasks()})


@app.route("/tasks/<task_id>", methods=["GET"])
@require_token
def task_detail(task_id):
    task = store.load_task(secure_filename(task_id))
    if task is None:
        return jsonify({"error": "任务不存在"}), 404
    if task.get("mp3"):
        task["download_url"] = f"/download/{task['mp3']}"
    return jsonify({"ok": True, "task": task})


@app.route("/tasks/<task_id>/retry", methods=["POST"])
@require_token
def task_retry(task_id):
    task = store.load_task(secure_filename(task_id))
    if task is None:
        return jsonify({"error": "任务不存在"}), 404
    source = task.get("source")
    if not source or not os.path.exists(os.path.join(UPLOAD_DIR, source)):
        return jsonify({"error": "原始视频已不存在，无法重试"}), 400

    input_path = os.path.join(UPLOAD_DIR, source)
    output_name = os.path.splitext(source)[0] + ".mp3"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    bitrate = task.get("bitrate") or "192k"
    speaker_count = task.get("speaker_count")
    model = task.get("model") or "fun-asr-2025-11-07"

    store.update_task(task["id"], status="queued", error=None)
    _start_task_thread(task["id"], input_path, output_path, output_name, bitrate, speaker_count, model)
    return jsonify({"ok": True, "status": "queued"})


@app.route("/tasks/<task_id>", methods=["DELETE"])
@require_token
def task_delete(task_id):
    ok = store.delete_task(secure_filename(task_id))
    if not ok:
        return jsonify({"error": "任务不存在或删除失败"}), 404
    return jsonify({"ok": True})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "文件过大，超过 2GB 限制"}), 413


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    print(f"ffmpeg 二进制: {FFMPEG_BIN}")
    print(f"访问密码(APP_PASSWORD): {APP_PASSWORD}")
    app.run(host=host, port=port, debug=True)
