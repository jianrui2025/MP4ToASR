"""生产环境启动入口：使用 waitress WSGI 服务器对外提供服务。

用法：
    python serve.py                # 默认 0.0.0.0:5000
    PORT=8080 python serve.py      # 自定义端口
"""
import os

from waitress import serve

from app import FFMPEG_BIN, app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    threads = int(os.environ.get("THREADS", "8"))

    print(f"ffmpeg 二进制: {FFMPEG_BIN}")
    print(f"生产服务器(waitress) 正在监听 http://{host}:{port} (threads={threads})")
    # 大文件上传需要较长超时
    serve(
        app,
        host=host,
        port=port,
        threads=threads,
        channel_timeout=600,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
    )
