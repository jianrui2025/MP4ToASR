# MP4 转 MP3 网页工具

一个简单的本地 Web 应用：上传 MP4（或其他视频）文件，提取音轨并转换为 MP3。
转换过程中的中间文件（上传的原始视频）会保留在项目的 `uploads/` 目录中，
生成的 MP3 保存在 `outputs/` 目录中。

## 功能特性

- 密码登录控制：仅需密码；连续 3 次错误锁定 10 分钟；每次刷新页面需重新验证
- 拖拽 / 点击上传视频或音频文件（视频：MP4 / MOV / MKV / AVI / WEBM / M4V；音频：MP3 / WAV / M4A / FLAC / AAC / OGG / OPUS / WMA / AMR / AIFF / CAF / MKA / AC3）
- 可选输出音质：128 / 192 / 256 / 320 kbps
- 上传进度显示与转换结果下载
- 使用 `imageio-ffmpeg` 自带的 ffmpeg 二进制，无需系统级安装 ffmpeg
- 中间文件保留在项目目录中，便于后续处理（如 ASR）
- 一键调用阿里云百炼 `fun-asr-2025-11-07` 进行语音识别，开启**说话人分离**，按说话人分别展示

## 语音识别（ASR）说明

转换得到 MP3 后，点击「语音识别(说话人分离)」即可：

1. 将 MP3 上传至百炼临时存储，获得 48 小时有效、百炼可访问的 `oss://` 临时 URL；
2. 以异步方式提交 `fun-asr-2025-11-07` 转写任务，并开启 `diarization_enabled` 说话人分离；
3. 轮询任务直至完成，下载结果并按 `speaker_id` 整理为对话片段展示。

> 为什么用 `oss://` 临时 URL：本机服务未对公网开放，百炼无法直接抓取本地 `/download` 地址，
> 故采用官方临时文件上传服务生成百炼可访问的 URL（适用于没有公网 URL 的场景）。

需配置 API Key 环境变量：

```bash
export DASHSCOPE_API_KEY='你的百炼API Key'
export DASHSCOPE_BASE_URL='xxxx'
# 可选：自定义百炼 Base URL（默认 https://dashscope.aliyuncs.com，末尾带不带 /api/v1 均可）
# export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com'
```

## 目录结构

```
MP4ToASR/
├── app.py              # Flask 后端
├── requirements.txt    # 依赖
├── templates/
│   └── index.html      # 前端页面
├── uploads/            # 上传的原始视频（中间文件，自动创建）
└── outputs/            # 转换生成的 MP3（自动创建）
```

## 安装与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置百炼 API Key（用于 ASR）
export DASHSCOPE_API_KEY='你的百炼API Key'
export DASHSCOPE_BASE_URL='xxxx'

# 3. 启动服务（生产服务器 waitress）
python serve.py
#   或开发模式： python app.py

# 4. 浏览器打开
#    http://localhost:5001   默认密码 123456（可用环境变量 APP_PASSWORD 修改）
```

## 说明

- 最大上传限制为 2GB，可在 `app.py` 中的 `MAX_CONTENT_LENGTH` 调整。
- `uploads/` 与 `outputs/` 目录中的文件会持续累积，可按需手动清理。
