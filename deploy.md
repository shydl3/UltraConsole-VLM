# UltraConsole 部署指南

> 本文档面向第一次部署 UltraConsole 的用户，覆盖环境准备、一键部署、手动部署、日常启停、常见问题与运行配置。
> 项目介绍与功能概览见 [README](README.md)，详细使用说明见 [使用手册](docs/USAGE.md)。

---

## 1. 环境要求

在开始之前，请确认你的电脑满足以下条件：

| 条件 | 说明 |
|---|---|
| 操作系统 | Linux / Windows WSL2（Ubuntu 22.04+ 已验证） |
| 显卡 | NVIDIA 显卡，已安装官方驱动（模型推理需要 CUDA 加速） |
| 磁盘空间 | 至少 **40 GB** 空闲（Python 依赖约 8 GB + 模型约 13 GB（2B+4B）+ 视频数据） |
| 网络 | 可访问国内镜像源（脚本默认全部走清华 / npmmirror / ModelScope 镜像） |

> **Windows 用户**：请先安装 [WSL2](https://learn.microsoft.com/zh-cn/windows/wsl/install)（在 PowerShell 管理员模式执行 `wsl --install`），并在 Windows 侧安装 [NVIDIA 显卡驱动](https://www.nvidia.cn/drivers/)。之后所有命令都在 WSL2 的 Ubuntu 终端里执行。（注意：Windows只要安装了nvidia显卡驱动，一般就代表ok了）
>
> **验证显卡是否可用**：打开powershell，在终端输入下面这条命令，能看到显卡信息输出即为正常：
>
> ```bash
> nvidia-smi
> ```

---

## 2. 获取项目代码

如果还没有项目代码，先用 git 克隆到本地（`git clone` 就是把远程仓库完整下载到你的电脑）：

```bash
git clone <你的仓库地址> ultraconsole
cd ultraconsole
```

已经有代码的用户，直接进入项目目录即可：

```bash
cd ultraconsole
```

> 后面所有命令都默认在 `ultraconsole` 项目根目录下执行。

---

## 3. 一键部署（推荐）

项目提供了 `deploy.sh` 部署脚本，**一条命令自动完成全部安装**：

```bash
bash deploy.sh
```

脚本执行过程中会提示你输入一次 **sudo 密码**（即开机/管理员密码，输入时屏幕不显示任何字符是正常现象，输完按回车即可）。

脚本会依次自动完成以下 9 个步骤（每步都已在电脑上完成过的会自动跳过，因此**中断后可以放心重新执行**）：

| 步骤 | 内容 | 作用 |
|---|---|---|
| 1 | 安装系统依赖 | curl（下载工具）、git、libgl1 等 OpenCV 运行所需的图形库 |
| 2 | 安装 Miniconda | Python 环境管理器，用于创建与系统隔离的 Python 环境 |
| 3 | 创建 `ultraconsole` 环境并安装 Python 依赖 | Python 3.10 + PyTorch + FastAPI + Qwen 推理库（首次约 6-8 GB，耗时最长） |
| 4 | 安装 Node.js 22 | 前端页面的运行环境 |
| 5 | 安装前端依赖 | `npm ci` 按锁定文件精确安装 React/Vite 等前端包 |
| 6 | 下载模型权重 | YOLO11m-Pose（约 40 MB）+ Qwen3-VL-2B（约 4 GB）+ Qwen3-VL-4B（约 9 GB），均来自 ModelScope |
| 7 | 写入 `backend/.env` | 告诉后端用哪个 Python 来跑大模型推理 |
| 8 | 校验安装 | 检查 PyTorch / CUDA / 模型文件是否全部就绪 |
| 9 | 启动服务 | 调用 `./start.sh` 启动前后端 |

看到绿色的 **`部署完成，服务已启动：http://localhost:5100`** 即表示成功，用浏览器打开该地址即可使用。

### 常用参数

```bash
bash deploy.sh --port 8080     # 换用 8080 端口启动前端（默认 5100）
bash deploy.sh --no-start      # 只安装部署，不自动启动
bash deploy.sh --skip-apt      # 跳过系统包安装（没有 sudo 权限时用）
bash deploy.sh --yolo /path/to/yolo11m-pose.pt   # 使用已下载好的 YOLO 权重文件
```

---

## 4. 手动分步部署（可选）

如果你希望了解每一步在做什么，或者一键脚本在你的环境上遇到问题，可以按下面的步骤逐条执行。

### 4.1 安装系统依赖

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates xz-utils git libgl1 libgomp1 libglib2.0-0t64
```

- `apt-get update`：刷新软件源列表，让系统知道有哪些软件可装。
- `libgl1` / `libglib2.0-0t64`：OpenCV（图像处理库）运行必需的系统图形库，缺了会报 `import cv2` 错误。
- Ubuntu 22.04 之前版本若没有 `libglib2.0-0t64`，改用 `libglib2.0-0`；一键脚本会自动尝试两者。

### 4.2 安装 Miniconda（已有 conda 可跳过）

```bash
curl -fL -o /tmp/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
```

- 第一条：从清华镜像下载 Miniconda 安装包。
- 第二条：`-b` 表示静默安装到 `$HOME/miniconda3` 目录。
- 第三条：让当前终端立刻认识 `conda` 命令。

### 4.3 创建 Python 环境并安装依赖

```bash
conda create -n ultraconsole python=3.10 -y
conda activate ultraconsole
pip install -r backend/requirements-wsl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- `conda create`：创建一个名为 `ultraconsole` 的独立 Python 3.10 环境，不会污染系统 Python。
- `conda activate`：切换到这个环境（之后终端提示符前会出现 `(ultraconsole)` 字样）。
- `pip install -r`：按依赖清单安装 PyTorch、FastAPI、Qwen 推理库等；`-i` 指定清华镜像加速下载。

### 4.4 安装 Node.js（已有 20+ 版本可跳过）

```bash
curl -fL -o /tmp/node.tar.xz https://npmmirror.com/mirrors/node/v22.14.0/node-v22.14.0-linux-x64.tar.xz
mkdir -p $HOME/.local/node
tar -xJf /tmp/node.tar.xz -C $HOME/.local/node --strip-components=1
export PATH="$HOME/.local/node/bin:$PATH"
echo 'export PATH="$HOME/.local/node/bin:$PATH"' >> $HOME/.bashrc
```

- 从 npmmirror 镜像下载 Node.js 22 压缩包并解压到用户目录（不需要 sudo）。
- 最后一条把路径写入 `~/.bashrc`，让以后每次打开终端都能使用 `node` / `npm` 命令。

### 4.5 安装前端依赖

```bash
cd frontend
npm ci --registry=https://registry.npmmirror.com
cd ..
```

- `npm ci`：严格按照 `package-lock.json` 安装前端依赖，保证每个人装出来的版本完全一致。

### 4.6 下载模型权重

```bash
# YOLO11m-Pose：人物姿态跟踪模型（约 40 MB）
mkdir -p backend/models
curl -fL -o backend/models/yolo11m-pose.pt \
  https://modelscope.cn/models/Ultralytics/YOLO11/resolve/master/yolo11m-pose.pt

# Qwen3-VL-2B + 4B：视觉语言大模型（约 4 GB + 9 GB，脚本自动校验完整性、支持断点续传）
MODELS_DIR=models VLM_MODEL_REPO=Qwen/Qwen3-VL-2B-Instruct \
VLM_MODEL_PATH=models/Qwen3-VL-2B-Instruct \
YOLO_PATH=backend/models/yolo11m-pose.pt VLM_MODEL_SOURCE=modelscope \
HF_ENDPOINT=https://hf-mirror.com \
python scripts/download_models.py

MODELS_DIR=models VLM_MODEL_REPO=Qwen/Qwen3-VL-4B-Instruct \
VLM_MODEL_PATH=models/Qwen3-VL-4B-Instruct \
YOLO_PATH=backend/models/yolo11m-pose.pt VLM_MODEL_SOURCE=modelscope \
HF_ENDPOINT=https://hf-mirror.com \
python scripts/download_models.py
```

| 模型 | 用途 | 存放位置 |
|---|---|---|
| `yolo11m-pose.pt` | 人员检测与姿态跟踪 | `backend/models/` |
| `Qwen3-VL-2B-Instruct` / `4B-Instruct` | 事件语义判定（VLM，2B 为默认） | `models/` |

### 4.7 写入后端配置

```bash
echo "VLM_PYTHON=$(which python)" >> backend/.env
```

- 把当前 conda 环境的 Python 路径写入 `backend/.env`，后端会用它来启动 Qwen 大模型推理。
- 前提是当前处于 `(ultraconsole)` 环境中（`which python` 才能查到正确路径）。

### 4.8 启动

```bash
./start.sh
```

看到 `访问地址：http://localhost:5100` 后，用浏览器打开即可。

---

## 5. 日常启动与停止

部署完成后，以后每次使用只需要两条命令：

```bash
./start.sh          # 启动（前端默认 5100 端口，后端固定 8001）
./stop.sh           # 停止全部服务
```

```bash
./start.sh 8080     # 指定前端端口启动
./start.sh --prod   # 生产模式：构建前端并由后端 8001 直接托管（页面与视频同源，无 Vite 代理）
```

- 日志目录：`tmp/ultraconsole/`（`backend.log` 后端日志、`frontend.log` 前端日志）。
- 后端启动时会自动选择 CUDA 并加载姿态模型，无需手动配置。
- **开发模式**（默认）由 Vite 提供页面，`/api` 经开发代理转发到后端；**生产模式**只需访问 `http://localhost:8001`，页面、接口、视频字节范围请求都在同一来源，视频拖动更稳，适合日常使用。

---

## 6. 常见问题

**Q1：提示 `command not found: conda`？**
conda 尚未初始化到当前终端。执行 `source ~/.bashrc`，或关闭终端重新打开。

**Q2：校验时显示 `cuda: False`？**
显卡驱动未生效。WSL2 用户请在 Windows PowerShell（管理员）执行 `wsl --shutdown`，然后重新打开 WSL 终端再运行 `bash deploy.sh`（已完成的步骤会自动跳过）。仍不行请重装 [NVIDIA 驱动](https://www.nvidia.cn/drivers/)。

**Q3：模型下载太慢或失败？**
脚本默认走 ModelScope（国内源）。如果仍失败，可手动从 [ModelScope Qwen3-VL-2B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct) 页面下载全部文件，放到 `models/Qwen3-VL-2B-Instruct/` 目录后重新执行 `bash deploy.sh`。

**Q4：端口被占用（`端口 5100 已被其他进程占用`）？**
换一个端口启动：`./start.sh 8080`。

**Q5：`import cv2` 报错 `libGL.so.1: cannot open shared object file`？**
缺少系统图形库，执行：`sudo apt-get install -y libgl1 libglib2.0-0t64`。

**Q6：部署脚本中途报错退出了怎么办？**
根据错误提示解决后，直接重新执行 `bash deploy.sh` 即可——脚本是幂等的，已完成的步骤会自动跳过。

---

## 7. 运行配置

以下环境变量写入 `backend/.env` 即可生效，普通用户无需修改：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `VLM_MODEL_PATH` | `models/Qwen3-VL-2B-Instruct` | Qwen3-VL 模型目录 |
| `VLM_PYTHON` | 当前 Python | VLM 推理解释器（独立环境时覆盖） |
| `VLM_DATA_ROOT` | `Rules/data/vlm` | 分析数据根目录 |
| `VLM_BROWSE_ROOT` | 用户主目录 `~` | 「从服务器选择视频」的文件浏览根目录，浏览与导入均被限制在该目录内；同盘导入使用硬链接，不复制、不上传 |
| `ULTRA_API_TOKEN` | *(空)* | HTTP 鉴权令牌（设置后接口需要携带令牌） |
| `ULTRA_DEVICE` | *(自动)* | 强制指定 `cpu` / `cuda` |
| `ULTRA_AUTOSTART` | `true` | 启动时自动选设备并加载模型 |
| `ULTRA_DEFAULT_MODEL` | `yolo11m-pose.pt` | 默认检测模型文件名 |
| `VLM_SEGMENT_SECONDS` | `600` | 视频分段时长（秒） |
| `VLM_MAX_UPLOAD_BYTES` / `VLM_MAX_DURATION_SEC` | `128 GiB` / `172800` | 单条视频的导入/上传大小 / 时长上限（`0` 表示不限制，12–24h 长录像默认可分析） |
| `VLM_PLAYBACK_MAX_DURATION_SEC` | `43200` | 生成回放代理的时长上限（`0` 表示不限制；超长录像直接播放原片） |

开发相关文档见 [dev-guide.md](dev-guide.md)；WSL2 安装说明见 [Microsoft 官方文档](https://learn.microsoft.com/zh-cn/windows/wsl/install)。

---

[返回 README](README.md) · [使用手册](docs/USAGE.md) · [开发指南](dev-guide.md)

