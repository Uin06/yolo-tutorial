# Ubuntu USB 摄像头人体检测与跟踪：YOLO 新手教程

从读取第一帧 USB 摄像头画面开始，逐步完成 **人体检测、连续跟踪、坐标导出和参数实验**。适合刚安装 Ubuntu、准备学习 Python 和计算机视觉的同学。

技术组合：Ubuntu 桌面系统 · Conda / Python 3.11 · OpenCV · Ultralytics YOLO11n · ByteTrack。

## 完成后能做什么

- 识别 USB 摄像头的采集节点，预览画面、截图和录制原始视频。
- 在画面中检测 person，为连续出现的人显示临时跟踪 ID。
- 显示当前有效轨迹数与处理循环 FPS，导出人体框和中心坐标。
- 在同一批视频上比较输入尺寸、置信度阈值和跟踪器。

跟踪 ID 是当前运行中的轨迹编号。遮挡、交叉或离开画面再进入可能换号；本教程不识别人的姓名，也不驱动摄像头转动。

## 先准备这些

| 项目 | 起步选择 |
|---|---|
| 系统 | Ubuntu 22.04 / 24.04 图形桌面 |
| 摄像头 | 能通过 V4L2 采集的 USB 摄像头，通常为 UVC 设备 |
| Python 环境 | 已安装 Miniconda、Anaconda 或 Miniforge |
| 算力 | 先用 CPU；有可用的 NVIDIA GPU 再切换 |
| 网络 | 初次安装依赖和获取权重需要联网 |

虚拟机需要先把 USB 设备连接到 Ubuntu 客体。纯 SSH、WSL 和容器还需要额外处理设备与显示配置。

## 快速开始

### 1. 获取项目

```bash
git clone https://github.com/Uin06/ubuntu-usb-yolo-tutorial.git
cd ubuntu-usb-yolo-tutorial
```

没安装 Git 时，也可以在仓库的 **Code → Download ZIP** 下载、解压，然后在解压后的项目根目录打开终端。下面的命令都在项目根目录执行。

### 2. 创建 Conda 环境

```bash
sudo apt update
sudo apt install v4l-utils usbutils
conda env create -f environment.yml
conda activate yolo-usb
```

`environment.yml` 安装 Python、pip 和项目依赖；Conda 管理环境，pip 把依赖安装在这个环境里。若同名环境已经存在，先 `conda activate yolo-usb` 并检查依赖，避免重复创建。

若 Conda 提示激活前需要初始化，执行 `conda init bash`，重新打开终端后激活。尚未安装 Conda 可参考[官方安装入口](https://www.anaconda.com/docs/getting-started/installation)。

检查环境：

```bash
python -c "import sys, cv2, torch, ultralytics; print(sys.executable); print('OpenCV:', cv2.__version__); print('PyTorch:', torch.__version__); print('Ultralytics:', ultralytics.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 3. 找到 USB 摄像头并预览

```bash
v4l2-ctl --list-devices
python camera_test.py --source /dev/video0
```

把 `/dev/video0` 换成实际采集节点，USB 摄像头不一定编号为 0。预览窗口中按 **s** 截图、按 **q** 退出。

### 4. 跟踪画面中的人

先退出预览程序，再运行：

```bash
python person_track.py --source /dev/video0 --device cpu
```

默认加载 `yolo11n.pt`，使用 ByteTrack，仅处理 person 类。首次运行通常会下载权重并可能安装跟踪依赖；提前在联网环境跑通。窗口显示人体框、临时 ID、有效轨迹数和循环 FPS，按 **q** 退出。

若 CPU 较慢，可以比较较小输入：

```bash
python person_track.py --source /dev/video0 --device cpu --imgsz 320
```

有 NVIDIA GPU 的电脑先检查 `nvidia-smi` 和 `torch.cuda.is_available()`，按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装合适的版本，确认可用后再使用 `--device 0`。

### 5. 保存记录或使用离线视频

```bash
python person_track.py --source /dev/video0 --csv outputs/tracks.csv --seconds 60
```

CSV 只记录具有有效 ID 的当前框；空画面没有人物行。`elapsed_s` 是本次运行经过的时间，不是离线视频原始时间戳。重复运行时更换 CSV 文件名，程序不会覆盖已有文件。

多人共用一只摄像头时，先录制原始视频，再让每个人独立处理同一段素材：

```bash
python camera_test.py --source /dev/video0 --record outputs/single_person.avi --seconds 30
python person_track.py --source outputs/single_person.avi --device cpu
```

## 按什么顺序学习

| 阶段 | 动手目标 | 阅读 |
|---|---|---|
| 1—2 | 环境、原始采集、截图和录像 | [完整教程](docs/tutorial.md) |
| 3 | 图片与实时人体检测 | [YOLO 原论文阅读指引](docs/papers.md) |
| 4 | 多人跟踪与 ID 变化 | [完整教程](docs/tutorial.md)中的跟踪练习 |
| 5 | 中心点、目标锁定、轨迹、CSV | [完整教程](docs/tutorial.md)中的代码修改 |
| 6—7 | 参数实验、失败分析、独立复现 | [实验记录表](experiment_template.csv) |

建议两周、约 16—20 小时。每个人都独立修改代码和讲解结果。

## 文件导航

| 文件 | 用途 |
|---|---|
| [camera_test.py](camera_test.py) | 摄像头测试、截图、录制 |
| [person_track.py](person_track.py) | 人体跟踪、循环 FPS、CSV |
| [environment.yml](environment.yml) | Conda 起始环境 |
| [requirements.txt](requirements.txt) | pip 依赖清单 |
| [完整教程](docs/tutorial.md) | 分阶段练习、验收与故障排查 |
| [Conda 迁移说明](docs/conda-migration.md) | 从 venv 切换、环境检查 |
| [论文阅读指引](docs/papers.md) | YOLOv1 / YOLO11 的区别、ByteTrack |
| [GitHub 首次上传指南](docs/github-upload.md) | 建仓库、上传、分享、以后更新 |

## 常见问题

- **摄像头打不开：**检查采集节点、USB 连接、权限，关闭占用摄像头的浏览器或其他程序。
- **提示缺包：**先 `conda activate yolo-usb`，打印 `sys.executable`；编辑器也选择同一环境。
- **没有显示窗口：**检查是否在 Ubuntu 图形桌面、是否装了 headless 变体；无显示环境可使用 `--no-display` 并指定 `--seconds` 或视频文件。
- **跟踪编号频繁改变：**先区分漏检、换号和两人交换编号，再比较阈值、遮挡和跟踪器。
- **帧率与摄像头标称值不同：**这里显示的是处理循环速度，包含读取、模型和绘图等开销。

更多排查步骤见[完整教程](docs/tutorial.md)。

## 验证范围

已经检查 Python 语法、命令帮助、坐标与 CSV 逻辑，并用替身接口检查正常结束和采集失败时的资源释放。当前发布准备环境没有真实 USB 摄像头或 Conda，尚未进行真实设备、依赖安装与 YOLO 的联合实测。

不安装依赖也可以先运行坐标自检：

```bash
python3 person_track.py --self-check
```

完成真机验证后，可以补充真实演示、电脑配置、实际依赖版本和失败案例。`environment.yml` 是起始配置，尚未锁定全部依赖版本。

## 许可与参考

本教程代码采用 [GNU AGPL-3.0-or-later](LICENSE)。Ultralytics 软件与预训练权重遵循其自己的许可，参考[官方说明](https://docs.ultralytics.com/help/contributing/)。论文通过[官方链接](docs/papers.md)获取。

参考：[OpenCV 视频采集](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html) · [Ultralytics 跟踪接口](https://docs.ultralytics.com/modes/track/) · [Conda 环境管理](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)。
