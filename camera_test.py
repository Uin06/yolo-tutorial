# SPDX-License-Identifier: AGPL-3.0-or-later
"""USB 摄像头预览、截图和录制。先验证这一层，再运行 YOLO。"""
import argparse
import math
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--mjpg", action="store_true", help="仅在摄像头支持 MJPG 时使用")
    parser.add_argument("--record", help="原始视频的 .avi 路径；不会写入界面上的文字")
    parser.add_argument("--seconds", type=float, default=0, help="0 表示手动按 q 停止")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0 or args.seconds < 0:
        parser.error("分辨率和帧率必须为正数，seconds 不能为负数")
    if args.record:
        path = Path(args.record)
        if path.suffix.lower() != ".avi":
            parser.error("录制文件请使用 .avi 扩展名")
        if path.exists():
            parser.error("录制文件已存在，请换一个名称")
        path.parent.mkdir(parents=True, exist_ok=True)

    import cv2

    source = int(args.source) if args.source.isdecimal() else args.source
    live = isinstance(source, int) or str(source).startswith(("/dev/video", "/dev/v4l/"))
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2 if live else cv2.CAP_ANY)
    writer = None
    count = 0
    started = time.perf_counter()
    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 {args.source}，请检查设备节点、占用和权限")
        if live:
            if args.mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_FPS, args.fps)
        header_fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(header_fps) or header_fps <= 0:
            header_fps = args.fps
        started = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                if live:
                    raise RuntimeError("摄像头读取失败，请检查 USB 连接和摄像头占用")
                break
            count += 1
            height, width = frame.shape[:2]
            if count == 1:
                print(f"实际图像尺寸: {width}x{height}; 驱动报告 FPS: {header_fps:.2f}")
                if args.record:
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"MJPG"),
                                             header_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("无法创建视频，请检查输出路径和编码器")
            if writer is not None:
                writer.write(frame)
            elapsed = time.perf_counter() - started
            if not args.no_display:
                preview = frame.copy()
                cv2.putText(preview, f"capture loop FPS: {count / max(elapsed, 1e-6):.1f}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("USB camera | s: snapshot, q: quit", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    Path("outputs").mkdir(exist_ok=True)
                    path = Path("outputs") / f"snapshot_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError("截图保存失败")
                    print(f"已保存: {path}")
                if key == ord("q"):
                    break
            if args.seconds and elapsed >= args.seconds:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        elapsed = time.perf_counter() - started
        print(f"读取 {count} 帧，耗时 {elapsed:.2f}s，读取循环平均 {count / max(elapsed, 1e-6):.2f} FPS")


if __name__ == "__main__":
    main()
