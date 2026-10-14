# SPDX-License-Identifier: AGPL-3.0-or-later
"""YOLO 人体跟踪基线：临时 ID、当前跟踪数量、循环 FPS、可选 CSV。"""
import argparse
import csv
import time
from pathlib import Path


CSV_HEADER = ["frame_index", "elapsed_s", "track_id", "confidence", "x1", "y1", "x2", "y2",
              "center_x", "center_y", "error_x", "error_y"]


def record_row(frame_index, elapsed, track_id, confidence, xyxy, width, height):
    """elapsed_s 是本次运行经过的墙钟时间，error_x/y 是相对图像中心的归一化偏差。"""
    x1, y1, x2, y2 = map(float, xyxy)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [frame_index, round(elapsed, 4), int(track_id), round(float(confidence), 4),
            x1, y1, x2, y2, cx, cy, 2 * cx / width - 1, 2 * cy / height - 1]


def self_check():
    row = record_row(1, 0.125, 7, 0.9, [10, 20, 110, 220], 200, 400)
    assert len(row) == len(CSV_HEADER)
    assert row[2] == 7 and row[8:10] == [60.0, 120.0]
    assert abs(row[10] + 0.4) < 1e-9 and abs(row[11] + 0.4) < 1e-9
    centered = record_row(2, 0.2, 7, 0.8, [0, 0, 200, 400], 200, 400)
    assert centered[10:] == [0.0, 0.0]
    print("CSV 列顺序、中心坐标和归一化偏差检查通过")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/dev/video0", help="设备节点、整数索引或视频文件")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="cpu", help="cpu；NVIDIA GPU 可用时填 0")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--csv", help="仅保存具有有效跟踪 ID 的当前框；不会覆盖已有文件")
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--self-check", action="store_true", help="不需要摄像头和第三方库")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not 0 <= args.conf <= 1 or args.imgsz <= 0 or min(args.seconds, args.max_frames) < 0:
        parser.error("请检查 conf、imgsz、seconds 和 max-frames 的范围")
    if args.csv:
        path = Path(args.csv)
        if path.exists():
            parser.error("CSV 已存在，请换一个名称")
        path.parent.mkdir(parents=True, exist_ok=True)

    import cv2
    from ultralytics import YOLO

    # 模型只加载一次。此练习使用 COCO 预训练检测模型，类别 0 是 person。
    model = YOLO(args.model)
    if model.names[0] != "person":
        raise RuntimeError("本脚本要求类别 0 为 person 的模型，请使用提供的 COCO 预训练模型")
    source = int(args.source) if args.source.isdecimal() else args.source
    live = isinstance(source, int) or str(source).startswith(("/dev/video", "/dev/v4l/"))
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2 if live else cv2.CAP_ANY)
    csv_file = None
    csv_writer = None
    count, window_count, fps = 0, 0, 0.0
    started = window_started = time.perf_counter()
    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 {args.source}，请先运行 camera_test.py")
        if live:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if args.csv:
            csv_file = open(args.csv, "x", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_HEADER)
        started = window_started = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                if live:
                    raise RuntimeError("摄像头读取失败，请检查连接；已保存的 CSV 会保留")
                break
            count += 1
            # persist=True 仅用于同一段视频的连续帧。
            result = model.track(frame, persist=True, tracker=args.tracker, classes=[0],
                                 conf=args.conf, imgsz=args.imgsz, device=args.device,
                                 verbose=False)[0]
            boxes = result.boxes
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
            elapsed = time.perf_counter() - started
            if csv_writer is not None and ids:
                coords = boxes.xyxy.cpu().tolist()
                scores = boxes.conf.cpu().tolist()
                height, width = frame.shape[:2]
                for track_id, confidence, xyxy in zip(ids, scores, coords):
                    csv_writer.writerow(record_row(count, elapsed, track_id, confidence,
                                                   xyxy, width, height))
            annotated = result.plot()
            window_count += 1
            now = time.perf_counter()
            if now - window_started >= 1:
                fps = window_count / (now - window_started)
                window_count, window_started = 0, now
            cv2.putText(annotated, f"tracks now: {len(ids)} | loop FPS: {fps:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if not args.no_display:
                cv2.imshow("YOLO person tracking | q: quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.seconds and elapsed >= args.seconds:
                break
            if args.max_frames and count >= args.max_frames:
                break
    finally:
        cap.release()
        if csv_file is not None:
            csv_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        elapsed = time.perf_counter() - started
        print(f"处理 {count} 帧，耗时 {elapsed:.2f}s，循环平均 {count / max(elapsed, 1e-6):.2f} FPS")
        print("ID 是本次运行的临时轨迹编号；当前轨迹数和出现过的 ID 总数都不等于真实总人数。")


if __name__ == "__main__":
    main()
