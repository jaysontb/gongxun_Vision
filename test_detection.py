import argparse
import sys
import time
from typing import Any, Dict, Tuple

import cv2


class MockSerial:  # pylint: disable=too-few-public-methods
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout = kwargs.get("timeout", 0.01)

    def read(self, size: int = 1) -> bytes:
        time.sleep(self.timeout)
        return b"\x00" * size

    def write(self, data: bytes) -> int:
        return len(data)

    def close(self) -> None:
        pass


def load_creama(use_real_uart: bool = False) -> Any:
    """Import Creama and optionally mock UART so the script runs standalone."""

    if use_real_uart:
        try:
            import serial  # type: ignore  # noqa: F401
        except ImportError:
            print("[Warn] 未安装 pyserial，无法使用真实串口，改用模拟串口。")
            use_real_uart = False

    if not use_real_uart:
        try:
            import serial  # type: ignore
            serial.Serial = MockSerial  # type: ignore[assignment]
        except ImportError:
            pass

    try:
        import Creama  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        print("[Error] 无法导入 Creama.py:", exc)
        print("提示：确认 Creama.py 位于相同目录，或解决其中的语法/依赖问题后再试。")
        sys.exit(1)

    if not use_real_uart and hasattr(Creama, "ser"):
        Creama.ser = MockSerial(port="COM-MOCK", baudrate=115200, timeout=0.01)  # type: ignore[attr-defined]
    return Creama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Creama 中的视觉函数快速测试颜色物料/凹槽检测，并打印中心坐标"
    )
    parser.add_argument(
        "--mode",
        choices=["block", "platform", "slot"],
        default="platform",
        help="检测目标类型：block=色块物料，platform=凸台，slot=凹槽；默认进行凸台检测，可通过 --mode 指定",
    )
    parser.add_argument(
        "--colors",
        nargs="+",
        choices=["red", "green", "blue"],
        default=["red", "green", "blue"],
        help="需要检测的颜色列表，默认同时检测红绿蓝三种物块",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头索引（默认 0）",
    )
    parser.add_argument(
        "--size-threshold",
        type=int,
        default=2000,
        help="色块面积阈值，仅在 mode=block 时生效",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="打印检测结果的最小时间间隔（秒）",
    )
    parser.add_argument(
        "--debug-visual",
        action="store_true",
        help="启用 Creama.DEBUG_VISUAL，用于查看中间掩膜等调试窗口",
    )
    parser.add_argument(
        "--debug-log",
        action="store_true",
        help="启用 Creama.DEBUG_LOG，在终端输出详细日志",
    )
    parser.add_argument(
        "--use-real-uart",
        action="store_true",
        help="使用真实串口（默认使用模拟串口，适合本地PC测试）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    creama = load_creama(args.use_real_uart)

    if args.use_real_uart:
        print("[Info] 使用真实串口，请确保设备已连接且 Creama.ser 正确初始化。")
    else:
        print("[Info] 已启用模拟串口，所有串口读写将被截获以便在 PC 上独立运行。")

    creama.DEBUG_VISUAL = args.debug_visual
    creama.DEBUG_LOG = args.debug_log

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
    if not cap.isOpened():
        print(f"[Error] 无法打开摄像头索引 {args.camera}")
        sys.exit(1)

    print("按 Ctrl+C 或在窗口中按 q 结束。")

    colors = list(dict.fromkeys(args.colors))
    color_styles = {
        "red": (0, 0, 255),
        "green": (0, 200, 0),
        "blue": (255, 120, 0),
    }
    print(f"当前检测模式: {args.mode}")
    print("正在检测颜色: " + ", ".join(colors))
    print(f"控制台每 {args.print_interval:.1f}s 推送一次检测结果摘要。")

    type_map = {
        "platform": "PLATFORM",
        "slot": "SLOT",
    }

    last_report = 0.0
    smoothing_alpha = 0.35
    smoothed_positions: Dict[str, Tuple[float, float]] = {}

    def smooth_center(color: str, raw_center: Tuple[int, int]) -> Tuple[int, int]:
        previous = smoothed_positions.get(color)
        if previous is None:
            smoothed_positions[color] = (float(raw_center[0]), float(raw_center[1]))
        else:
            smoothed_positions[color] = (
                (1 - smoothing_alpha) * previous[0] + smoothing_alpha * raw_center[0],
                (1 - smoothing_alpha) * previous[1] + smoothing_alpha * raw_center[1],
            )
        smoothed = smoothed_positions[color]
        return int(round(smoothed[0])), int(round(smoothed[1]))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] 摄像头读取失败，结束测试。")
                break

            display_frame = frame.copy()
            detections = {}
            overlay_lines = []

            if args.mode == "block":
                for color in colors:
                    center = creama.color_blocks_position_WL(frame, color, args.size_threshold, display_frame)
                    if center:
                        smoothed_center_point = smooth_center(color, center)
                        detections[color] = smoothed_center_point
                        cv2.circle(display_frame, smoothed_center_point, 8, color_styles[color], -1)
                        cv2.putText(
                            display_frame,
                            f"{color}:{smoothed_center_point}",
                            (smoothed_center_point[0] + 10, smoothed_center_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color_styles[color],
                            2,
                            cv2.LINE_AA,
                        )
                        overlay_lines.append((color, smoothed_center_point))
                    else:
                        overlay_lines.append((color, None))
            elif args.mode in type_map:
                target_label = type_map[args.mode]
                for color in colors:
                    detection = creama.find_specific_target(
                        frame,
                        color,
                        target_label,
                    )
                    if detection:
                        center, contour = detection
                        smoothed_center_point = smooth_center(color, center)
                        detections[color] = smoothed_center_point
                        draw_color = color_styles.get(color, (0, 255, 255))
                        cv2.drawContours(display_frame, [contour], -1, draw_color, 2)
                        cv2.circle(display_frame, smoothed_center_point, 8, draw_color, -1)
                        cv2.putText(
                            display_frame,
                            f"{args.mode}:{color} {smoothed_center_point}",
                            (smoothed_center_point[0] + 10, smoothed_center_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            draw_color,
                            2,
                            cv2.LINE_AA,
                        )
                        overlay_lines.append((color, smoothed_center_point))
                    else:
                        overlay_lines.append((color, None))

            line_y = 30
            for color, center in overlay_lines:
                label = color if args.mode == "block" else f"{args.mode}-{color}"
                text = f"{label:<15} {center if center else '未检测到'}"
                cv2.putText(
                    display_frame,
                    text,
                    (10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color_styles.get(color, (255, 255, 255)),
                    2,
                    cv2.LINE_AA,
                )
                line_y += 24

            now = time.time()
            if now - last_report >= args.print_interval:
                report_colors = colors if args.mode in ("block", "platform", "slot") else list(detections.keys())
                for color in report_colors:
                    if color in detections:
                        print(f"[{args.mode}] {color} center -> {detections[color]}")
                    else:
                        print(f"[{args.mode}] {color}: 未检测到目标")
                last_report = now

            cv2.imshow("vision_test", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n检测已终止。")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
