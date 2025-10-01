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
    choices=["block", "platform", "slot", "circular"],
        default="platform",
    help="检测目标类型：block=色块物料，platform=凸台，slot=凹槽，circular=同时检测凸台+凹槽；默认进行凸台检测，可通过 --mode 指定",
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
    parser.add_argument(
        "--radius-min",
        type=int,
        default=None,
        help="手动设定圆检测的最小像素半径（覆盖默认 COMMON 配置）",
    )
    parser.add_argument(
        "--radius-max",
        type=int,
        default=None,
        help="手动设定圆检测的最大像素半径（覆盖默认 COMMON 配置）",
    )
    parser.add_argument(
        "--radius-margin-scale",
        type=float,
        default=None,
        help="调整半径自适应时的弹性比例，例如 0.3/0.6，默认 0.5",
    )
    parser.add_argument(
        "--disable-radius-adapt",
        action="store_true",
        help="关闭基于掩膜面积的半径自适应，仅使用静态配置",
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

    if args.radius_min is not None:
        creama.CIRCULAR_TARGET_PARAMS["COMMON"]["min_radius"] = max(4, args.radius_min)
    if args.radius_max is not None:
        creama.CIRCULAR_TARGET_PARAMS["COMMON"]["max_radius"] = max(
            creama.CIRCULAR_TARGET_PARAMS["COMMON"]["min_radius"] + 2, args.radius_max
        )

    if args.radius_margin_scale is not None:
        creama.RADIUS_MARGIN_SCALE = max(0.1, float(args.radius_margin_scale))

    if args.disable_radius_adapt:
        creama.ENABLE_RADIUS_ADAPT = False

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

    def smooth_center(track_key: str, raw_center: Tuple[int, int]) -> Tuple[int, int]:
        """统一的中心点指数平滑器，track_key 可区分颜色与目标类型。"""
        previous = smoothed_positions.get(track_key)
        if previous is None:
            smoothed_positions[track_key] = (float(raw_center[0]), float(raw_center[1]))
        else:
            smoothed_positions[track_key] = (
                (1 - smoothing_alpha) * previous[0] + smoothing_alpha * raw_center[0],
                (1 - smoothing_alpha) * previous[1] + smoothing_alpha * raw_center[1],
            )
        smoothed = smoothed_positions[track_key]
        return int(round(smoothed[0])), int(round(smoothed[1]))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] 摄像头读取失败，结束测试。")
                break

            display_frame = frame.copy()
            detections = {}
            overlay_lines = []  # 收集文本叠加信息 (label, center, color BGR)

            if args.mode == "block":
                for color in colors:
                    center = creama.color_blocks_position_WL(frame, color, args.size_threshold, display_frame)
                    if center:
                        track_key = color
                        smoothed_center_point = smooth_center(track_key, center)
                        detections[track_key] = smoothed_center_point
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
                        overlay_lines.append((color, smoothed_center_point, color_styles[color]))
                    else:
                        overlay_lines.append((color, None, color_styles[color]))
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
                        track_key = f"{args.mode}-{color}"
                        smoothed_center_point = smooth_center(track_key, center)
                        detections[track_key] = smoothed_center_point
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
                        overlay_lines.append((f"{args.mode}-{color}", smoothed_center_point, draw_color))
                    else:
                        overlay_lines.append((f"{args.mode}-{color}", None, color_styles.get(color, (255, 255, 255))))
            elif args.mode == "circular":
                # 同一帧内先后检测凸台/凹槽，便于对比新圆检测逻辑
                for target_label, readable in (("PLATFORM", "platform"), ("SLOT", "slot")):
                    for color in colors:
                        detection = creama.find_specific_target(frame, color, target_label)
                        label_key = f"{readable}-{color}"
                        draw_color = color_styles.get(color, (0, 255, 255))
                        # 为了区分凸台/凹槽，凹槽使用更浅的描边颜色
                        if target_label == "SLOT":
                            draw_color = tuple(min(255, int(c * 1.2)) for c in draw_color)

                        if detection:
                            center, contour = detection
                            smoothed_center_point = smooth_center(label_key, center)
                            detections[label_key] = smoothed_center_point
                            cv2.drawContours(display_frame, [contour], -1, draw_color, 2)
                            cv2.circle(display_frame, smoothed_center_point, 8, draw_color, -1)
                            cv2.putText(
                                display_frame,
                                f"{readable}:{color} {smoothed_center_point}",
                                (smoothed_center_point[0] + 10, smoothed_center_point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                draw_color,
                                2,
                                cv2.LINE_AA,
                            )
                            overlay_lines.append((label_key, smoothed_center_point, draw_color))
                        else:
                            overlay_lines.append((label_key, None, draw_color))

            line_y = 30
            for label, center, bgr in overlay_lines:
                text = f"{label:<15} {center if center else '未检测到'}"
                cv2.putText(
                    display_frame,
                    text,
                    (10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    bgr,
                    2,
                    cv2.LINE_AA,
                )
                line_y += 24

            now = time.time()
            if now - last_report >= args.print_interval:
                if args.mode == "block":
                    report_keys = colors
                elif args.mode in ("platform", "slot"):
                    report_keys = [f"{args.mode}-{color}" for color in colors]
                elif args.mode == "circular":
                    report_keys = [
                        f"platform-{color}" for color in colors
                    ] + [f"slot-{color}" for color in colors]
                else:
                    report_keys = list(detections.keys())

                for key in report_keys:
                    if key in detections:
                        print(f"[{args.mode}] {key} center -> {detections[key]}")
                    else:
                        print(f"[{args.mode}] {key}: 未检测到目标")
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
