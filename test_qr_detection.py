#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二维码识别测试程序
独立于主程序的二维码识别测试工具
"""

import cv2
import numpy as np
import sys
import os

def display(img, bbox):
    """
    在图像上绘制二维码的边界框和中心点
    :param img: 输入图像
    :param bbox: 边界框坐标
    """
    if bbox is None or len(bbox) == 0:
        print("边界框为空或无效，无法绘制")
        return

    # 拉平成 N×2，后续处理都用它
    bbox = np.squeeze(bbox)  # -> (4, 2)
    if bbox.ndim != 2 or bbox.shape[1] != 2:
        print("边界框格式异常:", bbox.shape)
        return
    bbox = bbox.astype(int)

    print("Boundary Box Coordinates:", bbox)

    n = len(bbox)
    for j in range(n):
        pt1 = tuple(bbox[j])
        pt2 = tuple(bbox[(j + 1) % n])
        cv2.line(img, pt1, pt2, (255, 0, 0), 3)

    center_x = int(np.mean(bbox[:, 0]))
    center_y = int(np.mean(bbox[:, 1]))
    cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)

def test_qr_detection():
    """
    独立的二维码识别测试函数
    用于测试二维码识别功能，无需串口指令触发
    """
    print("开始二维码识别测试...")
    print("=" * 50)

    # 使用OpenCV的QR码检测器
    qrDecoder = cv2.QRCodeDetector()

    # 尝试打开摄像头10 (二维码专用摄像头)
    cap_qr = cv2.VideoCapture(1)

    if not cap_qr.isOpened():
        print("无法打开摄像头10，尝试摄像头0...")
        cap_qr = cv2.VideoCapture(0)

    if not cap_qr.isOpened():
        print("错误：无法打开任何摄像头！")
        print("请检查摄像头连接")
        return False

    print("二维码摄像头已打开")
    print("按 'q' 键退出测试")
    print("按 's' 键保存当前图像")
    print("-" * 30)

    frame_count = 0
    max_frames = 500  # 最多测试500帧
    last_qr_data = None

    try:
        while frame_count < max_frames:
            # 从摄像头读取一帧图像
            success, img = cap_qr.read()

            if not success:
                print(f"无法读取摄像头图像 (第{frame_count+1}帧)")
                frame_count += 1
                continue

            # 检测并解码二维码
            data, bbox, _ = qrDecoder.detectAndDecode(img)

            # 显示当前帧
            display_img = img.copy()
            cv2.putText(display_img, f"Frame: {frame_count+1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if data and bbox is not None:
                print(f"\n🎉 检测到二维码: '{data}'")

                # 在画面上绘制边界框和中心点
                display(display_img, bbox)

                # 计算并显示中心点
                center_x = int(np.mean(bbox[:, 0]))
                center_y = int(np.mean(bbox[:, 1]))
                cv2.circle(display_img, (center_x, center_y), 8, (0, 255, 0), -1)
                cv2.putText(display_img, f"Center: ({center_x}, {center_y})",
                           (center_x + 10, center_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 解析二维码内容并确定装配模式
                assembly_mode = 0x00

                # 多种解析方案
                if '1' in data or '同色' in data or 'SAME' in data.upper():
                    assembly_mode = 0x01  # 同色装配
                elif '2' in data or '异色' in data or 'DIFF' in data.upper():
                    assembly_mode = 0x02  # 异色错配
                else:
                    # 尝试解析第一个字符为数字
                    try:
                        mode_num = int(data[0])
                        if mode_num == 1:
                            assembly_mode = 0x01
                        elif mode_num == 2:
                            assembly_mode = 0x02
                    except:
                        assembly_mode = 0xFF  # 无法识别

                # 显示解析结果
                mode_text = "同色装配" if assembly_mode == 0x01 else "异色错配" if assembly_mode == 0x02 else "未知模式"
                cv2.putText(display_img, f"Mode: {mode_text}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                print(f"📊 识别结果: {mode_text} (0x{assembly_mode:02X})")
                print(f"📍 中心坐标: ({center_x}, {center_y})")
                print(f"📐 边界框大小: {bbox.shape}")

                # 如果数据发生变化，打印详细信息
                if data != last_qr_data:
                    print(f"📄 二维码内容: {data}")
                    last_qr_data = data

                print("-" * 30)

            else:
                cv2.putText(display_img, "No QR Code Detected", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示图像
            cv2.imshow("QR Code Test", display_img)
            frame_count += 1

            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出测试")
                break
            elif key == ord('s'):
                filename = f"qr_test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, img)
                print(f"图像已保存为: {filename}")

    except KeyboardInterrupt:
        print("\n收到中断信号，正在退出...")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        # 清理资源
        cap_qr.release()
        cv2.destroyAllWindows()
        print(f"\n测试完成，共处理了{frame_count}帧图像")
        return frame_count > 0

def main():
    """主函数"""
    print("二维码识别测试工具")
    print("=" * 50)
    print("此工具用于独立测试二维码识别功能")
    print("无需串口通信，直接测试摄像头和识别算法")
    print()

    # 检查OpenCV版本
    print(f"OpenCV版本: {cv2.__version__}")

    # 运行测试
    success = test_qr_detection()

    if success:
        print("✅ 测试成功完成")
    else:
        print("❌ 测试失败或未检测到二维码")

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())