import cv2
import matplotlib as plt
import numpy as np
import serial as ser
import serial
import threading
import time
import serial.tools.list_ports
from pyzbar.pyzbar import decode

# -------------------- 调试开关 --------------------
DEBUG_VISUAL = False  # 显示中间图像、掩膜等调试窗口
DEBUG_LOG = True      # 在终端打印调试信息

# -------------------- 阴影抑制配置 --------------------
# 这些常量用于在存在复杂光照/阴影的赛场环境中增强鲁棒性。
# 如果赛场光照非常稳定，可以关闭 ENABLE_SHADOW_SUPPRESSION 以节省算力。
ENABLE_SHADOW_SUPPRESSION = True
# V 通道(亮度)小于该阈值的像素会被视为阴影，随后从红色掩膜中剔除。
SHADOW_V_THRESHOLD = 110
# 用于净化阴影掩膜的小椭圆核，既能去掉孤立噪点也能保留阴影主干。若现场光线低，可灵活下调
_SHADOW_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# CLAHE 参数：clipLimit 控制对比度增强强度，tileGridSize 决定局部自适应区域大小。
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
_CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)


def log_debug(message):
    """根据开关打印调试信息，避免无关输出刷屏。"""
    if DEBUG_LOG:
        print(message)

# 定义串口接收和串口发送数组以及标志码unit任务码unit_target
receive = [0, 0, 0, 0]
send = [0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x77]
unit = 0
unit_target = 0

# 从这里开始到93行都是卡尔曼滤波算法的初始化步骤
kalman = cv2.KalmanFilter(2, 2)
kalman_2 = cv2.KalmanFilter(2, 2)
kalman_3 = cv2.KalmanFilter(2, 2)

kalman.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
kalman_2.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
kalman_3.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)

kalman.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)
kalman_2.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)
kalman_3.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-3
kalman_2.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-3
kalman_3.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-3

kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01
kalman_2.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01
kalman_3.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.01

kalman.statePre = np.array([[6], [6]], np.float32)
kalman_2.statePre = np.array([[6], [6]], np.float32)
kalman_3.statePre = np.array([[6], [6]], np.float32)

# 初始化变量
last_measurement = current_measurement = np.array((2, 2), np.float32)
last_prediction = current_prediction = np.array((2, 2), np.float32)

last_measurement_2 = current_measurement_2 = np.array((2, 2), np.float32)
last_prediction_2 = current_prediction_2 = np.array((2, 2), np.float32)

last_measurement_3 = current_measurement_3 = np.array((2, 2), np.float32)
last_prediction_3 = current_prediction_3 = np.array((2, 2), np.float32)


def kalman_filter(measured_value):
    global kalman, last_measurement, current_measurement, last_prediction, current_prediction

    last_measurement = current_measurement
    last_prediction = current_prediction

    # 更新测量值
    kalman.correct(measured_value)

    # 预测下一个状态
    current_prediction = kalman.predict()

    return current_prediction


def kalman_filter_2(measured_value):
    global kalman_2, last_measurement_2, current_measurement_2, last_prediction_2, current_prediction_2

    last_measurement_2 = current_measurement_2
    last_prediction_2 = current_prediction_2

    # 更新测量值
    kalman_2.correct(measured_value)

    # 预测下一个状态
    current_prediction_2 = kalman_2.predict()

    return current_prediction_2


def kalman_filter_3(measured_value):
    global kalman_3, last_measurement_3, current_measurement_3, last_prediction_3, current_prediction_3

    last_measurement_3 = current_measurement_3
    last_prediction_3 = current_prediction_3

    # 更新测量值
    kalman_3.correct(measured_value)

    # 预测下一个状态
    current_prediction_3 = kalman_3.predict()

    return current_prediction_3

# 创建串口进程函数
def uart_process():
    global receive, unit, unit_target
    
    while True:
        input_data = ser.read(4)
        com_input = list(input_data)
        if com_input:  # 如果读取结果非空，则输出
            log_debug(com_input)
            try:
                receive[0] = int(com_input[0])
                receive[1] = int(com_input[1])
                receive[2] = int(com_input[2])
                receive[3] = int(com_input[3])
            except:
                receive = [0, 0, 0, 0]
            log_debug(receive)
            if receive[0] == 102:  # 0x66
                unit = receive[1]
                unit_target = receive[2]

# 直接创建串口对象 
ser = serial.Serial(port="/dev/ttyS3", baudrate=115200, timeout=0.05)
serial_thread = threading.Thread(target=uart_process)
serial_thread.daemon = True
serial_thread.start()

PLATFORM_AREA_MIN = 20000  # 凸台最小面积
PLATFORM_AREA_MAX = 48000  # 凸台最大面积
SLOT_AREA_MIN = 34000      # 凹槽最小面积
SLOT_AREA_MAX = 60000      # 凹槽最大面积
BLOCK_AREA_MIN= 32000      # 物块最小面积
BLOCK_AREA_MAX= 58000      # 物块最大面积

# --- HSV颜色阈值 ---
color_thresholds = {'red': {'Lower1': np.array([170, 120, 120]), 'Upper1': np.array([180, 255, 255]),'Lower2': np.array([0, 120, 120]), 'Upper2': np.array([20, 255, 255])},
              'blue': {'Lower': np.array([100, 140, 110]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([38, 100, 60]), 'Upper': np.array([90, 255, 255])},
              }

# 定义一个函数，用于对检测到的二维码角点进行排序
def sort_points(pts):
    if pts is None or len(pts) == 0:
        return np.array([])
    pts = pts.reshape(-1, 2)
    hull = cv2.convexHull(pts)
    if hull is None or len(hull) == 0:
        return np.array([])
    return hull.reshape(-1, 2)

def display(img, bbox):
    """
    在图像上绘制二维码的边界框和中心点
    :param img: 输入图像
    :param bbox: 边界框坐标
    """
    if bbox is None or len(bbox) == 0:
        print("边界框为空或无效，无法绘制")
        return

    # 将边界框坐标转换为整数类型
    bbox = bbox.astype(int)
    # 打印边界框坐标
    print("Boundary Box Coordinates:", bbox)

    # 绘制边界框
    n = len(bbox)
    for j in range(n):
        pt1 = tuple(bbox[j])
        pt2 = tuple(bbox[(j + 1) % n])
        cv2.line(img, pt1, pt2, (255, 0, 0), 3)

    # 计算边界框的中心点
    center_x = int(np.mean(bbox[:, 0]))
    center_y = int(np.mean(bbox[:, 1]))

    # 绘制中心点
    cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)

def color_blocks_position_WL(img, color, size_code, output_img):
    """借鉴开源逻辑的颜色块定位，增加平滑及可选调试输出。"""
    if img is None:
        log_debug("无画面，跳过颜色检测")
        return None

    color_dist = {
        'red': {
            'Lower1': np.array([170, 43, 46]), 'Upper1': np.array([180, 255, 255]),
            'Lower2': np.array([0, 43, 46]), 'Upper2': np.array([10, 255, 255])
        },
        'blue': {'Lower': np.array([100, 140, 110]), 'Upper': np.array([124, 255, 255])},
        'green': {'Lower': np.array([38, 100, 60]), 'Upper': np.array([90, 255, 255])},
    }

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    eroded = cv2.erode(hsv, None, iterations=2)
    dilated = cv2.dilate(eroded, None, iterations=2)

    if color == 'red':
        mask1 = cv2.inRange(dilated, color_dist['red']['Lower1'], color_dist['red']['Upper1'])
        mask2 = cv2.inRange(dilated, color_dist['red']['Lower2'], color_dist['red']['Upper2'])
        mask = cv2.add(mask1, mask2)
    else:
        mask = cv2.inRange(dilated, color_dist[color]['Lower'], color_dist[color]['Upper'])

    if DEBUG_VISUAL:
        cv2.imshow(f"mask_{color}", mask)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return None

    best_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_cnt)
    if area <= size_code:
        return None

    rect = cv2.minAreaRect(best_cnt)
    box = cv2.boxPoints(rect)
    box = np.int_(box)
    cx, cy = rect[0]

    if DEBUG_VISUAL:
        cv2.drawContours(output_img, [box], -1, (0, 0, 255), 2)
        cv2.circle(output_img, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    log_debug(f"{color} block area={area:.0f} center=({int(cx)}, {int(cy)})")
    return int(cx), int(cy)

def find_specific_target(frame, color_to_find, target_type):
    """
    根据指定的颜色和目标类型，寻找唯一的目标。
    :param frame: 摄像头画面
    :param color_to_find: 'red', 'green', 或 'blue'
    :param target_type: 'PLATFORM', 'SLOT', 或 'STACK_BASE'
    :return: 目标的中心像素坐标 (cx, cy) 或 None
    """
    # 步骤 1: 预处理 - 先使用双边滤波抑制色彩噪声，再轻微高斯模糊平滑
    bilateral_frame = cv2.bilateralFilter(frame, 7, 75, 75)
    blurred_frame = cv2.GaussianBlur(bilateral_frame, (5, 5), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    if ENABLE_SHADOW_SUPPRESSION:
        # 对 V 通道做局部对比度增强，可缩小阴影与高光的亮度差距，避免阴影被误判为红色主体。
        h_channel, s_channel, v_channel = cv2.split(hsv_frame)
        v_channel = _CLAHE.apply(v_channel)
        hsv_frame = cv2.merge((h_channel, s_channel, v_channel))
    
    # 步骤 2: 颜色分割 (已修复对红色的特殊处理)
    mask = None
    if color_to_find == 'red':
        # 红色在HSV空间有两个范围，需要分别处理后合并
        lower1, upper1 = color_thresholds['red']['Lower1'], color_thresholds['red']['Upper1']
        lower2, upper2 = color_thresholds['red']['Lower2'], color_thresholds['red']['Upper2']
        mask1 = cv2.inRange(hsv_frame, lower1, upper1)
        mask2 = cv2.inRange(hsv_frame, lower2, upper2)
        mask = cv2.add(mask1, mask2) # 合并两个红色范围的掩码

        if ENABLE_SHADOW_SUPPRESSION:
            # 利用 V 通道生成阴影掩膜，随后从红色掩膜中剔除这些暗区，降低阴影凸起的概率。
            v_channel = hsv_frame[:, :, 2]
            shadow_mask = cv2.inRange(v_channel, 0, SHADOW_V_THRESHOLD)
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, _SHADOW_KERNEL, iterations=1)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow_mask))
    else:
        # 其他颜色直接获取阈值
        lower_hsv = color_thresholds[color_to_find]['Lower']
        upper_hsv = color_thresholds[color_to_find]['Upper']
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    # 步骤 3: 形态学处理 - 去除噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # 使用开运算去除小噪点，再用较大的闭运算填平缝隙/毛刺
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    # 中值滤波在保持轮廓面积的同时去掉孤立毛刺
    cleaned_mask = cv2.medianBlur(cleaned_mask, 5)
    # 轻度高斯模糊 + 二值化，让轮廓边沿更顺滑
    cleaned_mask = cv2.GaussianBlur(cleaned_mask, (7, 7), 0)
    _, cleaned_mask = cv2.threshold(cleaned_mask, 40, 255, cv2.THRESH_BINARY)

    if DEBUG_VISUAL:
        cv2.imshow("mask_raw", mask)
        cv2.imshow("mask_cleaned", cleaned_mask)
        if ENABLE_SHADOW_SUPPRESSION and color_to_find == 'red':
            cv2.imshow("shadow_mask", shadow_mask)
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 步骤 4: 根据“状态值”(target_type) 进行精确过滤
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 计算中心点
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if DEBUG_VISUAL:
            cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

        # 核心逻辑：根据状态值选择不同的判断标准
        if target_type == "PLATFORM":
            if PLATFORM_AREA_MIN < area < PLATFORM_AREA_MAX:
                log_debug(f"{color_to_find} platform area={area:.0f} center=({cx}, {cy})")
                return ((cx, cy), cnt)  # 找到了！返回中心点和轮廓

        elif target_type == "SLOT":
            if SLOT_AREA_MIN < area < SLOT_AREA_MAX:
                log_debug(f"{color_to_find} slot area={area:.0f} center=({cx}, {cy})")
                return ((cx, cy), cnt)  # 找到了！返回中心点和轮廓

        elif target_type == "STACK_BASE":
            # 寻找码垛基座（已放置的物块）的逻辑会更复杂一些
            # 简单版本：它的面积应该和物块面积接近
            if BLOCK_AREA_MIN < area < BLOCK_AREA_MAX:
                # 这里还可以增加判断，比如它是否在一个凹槽的轮廓内部
                log_debug(f"{color_to_find} stack area={area:.0f} center=({cx}, {cy})")
                return ((cx, cy), cnt)  # 找到了！返回中心点和轮廓

    return None # 如果循环结束还没找到，就返回None

def detect_platforms(frame):
    """
    检测所有凸台，显示其轮廓并返回中心点坐标。
    :param frame: 输入的图像帧
    :return: 检测到的凸台中心点坐标列表
    """
    platform_positions = []
    try:
        colors_to_check = ['red', 'green', 'blue']
        for color in colors_to_check:
            platform_result = find_specific_target(frame, color, 'PLATFORM')
            if platform_result:
                platform_pos, platform_contour = platform_result
                if DEBUG_VISUAL:
                    cv2.drawContours(frame, [platform_contour], -1, (0, 255, 255), 2)
                    cv2.putText(frame, f"{color.capitalize()} Platform", (platform_pos[0] + 10, platform_pos[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                log_debug(f"检测到 {color} 凸台，中心坐标: {platform_pos}")
                platform_positions.append(platform_pos)
    except Exception as e:
        log_debug(f"处理凸台检测时发生错误: {e}")
    return platform_positions

def detect_slots(frame):
    """
    检测所有凹槽，显示其轮廓并返回中心点坐标。
    :param frame: 输入的图像帧
    :return: 检测到的凹槽中心点坐标列表
    """
    slot_positions = []
    try:
        colors_to_check = ['red', 'green', 'blue']
        for color in colors_to_check:
            slot_result = find_specific_target(frame, color, 'SLOT')
            if slot_result:
                slot_pos, slot_contour = slot_result
                if DEBUG_VISUAL:
                    cv2.drawContours(frame, [slot_contour], -1, (255, 255, 0), 2)
                    cv2.putText(frame, f"{color.capitalize()} Slot", (slot_pos[0] + 10, slot_pos[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                log_debug(f"检测到 {color} 凹槽，中心坐标: {slot_pos}")
                slot_positions.append(slot_pos)
    except Exception as e:
        log_debug(f"处理凹槽检测时发生错误: {e}")
    return slot_positions

if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 主函数默认使用摄像头0

    while True:
        success, img0 = cap.read()

        if success:
            if (unit == 1):
                # 物料盘识别定位
                try:
                    # 检测所有凸台位置
                    platform_positions = detect_platforms(img0)
                    if platform_positions:
                        # 取第一个检测到的凸台作为物料盘中心
                        center_x, center_y = platform_positions[0]
                        # 使用卡尔曼滤波平滑中心坐标
                        measurement = np.array([[center_x], [center_y]], dtype=np.float32)
                        prediction = kalman_filter(measurement)
                        qx0 = int(prediction[0][0])
                        qy0 = int(prediction[1][0])

                        send[1] = 0x01
                        send[2] = (qx0 & 0xff00) >> 8
                        send[3] = (qx0 & 0xff)
                        send[4] = (qy0 & 0xff00) >> 8
                        send[5] = (qy0 & 0xff)
                        send[6] = 0x77
                        FH = bytearray(send)
                        write_len = ser.write(FH)
                        print(f"物料盘中心坐标: ({qx0}, {qy0})")
                        print(send)
                except:
                    pass

            elif (unit == 2):
                # 物料识别
                try:
                    color_map = {1: 'red', 2: 'green', 3: 'blue'}
                    if unit_target in color_map:
                        color_name = color_map[unit_target]
                        result = color_blocks_position_WL(img0, color_name, 2000, img0)
                        if result:
                            measured_x, measured_y = result
                            # 卡尔曼滤波降低抖动
                            z = np.array([[measured_x], [measured_y]], dtype=np.float32)
                            x = kalman_filter(z)
                            pos_x = int(x[0][0])
                            pos_y = int(x[1][0])
                            
                            send[1] = 0x02
                            send[2] = unit_target  # 颜色码
                            send[3] = (pos_x & 0xff00) >> 8
                            send[4] = (pos_x & 0xff)
                            send[5] = (pos_y & 0xff00) >> 8
                            send[6] = (pos_y & 0xff)
                            send[7] = 0x77
                            FH = bytearray(send)
                            write_len = ser.write(FH)
                            print(f"{color_name}物料坐标: ({pos_x}, {pos_y})")
                            print(send)
                except:
                    pass

            elif (unit == 10):  # 0x0A - 凸台识别
                # 凸台识别
                try:
                    color_map = {1: 'red', 2: 'green', 3: 'blue'}
                    if unit_target in color_map:
                        color_name = color_map[unit_target]
                        platform_result = find_specific_target(img0, color_name, 'PLATFORM')
                        if platform_result:
                            platform_pos, platform_contour = platform_result
                            # 第二组滤波器专门平滑凸台位置
                            z = np.array([[platform_pos[0]], [platform_pos[1]]], dtype=np.float32)
                            pred = kalman_filter_2(z)
                            px = int(pred[0][0])
                            py = int(pred[1][0])
                            
                            send[1] = 0x0A
                            send[2] = (px & 0xff00) >> 8
                            send[3] = (px & 0xff)
                            send[4] = (py & 0xff00) >> 8
                            send[5] = (py & 0xff)
                            send[6] = 0x77
                            FH = bytearray(send)
                            write_len = ser.write(FH)
                            print(f"{color_name}凸台坐标: ({px}, {py})")
                            print(send)
                except:
                    pass

            elif (unit == 11):  # 0x0B - 凹槽识别
                # 凹槽识别
                try:
                    color_map = {1: 'red', 2: 'green', 3: 'blue'}
                    if unit_target in color_map:
                        color_name = color_map[unit_target]
                        slot_result = find_specific_target(img0, color_name, 'SLOT')
                        if slot_result:
                            slot_pos, slot_contour = slot_result
                            # 第三组滤波器专门平滑凹槽位置
                            z = np.array([[slot_pos[0]], [slot_pos[1]]], dtype=np.float32)
                            pred = kalman_filter_3(z)
                            sx = int(pred[0][0])
                            sy = int(pred[1][0])
                            
                            send[1] = 0x0B
                            send[2] = (sx & 0xff00) >> 8
                            send[3] = (sx & 0xff)
                            send[4] = (sy & 0xff00) >> 8
                            send[5] = (sy & 0xff)
                            send[6] = 0x77
                            FH = bytearray(send)
                            write_len = ser.write(FH)
                            print(f"{color_name}凹槽坐标: ({sx}, {sy})")
                            print(send)
                except:
                    pass

            elif (unit == 9):
                # OpenCV二维码检测器
                qrDecoder = cv2.QRCodeDetector()

                # 打开二号摄像头
                cap2 = cv2.VideoCapture(2)
                if not cap2.isOpened():
                    print("无法打开摄像头")

                else:
                    try:
                        # 从摄像头读取一帧图像
                        success, img0 = cap2.read()
                        if not success:
                            print("无法读取摄像头图像")
                            continue

                        # 检测并解码二维码
                        # 使用 detectAndDecode 方法，它会返回三个值：数据、边界框和矫正后的图像
                        # 但我们只需要前两个
                        data, bbox, _ = qrDecoder.detectAndDecode(img0)

                        if data and bbox is not None:
                            print(f"检测到二维码: 数据 = '{data}'")

                            # 在画面上绘制边界框和中心点
                            display(img0, bbox)
                            cv2.imshow("Results", img0)

                            send[1] = 0x09
                            
                            send[2] = int(data[0])
                            send[3] = int(data[1])
                            send[4] = int(data[2])
                            send[5] = int(data[4])
                            send[6] = int(data[5])
                            send[7] = int(data[6])
                            send[8] = 0x77
                            print(send)
                            FH = bytearray(send)
                            write_len = ser.write(FH)

                        else:
                            print("未检测到二维码")
                            cv2.imshow("Results", img0)
                    except:
                        print("error")
                    cap2.release()
                    cv2.destroyAllWindows()

            else:
                send[1] = 0xAA
                send[2] = 0x03
                send[3] = 0x77
                FH = bytearray(send)
                write_len = ser.write(FH)
                print(send)
                
            cv2.imshow("videoo", img0)
        else:
            break

        m_key = cv2.waitKey(1) & 0xFF

        if m_key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
