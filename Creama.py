import cv2
import matplotlib as plt
import numpy as np
import serial as ser
import serial
import threading
import time
import serial.tools.list_ports
from pyzbar.pyzbar import decode

# 定义串口接收和串口发送数组以及标志码unit任务码unit_target
receive = [0, 0, 0, 0]
send = [0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x77]
unit = 0
unit_target = 0

# 创建串口进程函数
def uart_process():
    global receive, unit, unit_target
    
    while True:
        input_data = ser.read(4)
        com_input = list(input_data)
        if com_input:  # 如果读取结果非空，则输出
            print(com_input)
            try:
                receive[0] = int(com_input[0])
                receive[1] = int(com_input[1])
                receive[2] = int(com_input[2])
                receive[3] = int(com_input[3])
            except:
                receive = [0, 0, 0, 0]
            print(receive)
            if receive[0] == 102:  # 0x66
                unit = receive[1]
                unit_target = receive[2]

# 直接创建串口对象 
ser = serial.Serial(port="COM3", baudrate=115200, timeout=0.05)
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
color_thresholds = {'red': {'Lower1': np.array([170, 43, 46]), 'Upper1': np.array([180, 255, 255]),'Lower2': np.array([0, 80, 80]), 'Upper2': np.array([20, 255, 255])},
              'blue': {'Lower': np.array([100, 140, 110]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([38, 100, 60]), 'Upper': np.array([90, 255, 255])},
              }


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

def color_blocks_position_WL (img, color, size_code, output_img):
    """
    根据命令借助HSV颜色阈值筛选识别指定的物料
    :param img: 输入的图像
    :param color: 要追踪的颜色目标对象（RGB）
    :param size_code: 颜色色块面积限幅（用于灵活筛选追踪颜色大小）
    :param output_img: 用于绘制轮廓和点的输出图像
    :return: 相应色块的中心坐标
    """
    # 定义颜色识别阈值（HSV）
    color_dist = {'red': {'Lower1': np.array([170, 43, 46]), 'Upper1': np.array([180, 255, 255]),'Lower2': np.array([0, 43, 46]), 'Upper2': np.array([10, 255, 255])},
              'blue': {'Lower': np.array([100, 140, 110]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([38, 100, 60]), 'Upper': np.array([90, 255, 255])},
              }
    ball_color = color
    if img is not None:
        gs_img = cv2.GaussianBlur(img, (5, 5), 0)                     # 高斯模糊
        hsv_img = cv2.cvtColor(gs_img, cv2.COLOR_BGR2HSV)                 # 转化成HSV图像
        erode_hsv = cv2.erode(hsv_img, None, iterations=2)                   # 腐蚀 粗的变细（平整边缘）
        dilate_hsv = cv2.dilate(erode_hsv, None, iterations=2)               # 膨胀 细的变粗（连接断开区域）
        inRange_hsv = None
        # 红色在HSV空间阈值有两部分所以需要特殊处理
        if (ball_color == 'red'):
            inRange_hsv1 = cv2.inRange(dilate_hsv, color_dist[ball_color]['Lower1'], color_dist[ball_color]['Upper1'])
            res1 = cv2.bitwise_and(dilate_hsv, dilate_hsv, mask=inRange_hsv1)
            inRange_hsv2 = cv2.inRange(dilate_hsv, color_dist[ball_color]['Lower2'], color_dist[ball_color]['Upper2'])
            res2 = cv2.bitwise_and(dilate_hsv, dilate_hsv, mask=inRange_hsv2)
            inRange_hsv = inRange_hsv1 + inRange_hsv2
        else:
            inRange_hsv = cv2.inRange(dilate_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
        # cv2.imshow("end2",dilate_hsv) # 移除中间图像显示
        # cv2.imshow("end",inRange_hsv) # 移除中间图像显示
        cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]# 找角点
        
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)# 筛选面积最大的色块
            size = int(cv2.contourArea(c))

            if (size > size_code): # 面积限幅
                print(f"检测到的大轮廓面积: {size}")
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                # 在绘制之前检查box的有效性
                if box is not None and len(box) > 0:
                    cv2.drawContours(output_img, [np.int_(box)], -1, (0, 0, 255), 2)
                
                center_x, center_y = rect[0]
                print(f"绘制轮廓并返回中心坐标: ({int(center_x)}, {int(center_y)})")
                return (int(center_x), int(center_y)) # 返回相应色块的中心坐标
            else:
                # 面积不满足时不打印信息，避免刷屏
                pass
        else:
            print("未检测到任何轮廓。")
            pass
    else:
        print("无画面")
    return None # 如果没有检测到物块，返回 None

def find_specific_target(frame, color_to_find, target_type):
    """
    根据指定的颜色和目标类型，寻找唯一的目标。
    :param frame: 摄像头画面
    :param color_to_find: 'red', 'green', 或 'blue'
    :param target_type: 'PLATFORM', 'SLOT', 或 'STACK_BASE'
    :return: 目标的中心像素坐标 (cx, cy) 或 None
    """
    # 步骤 1: 预处理 - 高斯模糊减少噪声
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    
    # 步骤 2: 颜色分割 (已修复对红色的特殊处理)
    mask = None
    if color_to_find == 'red':
        # 红色在HSV空间有两个范围，需要分别处理后合并
        lower1, upper1 = color_thresholds['red']['Lower1'], color_thresholds['red']['Upper1']
        lower2, upper2 = color_thresholds['red']['Lower2'], color_thresholds['red']['Upper2']
        mask1 = cv2.inRange(hsv_frame, lower1, upper1)
        mask2 = cv2.inRange(hsv_frame, lower2, upper2)
        mask = cv2.add(mask1, mask2) # 合并两个红色范围的掩码
    else:
        # 其他颜色直接获取阈值
        lower_hsv = color_thresholds[color_to_find]['Lower']
        upper_hsv = color_thresholds[color_to_find]['Upper']
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    # 步骤 3: 形态学处理 - 去除噪点
    kernel = np.ones((5, 5), np.uint8)
    # 使用开运算去除小的噪点
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 可以选择性地再加一个闭运算来填充物体内部的小洞
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 步骤 4: 根据“状态值”(target_type) 进行精确过滤
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 计算中心点
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 核心逻辑：根据状态值选择不同的判断标准
        if target_type == "PLATFORM":
            if PLATFORM_AREA_MIN < area < PLATFORM_AREA_MAX:
                return ((cx, cy), cnt) # 找到了！返回中心点和轮廓

        elif target_type == "SLOT":
            if SLOT_AREA_MIN < area < SLOT_AREA_MAX:
                return ((cx, cy), cnt) # 找到了！返回中心点和轮廓

        elif target_type == "STACK_BASE":
            # 寻找码垛基座（已放置的物块）的逻辑会更复杂一些
            # 简单版本：它的面积应该和物块面积接近
            if BLOCK_AREA_MIN < area < BLOCK_AREA_MAX:
                # 这里还可以增加判断，比如它是否在一个凹槽的轮廓内部
                return ((cx, cy), cnt) # 找到了！返回中心点和轮廓

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
                cv2.drawContours(frame, [platform_contour], -1, (0, 255, 255), 2) # 绘制轮廓
                cv2.putText(frame, f"{color.capitalize()} Platform", (platform_pos[0] + 10, platform_pos[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                print(f"检测到 {color} 凸台，中心坐标: {platform_pos}")
                platform_positions.append(platform_pos)
    except Exception as e:
        print(f"处理凸台检测时发生错误: {e}")
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
                cv2.drawContours(frame, [slot_contour], -1, (255, 255, 0), 2) # 绘制轮廓
                cv2.putText(frame, f"{color.capitalize()} Slot", (slot_pos[0] + 10, slot_pos[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                print(f"检测到 {color} 凹槽，中心坐标: {slot_pos}")
                slot_positions.append(slot_pos)
    except Exception as e:
        print(f"处理凹槽检测时发生错误: {e}")
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
                        
                        send[1] = 0x01
                        send[2] = (center_x & 0xff00) >> 8
                        send[3] = (center_x & 0xff)
                        send[4] = (center_y & 0xff00) >> 8
                        send[5] = (center_y & 0xff)
                        send[6] = 0x77
                        FH = bytearray(send)
                        write_len = ser.write(FH)
                        print(f"物料盘中心坐标: ({center_x}, {center_y})")
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
                            pos_x, pos_y = result
                            
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
                            
                            send[1] = 0x0A
                            send[2] = (platform_pos[0] & 0xff00) >> 8
                            send[3] = (platform_pos[0] & 0xff)
                            send[4] = (platform_pos[1] & 0xff00) >> 8
                            send[5] = (platform_pos[1] & 0xff)
                            send[6] = 0x77
                            FH = bytearray(send)
                            write_len = ser.write(FH)
                            print(f"{color_name}凸台坐标: {platform_pos}")
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
                            
                            send[1] = 0x0B
                            send[2] = (slot_pos[0] & 0xff00) >> 8
                            send[3] = (slot_pos[0] & 0xff)
                            send[4] = (slot_pos[1] & 0xff00) >> 8
                            send[5] = (slot_pos[1] & 0xff)
                            send[6] = 0x77
                            FH = bytearray(send)
                            write_len = ser.write(FH)
                            print(f"{color_name}凹槽坐标: {slot_pos}")
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
