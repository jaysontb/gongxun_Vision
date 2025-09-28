import cv2
import matplotlib as plt
import numpy as np
import serial as ser
import serial
import threading
import serial.tools.list_ports
from pyzbar.pyzbar import decode
def color_tutai_position(img):
    """
    凸台识别辅助实现定位
    :param img: 输入的图像
    :return: 三个凸台的中心坐标 x1,y1,x2,y2,x3,y3
    """
    # 定义颜色识别阈值（HSV）
    color_dist = {'red': {'Lower1': np.array([156, 60, 60]), 'Upper1': np.array([180, 255, 255]),'Lower2': np.array([0, 60, 60]), 'Upper2': np.array([6, 255, 255])},
              'blue': {'Lower': np.array([100, 140, 110]), 'Upper': np.array([124, 255, 255])},
              'green': {'Lower': np.array([38, 100, 60]), 'Upper': np.array([90, 255, 255])},
              }
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#转化成HSV图像
    
    erode_hsv = cv2.erode(img, None, iterations=2)# 腐蚀 粗的变细（平整边缘）
    kernel = np.ones((7,7),np.uint8)
    dilate_hsv = cv2.dilate(erode_hsv, kernel, iterations=2)# 膨胀 细的变粗（连接断开区域）
    gray_img=cv2.cvtColor(dilate_hsv,cv2.COLOR_BGR2GRAY)#转化为灰度图

    # 限制对比度自适应直方图均衡，非常好的增强对光线鲁棒性的方法，但是阈值过大容易出现噪点
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))  # 5.0
    clahed = clahe.apply(gray_img) #对灰度图做 CLAHE 均衡化处理

    #计算形态学梯度（增强物体边缘）
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    gradient=cv2.morphologyEx(clahed,cv2.MORPH_GRADIENT,kernel)#形态学梯度

    result = cv2.GaussianBlur(gradient, (7, 7), 3, 3)#高斯模糊 平滑边缘

    equal_img = cv2.convertScaleAbs(result, alpha=4, beta=0)#线性变换增强对比度
    cv2.imshow("video2",equal_img)
    equal_img=cv2.GaussianBlur(equal_img, (9, 9), 3, 3)#再一次高斯模糊，平滑边缘

    _,thresh = cv2.threshold(equal_img, 70, 255, cv2.THRESH_BINARY)#二值化

    thresh=cv2.GaussianBlur(thresh,(9,9),3,3)#再次高斯模糊，平滑边缘
    
    #霍夫圆检测
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT_ALT, 1.5, 50, param1=100, param2=0.95, minRadius=15,maxRadius=50)
    try:
        if (len(circles[0]) == 3):
            circles = np.uint16(np.around(circles))
            # 遍历
            for circle in circles[0, :]:
                cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
                cv2.circle(img, (circle[0], circle[1]), 2, (255, 0, 0), 2)
            circle_all = [circles[0][0], circles[0][1], circles[0][2]]
            circle_list = sorted(circle_all, key=lambda x: x[0])
            return (circle_list[0][0], circle_list[0][1], circle_list[1][0], circle_list[1][1], circle_list[2][0],
                    circle_list[2][1])  # 依次返回色环的中心坐标
    except:
        pass


