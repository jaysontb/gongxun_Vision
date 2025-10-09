#!/bin/bash#!/bin/bash#!/bin/bash#!/bin/bash



# 视觉识别服务启动脚本

sleep 5

# ============================================================# ================================================================

LOG_DIR="/var/log/vision"

LOG_FILE="$LOG_DIR/vision.log"# 视觉识别服务启动脚本 (与sudo python3运行环境一致)



mkdir -p "$LOG_DIR" 2>/dev/null# ============================================================# ============================================================# 视觉识别自启动脚本



SERIAL_PORT="/dev/ttyS3"

if [ -e "$SERIAL_PORT" ]; then

    stty -F "$SERIAL_PORT" 115200 cs8 -cstopb -parenb -crtscts 2>/dev/null# 等待系统完全启动# 视觉识别服务启动脚本 (与sudo python3运行环境一致)# 功能: 自动配置串口、激活Python环境、运行Creama.py

    chmod 666 "$SERIAL_PORT" 2>/dev/null

fisleep 5



cd /home/cat || exit 1# ============================================================# ================================================================



echo "==========================================" >> "$LOG_FILE"# 日志目录

echo "视觉服务启动: $(date)" >> "$LOG_FILE"

echo "==========================================" >> "$LOG_FILE"LOG_DIR="/var/log/vision"



/usr/bin/python3 Creama.py 2>&1 | tee -a "$LOG_FILE"LOG_FILE="$LOG_DIR/vision.log"


ERROR_LOG="$LOG_DIR/vision_error.log"# 等待系统完全启动# 日志文件路径



# 确保日志目录存在sleep 5LOG_DIR="/var/log/vision"

mkdir -p "$LOG_DIR" 2>/dev/null || true

LOG_FILE="$LOG_DIR/vision.log"

# 配置串口设备

SERIAL_PORT="/dev/ttyS3"# 日志目录

if [ -e "$SERIAL_PORT" ]; then

    stty -F "$SERIAL_PORT" 115200 cs8 -cstopb -parenb -crtscts 2>/dev/null || trueLOG_DIR="/var/log/vision"# 创建日志目录

    chmod 666 "$SERIAL_PORT" 2>/dev/null || true

    echo "[INFO] 串口配置成功" >> "$LOG_FILE"LOG_FILE="$LOG_DIR/vision.log"mkdir -p "$LOG_DIR"

fi

ERROR_LOG="$LOG_DIR/vision_error.log"

# 切换到工作目录

cd /home/cat || exit 1# 日志函数



# 打印启动信息# 确保日志目录存在log() {

echo "========================================" >> "$LOG_FILE"

echo "[INFO] 视觉服务启动: $(date)" >> "$LOG_FILE"mkdir -p "$LOG_DIR"    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"

echo "[INFO] Python: /usr/bin/python3" >> "$LOG_FILE"

echo "[INFO] 用户: $(whoami)" >> "$LOG_FILE"chown cat:cat "$LOG_DIR"}

echo "========================================" >> "$LOG_FILE"



# 运行视觉程序(使用系统Python)

/usr/bin/python3 Creama.py 2>&1 | tee -a "$LOG_FILE"# 配置串口设备log "========== 视觉服务启动 =========="


SERIAL_PORT="/dev/ttyS3"

if [ -e "$SERIAL_PORT" ]; then# 1. 等待系统完全启动(重要!)

    # 配置波特率和串口参数log "等待系统启动完成..."

    stty -F "$SERIAL_PORT" 115200 cs8 -cstopb -parenb -crtsctssleep 5

    # 修改权限,允许用户访问

    chmod 666 "$SERIAL_PORT"# 2. 查找并配置串口

    echo "[INFO] 串口 $SERIAL_PORT 配置成功 (115200 8N1)" | tee -a "$LOG_FILE"log "配置串口参数..."

else

    echo "[ERROR] 串口 $SERIAL_PORT 不存在!" | tee -a "$ERROR_LOG"# LubanCat2板载串口ttyS3

    exit 1SERIAL_PORT="/dev/ttyS3"

fi

# 如果使用USB转串口,可以取消下面3行注释,并注释掉上面的SERIAL_PORT

# 切换到工作目录# SERIAL_PORT=$(ls /dev/ttyUSB* 2>/dev/null | head -n 1)

cd /home/cat || exit 1# [ -z "$SERIAL_PORT" ] && SERIAL_PORT=$(ls /dev/ttyACM* 2>/dev/null | head -n 1)

# [ -z "$SERIAL_PORT" ] && log "错误: 未找到USB串口设备!" && exit 1

# 打印启动信息

echo "============================================" | tee -a "$LOG_FILE"# 检查串口是否存在

echo "[INFO] 视觉识别服务启动于: $(date)" | tee -a "$LOG_FILE"if [ ! -e "$SERIAL_PORT" ]; then

echo "[INFO] Python路径: /usr/bin/python3" | tee -a "$LOG_FILE"    log "错误: 串口设备 $SERIAL_PORT 不存在!"

echo "[INFO] 工作目录: $(pwd)" | tee -a "$LOG_FILE"    log "可用串口: $(ls /dev/tty{USB,ACM}* 2>/dev/null || echo '无')"

echo "[INFO] 执行用户: $(whoami)" | tee -a "$LOG_FILE"    exit 1

echo "============================================" | tee -a "$LOG_FILE"fi



# 使用系统Python运行(与sudo python3一致)log "找到串口: $SERIAL_PORT"

/usr/bin/python3 Creama.py 2>&1 | tee -a "$LOG_FILE"

# 配置串口波特率115200, 8N1, 无流控

# 如果程序异常退出,记录错误stty -F "$SERIAL_PORT" 115200 cs8 -cstopb -parenb -crtscts

EXIT_CODE=$?if [ $? -eq 0 ]; then

if [ $EXIT_CODE -ne 0 ]; then    log "串口配置成功: 115200 8N1"

    echo "[ERROR] 视觉程序异常退出,退出码: $EXIT_CODE" | tee -a "$ERROR_LOG"else

fi    log "错误: 串口配置失败!"

    exit 1
fi

# 设置串口权限(允许普通用户访问)
chmod 666 "$SERIAL_PORT"
log "串口权限设置完成"

# 3. 切换到代码目录
CODE_DIR="/home/cat"  # Creama.py所在目录
cd "$CODE_DIR" || {
    log "错误: 无法进入目录 $CODE_DIR"
    exit 1
}
log "工作目录: $(pwd)"

# 4. 激活Python虚拟环境(如果使用)
# 如果你没有使用虚拟环境,注释掉下面4行
# if [ -d "venv" ]; then
#     source venv/bin/activate
#     log "Python虚拟环境已激活"
# fi

# 5. 启动视觉识别程序
log "启动 Creama.py..."
PYTHON_BIN="python3"  # 或者 python

# 方式1: 直接运行(推荐)
"$PYTHON_BIN" Creama.py 2>&1 | tee -a "$LOG_FILE"

# 方式2: 后台运行(如果需要)
# nohup "$PYTHON_BIN" Creama.py >> "$LOG_FILE" 2>&1 &
# VISION_PID=$!
# log "视觉程序已启动, PID: $VISION_PID"

# 如果程序异常退出
log "视觉程序已退出"
exit 0