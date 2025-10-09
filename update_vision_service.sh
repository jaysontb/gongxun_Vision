#!/bin/bash

# ============================================================
# 视觉服务更新部署脚本
# 用途: 将修改后的配置文件部署到LubanCat2系统
# ============================================================

echo "=========================================="
echo "  视觉服务配置更新脚本"
echo "=========================================="

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then 
    echo "[ERROR] 请使用 sudo 运行此脚本"
    echo "用法: sudo bash update_vision_service.sh"
    exit 1
fi

# 1. 复制服务配置文件
echo "[1/5] 复制 vision.service 到 /etc/systemd/system/ ..."
cp vision.service /etc/systemd/system/vision.service
chmod 644 /etc/systemd/system/vision.service

# 2. 复制启动脚本
echo "[2/5] 复制 vision_service.sh 到 /home/cat/ ..."
cp vision_service.sh /home/cat/vision_service.sh
chmod +x /home/cat/vision_service.sh
chown cat:cat /home/cat/vision_service.sh

# 3. 确保日志目录存在
echo "[3/5] 创建日志目录 /var/log/vision/ ..."
mkdir -p /var/log/vision
chown cat:cat /var/log/vision
chmod 755 /var/log/vision

# 4. 重新加载systemd配置
echo "[4/5] 重新加载 systemd 配置..."
systemctl daemon-reload

# 5. 重启服务
echo "[5/5] 重启 vision.service ..."
systemctl restart vision.service

# 等待服务启动
sleep 3

# 显示服务状态
echo ""
echo "=========================================="
echo "  服务状态"
echo "=========================================="
systemctl status vision.service --no-pager -l

echo ""
echo "=========================================="
echo "  部署完成!"
echo "=========================================="
echo ""
echo "常用命令:"
echo "  查看状态:   sudo systemctl status vision.service"
echo "  查看日志:   sudo tail -f /var/log/vision/vision.log"
echo "  重启服务:   sudo systemctl restart vision.service"
echo "  停止服务:   sudo systemctl stop vision.service"
echo ""
