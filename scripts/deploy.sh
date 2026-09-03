#!/bin/bash
# 制造业Agent 完整部署脚本 - 服务器上直接运行
set -e
cd /root

echo "=== 1. 停止旧服务 ==="
systemctl stop mfg-agent 2>/dev/null || true
systemctl stop nginx 2>/dev/null || true
pkill -9 -f uvicorn 2>/dev/null || true
sleep 2

echo "=== 2. 清理重建数据库 ==="
rm -f data/manufacturing.db

echo "=== 3. 创建 systemd 服务 ==="
cat > /etc/systemd/system/mfg-agent.service << 'SVCEND'
[Unit]
Description=Manufacturing Agent Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/usr/bin/python3 -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=10
StandardOutput=append:/var/log/mfg-agent.log
StandardError=append:/var/log/mfg-agent.log

[Install]
WantedBy=multi-user.target
SVCEND

echo "=== 4. 重建 nginx 配置 ==="
cat > /etc/nginx/conf.d/mfg-agent.conf << 'NGXEND'
server {
    listen 80;
    server_name _;
    client_max_body_size 50m;

    location / {
        root /root/frontend/chat/dist;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    location /admin/ {
        alias /root/frontend/admin/dist/;
        index index.html;
        try_files $uri $uri/ /admin/index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_buffering off;
    }

    location /uploads/ { proxy_pass http://127.0.0.1:8000; }
    location /metrics  { proxy_pass http://127.0.0.1:8000; }
}
NGXEND

echo "=== 5. 启动服务 ==="
systemctl daemon-reload
systemctl enable mfg-agent nginx
systemctl restart nginx
systemctl restart mfg-agent
sleep 5

echo "=== 6. 等待启动 (种子加载约2分钟) ==="
for i in $(seq 1 36); do
    if curl -s http://127.0.0.1:8000/api/v1/health > /dev/null 2>&1; then
        echo ""
        echo "=== 启动成功 ==="
        curl -s http://127.0.0.1:8000/api/v1/health
        echo ""
        echo "访问地址:"
        echo "  智能助手: http://8.163.102.79"
        echo "  管理后台: http://8.163.102.79/admin"
        echo "  API文档:  http://8.163.102.79/docs"
        exit 0
    fi
    sleep 10
    echo -n "."
done

echo ""
echo "启动超时, 查看日志:"
echo "  journalctl -u mfg-agent -n 20"
echo "  tail -30 /var/log/mfg-agent.log"
systemctl status mfg-agent --no-pager | head -6
