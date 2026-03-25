# A_QUANT 监控台 — 部署手册

## 目录结构
```
/opt/
├── A_QUANT_PRO/               ← 主量化项目
└── a_quant_dashboard/         ← 本监控台（独立，不依赖主项目代码）
    ├── app.py
    ├── requirements.txt
    ├── .env                   ← 本地创建，勿提交 git
    └── static/
        └── index.html
```

---

## 部署步骤

### 1. 上传代码到服务器
```bash
# 从本地上传
scp -r a_quant_dashboard/ <user>@<server-ip>:/opt/

# 或 rsync（推荐，支持增量更新）
rsync -avz --exclude='.env' a_quant_dashboard/ <user>@<server-ip>:/opt/a_quant_dashboard/
```

### 2. 安装依赖
```bash
ssh <user>@<server-ip>
cd /opt/a_quant_dashboard
python3.8 -m venv venv          # 指定 python3.8
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置环境变量
```bash
cp .env.example .env
nano .env
```
按照 `.env.example` 填入真实值：
```
DB_HOST=<阿里云 RDS 地址>
DB_PORT=3306
DB_USER=<数据库用户名>
DB_PASS=<数据库密码>
DB_NAME=a_quant
HOST=0.0.0.0
PORT=8000
```

### 4. 测试运行
```bash
source venv/bin/activate
python app.py
# 访问 http://<server-ip>:8000 验证
```

---

## 生产部署（systemd）

```bash
cat > /etc/systemd/system/a_quant_dashboard.service << 'EOF'
[Unit]
Description=A QUANT Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/a_quant_dashboard
ExecStart=/opt/a_quant_dashboard/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
EnvironmentFile=/opt/a_quant_dashboard/.env

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable a_quant_dashboard
systemctl start a_quant_dashboard
systemctl status a_quant_dashboard
```

---

## Nginx 反向代理（80 端口）

```bash
cat > /etc/nginx/sites-available/a_quant << 'EOF'
server {
    listen 80;
    server_name <server-ip-or-domain>;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
EOF

ln -s /etc/nginx/sites-available/a_quant /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

---

## 运维

```bash
systemctl status a_quant_dashboard    # 查看状态
journalctl -u a_quant_dashboard -f    # 实时日志
systemctl restart a_quant_dashboard   # 重启
```

## 更新代码

```bash
rsync -avz --exclude='.env' a_quant_dashboard/ <user>@<server-ip>:/opt/a_quant_dashboard/
ssh <user>@<server-ip> 'systemctl restart a_quant_dashboard'
```

---

## 注意事项

1. **RDS 白名单**：确认服务器 IP 已加入阿里云 RDS 安全组白名单
2. **防火墙**：服务器 8000 端口（或 80）需开放入站规则（阿里云 ECS 在安全组配置，本地 iptables 无需改动）
3. **.env 安全**：已加入 `.gitignore`，不会被提交
4. **无鉴权**：如需限制访问可在 Nginx 层加 basic auth
