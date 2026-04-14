"""
blog_server.py — 博客静态文件服务器（端口 8502）
用法：python blog_server.py
"""

import http.server
import os
from pathlib import Path

PORT     = 8502
BLOG_DIR = Path(__file__).parent / 'blog'

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BLOG_DIR), **kwargs)

    def log_message(self, format, *args):
        pass  # 静默日志，减少噪音

if __name__ == '__main__':
    os.chdir(BLOG_DIR)
    with http.server.HTTPServer(('', PORT), Handler) as httpd:
        print(f'✅ 博客服务已启动：http://localhost:{PORT}')
        print(f'   目录：{BLOG_DIR}')
        print('   按 Ctrl+C 停止')
        httpd.serve_forever()
