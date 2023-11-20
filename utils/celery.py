# celery.py

import os
from celery import Celery

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')

# 创建 Celery 实例
app = Celery('your_project')

# 使用 Django 配置
app.config_from_object('django.conf:settings', namespace='CELERY')

# 从所有注册的应用中加载任务模块
app.autodiscover_tasks()
