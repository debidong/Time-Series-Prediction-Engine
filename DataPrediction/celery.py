# celery.py

import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DataPrediction.settings')
app = Celery('DataPrediction')

# 使用 Django 配置
app.config_from_object('django.conf:settings', namespace='CELERY')

# 从所有注册的应用中加载任务模块
app.autodiscover_tasks()
