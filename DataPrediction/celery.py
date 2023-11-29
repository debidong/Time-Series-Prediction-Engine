# celery.py

import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DataPrediction.settings')
app = Celery('DataPrediction')

# 使用 Django 配置
app.config_from_object('django.conf:settings', namespace='CELERY')
app.conf.update(
    # 设置为 False，以启用异步执行
    CELERY_TASK_ALWAYS_EAGER=False,
)

# 从所有注册的应用中加载任务模块
app.autodiscover_tasks()
