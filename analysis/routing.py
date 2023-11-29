from django.urls import re_path
  
from . import consumers
  
websocket_urlpatterns = [
    re_path(r'api/analysis/progress', consumers.TrainingProgressConsumer.as_asgi()),
]