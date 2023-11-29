# consumers.py
from channels.generic.websocket import AsyncWebsocketConsumer
import json

class TrainingProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "training_progress"
        # 将连接加入组
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def send_training_progress(self, event):
        progress = event['progress']
        algoID = event['algoID']
        # 发送训练进度到连接的 WebSocket 客户端
        await self.send(text_data=json.dumps({
            'progress': progress,
            'algoID': algoID
        }))
