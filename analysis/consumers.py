import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer

class TrainingProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "training_progress"
        # 将连接加入组
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

        # 启动心跳包定时器
        # 废弃，改为前端向后端发送心跳
        # self.heartbeat_task = asyncio.ensure_future(self.send_heartbeat())

    async def disconnect(self, close_code):
        # 关闭连接时取消定时器
        # self.heartbeat_task.cancel()
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def send_training_progress(self, event):
        progress = event['progress']
        algoID = event['algoID']
        # 发送训练进度到连接的 WebSocket 客户端
        await self.send(text_data=json.dumps({
            'progress': progress,
            'algoID': algoID
        }))

    # async def send_heartbeat(self):
    #     while True:
    #         # 发送心跳包到 WebSocket 客户端
    #         await self.send(text_data=json.dumps({'heartbeat': 'ping'}))
    #         await asyncio.sleep(10)