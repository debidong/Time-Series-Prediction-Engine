from django.utils import timezone
import os
import pandas as pd
from rest_framework.views import APIView, Response, status
from .lib.regression import AR, ARIMA_model, Fbprophet
from .lib.nn import infer

from utils.storage import is_allowed_file, is_duplicate_name, TEMP_FOLDER
from .models import File

class ARView(APIView):
    def post(self, request):
        '''提交预测，使用AR模型
        '''
        try:
            pk = request.data.get('pk')
            order = request.data.get('order')
            target = request.data.get('target')
            window = request.data.get('window')
            step = request.data.get('step')
            _, figure = AR(pk, target, order, window, step)
            res = {
                "status": 200,
                "message": "提交成功",
                "content": {
                    "figure": './api/' + figure
                },
            }

            # 删除暂存文件
            # TODO:
            # 前端上传文件之后反悔可能会引起文件残留，需要引入特定的信号来清除残留文件，或者定期手动清理
            # delete_file(pk)
            return Response(res, status=status.HTTP_200_OK)
        except:
            res = {
                "status": 500,
                "content": "参数有误"
            }
            return Response(res, status=status.HTTP_400_BAD_REQUEST)

class ARIMAView(APIView):
    def post(self, request):
        '''提交预测，使用ARIMA模型
        '''
        pk = request.data.get('pk')
        order = request.data.get('order')
        target = request.data.get('target')
        window = request.data.get('window')
        step = request.data.get('step')
        try:
            _, figure = ARIMA_model(pk, target, order, window, step)
            res = {
                "status": 200,
                "content": "提交成功",
                "content": {
                        "figure": './api/' + figure
                },
            }
            # delete_file(pk)
            return Response(res, status=status.HTTP_200_OK)
        except:
            res = {
                "status": 500,
                "content": "参数有误"
            }
            return Response(res, status=status.HTTP_400_BAD_REQUEST)

class FbprophetView(APIView):
    def post(self, request):
        '''提交预测，使用fbprophet模型
        '''
        pk = request.data.get('pk')
        target = request.data.get('target')
        window = request.data.get('window')
        step = request.data.get('step')
        periods = request.data.get('periods')
        freq = request.data.get('freq')
        try:
            _, figure = Fbprophet(pk, target, window, step, periods, freq)
            res = {
                "status": 200,
                "message": "提交成功",
                "content": {
                    "figure": './api/' + figure
                }
            }
            # delete_file(pk)
            return Response(res, status=status.HTTP_200_OK)
        except:
            res = {
                "status": 500,
                "message": "参数有误",
                "content": None
            }
            return Response(res, status=status.HTTP_400_BAD_REQUEST)
        
class InferView(APIView):
    def post(self, request):
        '''提交预测，使用训练完毕的神经网络模型
        '''
        pk_model = request.data.get("pk_model")
        pk_file = request.data.get("pk_file")
        _, figure = infer(pk_model, pk_file)
        try:
            res = {
                "status": 200,
                "message": "提交成功",
                "content": {
                    "figure": './api/' + figure
                }
            }
            return Response(res, status=status.HTTP_200_OK)
        except:
            res = {
                "status": 500,
                "message": "参数有误",
                "content": None
            }
            return Response(res, status=status.HTTP_400_BAD_REQUEST)

class FileView(APIView):
    def post(self, request):
        """文件上传
        """
        file = request.FILES['file']
        name = file.name.split(".")[0] + '-' +timezone.now().strftime('%Y%m%d%H%M%S') + '.csv'
        if not (file and is_allowed_file(file.name)):
            return Response(status=status.HTTP_400_BAD_REQUEST)
        
        if is_duplicate_name(name, directory_path=TEMP_FOLDER):
            res = {
                "status": 500,
                "message": "存在重复文件",
                "content": "false"
            }
            return Response(res, status=status.HTTP_200_OK)

        path = TEMP_FOLDER+"/"+name
        with open(path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        row, column = pd.read_csv(path).shape
        path = TEMP_FOLDER+"/"+name
        dumped_file = pd.read_csv(path)      
        row, column = dumped_file.shape

        f = File.objects.create(
            path=path,
            name=name,
            row=row,
            column=column,
            created=timezone.now()
        )
        
        res = {
            "status": 200,
            "message": "上传成功",
            "content": f.pk
        }
        return Response(res, status=status.HTTP_200_OK)
    
    def put(self, request):
        """文件修改
        """
        pass
    
def delete_file(pk: int):
    '''删除上传的文件
    '''
    try:
        file = File.objects.get(pk=pk)
        os.remove(file.path)
        file.delete()
    except:
        pass