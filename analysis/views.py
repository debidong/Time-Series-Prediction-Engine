import os
from django.http import FileResponse
import pandas as pd
import numpy as np
from rest_framework.views import APIView, Response, status
from django.core.paginator import Paginator
import re

from .models import Algorithm, Result
from file.models import File
from .lib.run import train
from utils.db import redis_conn

class GetAlgorithmView(APIView):
    """查询全部模型及算法
    """
    def post(self, request):
        pages = Paginator(Algorithm.objects.order_by("name"), int(request.data.get("size")))
        algos = pages.get_page(request.data.get("currentPage")).object_list
        res = {
            "status": 200,
            "message": "查询成功",
            "content": {
                "columns": [
                    {
                        "name": "模型名称",
                        "value": "modelName"
                    },
                    {
                        "name": "模型描述",
                        "value": "modelDescription",
                        "tooltip": True
                    },
                    {
                        "name": "数据集",
                        "value": "dataset"
                    },
                    {
                        "name": "算法名称",
                        "value": "neuralNetwork"
                    },
                    {
                        "name": "预测目标",
                        "value": "target"
                    },
                    {
                        "name": "训练状态",
                        "value": "status",
                    },
                    {
                        "name": "训练进度",
                        "value": "progress",
                    }
                    ],
                "data": [],
                "componentTitle": "",
                "pageTotal": pages.count
            }
        }

        for algo in algos:
            a = {
                "id": algo.pk,
                "modelName": algo.name,
                "modelDescription": algo.description,
                "dataset": algo.dataset.name,
                "neuralNetwork": algo.neuralNetwork,
                "target": algo.target,
                "status": algo.status
            }
            if redis_conn.get(algo.pk) is not None:
                progress = redis_conn.get(algo.pk).decode()
                a['progress'] = progress
            else:
                a['progress'] = 0
            
            res["content"]["data"].append(a)
        return Response(res, status=status.HTTP_200_OK)

class AlgorithmView(APIView):
    def post(self, request):
        '''新增模型和算法
        '''
        # 未对算法/target等进行进一步验证，日后可进行完善
        dataset = File.objects.get(pk=request.data.get("dataset"))
        if dataset is not None:
            algo = Algorithm(
                description = request.data.get("description"),
                name = request.data.get("name"),
                neuralNetwork = request.data.get("neuralNetwork"),
                layers = request.data.get("layers"),
                learningRate = request.data.get("learningRate"),
                neurons = request.data.get("neurons"),
                epoch = request.data.get("rounds"),
                batchSize = request.data.get("batchSize"),
                optimization = request.data.get("optimization"),
                selected = request.data.get("selected"),
                target = request.data.get("target"),
                verificationRate = request.data.get("verificationRate"),
                dataset = dataset,
            )
            algo.save()
            res = {
                "status": 200,
                "content": "上传成功"
            }
            return Response(res, status=status.HTTP_200_OK)
        else:
            res = {
                "status": 500,
                "content": "数据集不存在"
            }
            return Response(res, status=status.HTTP_400_BAD_REQUEST)
        
    def delete(self, request):
        """删除模型及算法
        """
        algo = Algorithm.objects.get(pk=request.data.get("id"))
        result = Result.objects.get(algo=algo)
        if algo is not None and result is not None:
            print(algo)
            # 删除神经网络模型
            # img_path = [result.difference, result.loss]
            # for p in img_path:
            #     os.remove(p[6:])
            model_name = algo.name + '.pth'
            print("name", algo.name)
            # os.remove(os.path.join('torch_models', model_name))
            os.remove('./torch_models/'+model_name)
            # 删除数据库内算法记录
            algo.delete()
            res = {
                "status": 200,
                "content": "删除成功"
            }
            return Response(res, status=status.HTTP_200_OK)
        else:
            res = {
                "status": 500,
                "content": "数据集不存在"
            }
            return Response(res, status=status.HTTP_400_BAD_REQUEST)



def analyze_csv(file_path):
    '''分析CSV文件
    '''
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    global current_file_row, current_file_column
    current_file_row, current_file_column = df.shape
    # 存放所有列的分析结果
    all_columns_analysis = []
    # 从数据框中随机选择一个实例值
    random_row = df.sample(1).iloc[0]
    count_i = 1
    for column in df.columns:
        # 列的分析结果
        column_analysis = {"columnName": column}
        column_analysis["id"] = str(count_i)
        # 数据类型
        dtype = df[column].dtype
        if dtype == "object":
            column_analysis["dataType"] = "string"
            column_analysis["min"] = "-"
            column_analysis["max"] = "-"
        else:
            column_analysis["dataType"] = str(dtype)
            column_analysis["min"] = str(np.float64(df[column].min()))
            column_analysis["max"] = str(np.float64(df[column].max()))
        # 为每一列添加一个实例值
        column_analysis["exampleData"] = str(random_row[column])
        count_i += 1
        # 将列的分析结果添加到总列表中
        all_columns_analysis.append(column_analysis)
    return np.array(all_columns_analysis)

class TrainingView(APIView):
    def post(self, request):
        algo: Algorithm = Algorithm.objects.get(pk=request.data.get("id"))
        if algo is not None:
            # 搭配celery在后台进行训练
            train.delay(algo.pk)
            res = {
                "status": 200,
                "message": "提交成功",
            }
            return Response(res, status=status.HTTP_200_OK)
        else: 
            res = {
                "status": 500,
                "content": "算法不存在"
            }
            return Response(res, status=status.HTTP_400_BAD_REQUEST)



class ResultView(APIView):
    def get(self, request, image):
        print("--", os.getcwd())
        image_path = os.path.join('result', image)

        print(image_path)
        if os.path.exists(image_path):
            response = FileResponse(open(image_path, 'rb'), content_type='image/png')
            response['Content-Disposition'] = f'inline; filename={image}'
            return response
        else:
            return Response(status=status.HTTP_404_NOT_FOUND)
    def post(self, request):
        algo: Algorithm = Algorithm.objects.get(pk=request.data.get("id"))
        if algo is not None:
            result = Result.objects.get(algo=algo)
            if result is not None:
                res = {
                    "status": 200,
                    "message": "查询成功",
                    "content": {
                        "mse": str(result.mse),
                        "rmse": str(result.rmse),
                        "mae": str(result.mae),
                        "difference": result.difference,
                        "loss": result.loss,
                    },
                }
                return Response(res, status=status.HTTP_200_OK)
            else:
                res = {
                    "status": 500,
                    "message": "训练未完成"
                }
                return Response(res, status=status.HTTP_400_BAD_REQUEST)
        else:
            res = {
                "status": 500,
                "message": "算法不存在"
            }
            return Response(res, status=status.HTTP_400_BAD_REQUEST)