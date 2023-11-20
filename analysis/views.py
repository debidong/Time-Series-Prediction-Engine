import pandas as pd
import numpy as np
from rest_framework.views import APIView, Response, status
from django.core.paginator import Paginator

from .models import Algorithm
from file.models import File
from .lib.run import train

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
                        "name": "序号",
                        "value": "id"
                    },
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
                        "name": "操作",
                        "value": "operation",
                        "slot": True
                    }
                    ],
                "data": [],
                "componentTitle": "",
                "pageTotal": pages.count
            }
        }

        for algo in algos:
            res["content"]["data"].append({
                "id": algo.pk,
                "modelName": algo.name,
                "modelDescription": algo.description,
                "dataset": algo.dataset.name,
                "neuralNetwork": algo.neuralNetwork,
                "target": algo.target,
                "status": algo.status
            })
        return Response(res, status=status.HTTP_200_OK)

class AlgorithmView(APIView):
    def post(self, request):
        '''新增模型和算法
        '''
        # 未对算法/target等进行进一步验证，日后可进行完善
        dataset = File.objects.get(pk=request.data.get(dataset))
        if dataset is not None:
            algo = Algorithm(
                description = request.data.get("description"),
                name = request.data.get("name"),
                neuralNetwork = request.data.get("neuralNetwork"),
                layers = request.data.get("layers"),
                learningRate = request.data.get("learningRate"),
                neurons = request.data.get("neurons"),
                rounds = request.data.get("rounds"),
                batchSize = request.data.get("batchSize"),
                optimization = request.data.get("optimization"),
                target = request.data.get("target"),
                verificationRate = request.data("verificationRate"),
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
        algo = Algorithm(pk=id)
        if algo is not None:
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
        algo: Algorithm = File.objects.get(pk=request.data.get("algo"))
        if algo is not None:
            # 后台进行训练
            train.delay(algo.dataset, algo)
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
    def post(self, request):
        pass