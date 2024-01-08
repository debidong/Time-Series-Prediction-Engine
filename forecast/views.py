from django.utils import timezone
import os
import pandas as pd
from rest_framework.views import APIView, Response, status
from .lib.regression import AR, ARIMA_model, Fbprophet
from .lib.nn import infer
from django.core.paginator import Paginator

from file.storage import is_allowed_file, is_duplicate_name, FILE_FOLDER, FORECAST_FOLDER
from .models import File
from analysis.views import analyze_csv

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
            delete_file(pk)
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
        _, figure = ARIMA_model(pk, target, order, window, step)
        res = {
            "status": 200,
            "content": "提交成功",
            "content": {
                    "figure": './api/' + figure
            },
        }
        delete_file(pk)
        return Response(res, status=status.HTTP_200_OK)
    
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
        _, figure = Fbprophet(pk, target, window, step, periods, freq)
        res = {
            "status": 200,
            "message": "提交成功",
            "content": {
                "figure": './api/' + figure
            }
        }
        delete_file(pk)
        return Response(res, status=status.HTTP_200_OK)
    
class InferView(APIView):
    def post(self, request):
        '''提交预测，使用训练完毕的神经网络模型
        '''
        pk_model = request.data.get("pk_model")
        pk_file = request.data.get("pk_file")
        _, figure = infer(pk_model, pk_file)
        res = {
            "status": 200,
            "message": "提交成功",
            "content": {
                "figure": './api/' + figure
            }
        }
        return Response(res, status=status.HTTP_200_OK)
    
class TableView(APIView):
    def post(self, request):
        '''查询数据库中全部的文件
        '''
        pages = Paginator(File.objects.order_by('created'), int(request.data.get("size")))
        files = pages.get_page(request.data.get("currentPage")).object_list
        

        res = {
            "status": 200,
            "message": "查询成功",
            "content": {
                "columns": [
                    {
                        "name": "时空数据",
                        "value": "dataName"
                    },
                    {
                        "name": "数据集描述",
                        "value": "dataDescription",
                        "tooltip": True
                    },
                    {
                        "name": "行数",
                        "value": "rowCount"
                    },
                    {
                        "name": "字段数",
                        "value": "fieldCount"
                    },
                    {
                        "name": "创建时间",
                        "value": "creationTime"
                    },
                    {
                        "name": "操作",
                        "value": "operation",
                        # "slot": True
                    }
                    ],
                "data": [],
                "componentTitle": "",
                "pageTotal": pages.count
            }
        }
        for file in files:
            res["content"]["data"].append({
                "id": file.pk,
                "dataName": file.name,
                "dataDescription": file.description,
                "rowCount": file.row,
                "fieldCount": file.column,
                "creationTime": file.created.strftime('%Y-%m-%d %H:%M:%S')
            })
        return Response(res, status=status.HTTP_200_OK)

class GetFileView(APIView):
    def post(self, request):
        """解析单个文件
        """
        
        file = File.objects.get(pk=request.data.get("id")) or File.objects.get(name=request.data.get("name"))
        
        if file is not None:
            result = analyze_csv(file.path)
            res = {"message": "查询成功", "status": 200, "content": {
                "columns": [
                    {
                        "name": "序号",
                        "value": "id"
                    },
                    {
                        "name": "字段名称",
                        "value": "columnName"
                    },
                    {
                        "name": "数据类型",
                        "value": "dataType"
                    },
                    {
                        "name": "最小值",
                        "value": "min"
                    },
                    {
                        "name": "最大值",
                        "value": "max"
                    },
                    {
                        "name": "示例值",
                        "value": "exampleData"
                    },
                ],
                "data": [],
                }
            }
            for item in result:
                res['content']['data'].append(item)
            return Response(res, status=status.HTTP_200_OK)
        else:
            res = {
                "status": 500,
                "message": "文件不存在",
                "content": False
            }
            return Response(res, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class FileView(APIView):
    def post(self, request):
        """文件上传
        """
        file = request.FILES['file']
        name = file.name.split(".")[0] + '-' +timezone.now().strftime('%Y%m%d%H%M%S') + '.csv'
        if not (file and is_allowed_file(file.name)):
            return Response(status=status.HTTP_400_BAD_REQUEST)
        
        if is_duplicate_name(name, directory_path=FORECAST_FOLDER):
            res = {
                "status": 500,
                "message": "存在重复文件",
                "content": "false"
            }
            return Response(res, status=status.HTTP_200_OK)

        path = FORECAST_FOLDER+"/"+name
        with open(path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        row, column = pd.read_csv(path).shape
        path = FORECAST_FOLDER+"/"+name
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

    def delete(self, request):
        """文件删除
        """
        id = request.data.get("id")
        if id is not None:
            file = File.objects.get(pk=id)

            print(file)
            if file is not None:
                os.remove(file.path)
                file.delete()
                res = {
                    "status": 200,
                    "message": "删除成功"
                }
                return Response(res, status=status.HTTP_200_OK)
            else:
                res = {
                    "status": 500,
                    "message": "文件不存在",
                }
                return Response(res, status=status.HTTP_400_BAD_REQUEST)
        else:
            # 在数据集记录未插入数据库之前，使用文件名删除暂存区的文件
            name = request.data.get("name")
            try:
                # 这一步有可能造成目录遍历删除任意文件，但是业务非要这么做没办法
                path = './upload/'+name
                os.remove(path)
                res = {
                    "status": 200,
                    "message": "删除成功"
                }
                return Response(res, status=status.HTTP_200_OK)
            except:
                res = {
                    "status": 500,
                    "message": "文件不存在"
                }
                return Response(res, status=status.HTTP_400_BAD_REQUEST)
            
class InsertView(APIView):
    """将数据集记录插入至数据库中
    """
    def post(self, request):
        name = request.data.get("name")
        path = "./upload/"+name
        dumped_file = pd.read_csv(path)      
        row, column = dumped_file.shape

        File.objects.create(
            path=path,
            description=request.data.get("description"),
            name=name,
            row=row,
            column=column,
            created=timezone.now()
        ).save()

        res = {
            "status": 200,
            "message": "插入成功",
            "content": True
        }
        return Response(res, status=status.HTTP_200_OK)
    
def delete_file(pk: int):
    file = File.objects.get(pk=pk)
    os.remove(file.path)
    file.delete()