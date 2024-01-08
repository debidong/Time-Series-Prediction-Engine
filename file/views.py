from rest_framework.views import APIView, Response, status
from django.core.paginator import Paginator
from django.utils import timezone
import os
import pandas as pd
import datetime

from .storage import is_allowed_file, is_duplicate_name, FILE_FOLDER
from .models import File
from analysis.views import analyze_csv


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
                        "name": "数据集名称",
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
                        "slot": True
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
        if not (file and is_allowed_file(file.name)):
            return Response(status=status.HTTP_400_BAD_REQUEST)
        
        if is_duplicate_name(file.name):
            res = {
                "status": 500,
                "message": "存在重复文件",
                "content": False
            }
            return Response(res, status=status.HTTP_200_OK)
        path = FILE_FOLDER+"/"+file.name
        with open(path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        row, column = pd.read_csv(path).shape
        res = {
            "status": 200,
            "message": "上传成功",
            "content": True
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
