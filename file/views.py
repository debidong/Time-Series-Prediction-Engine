from rest_framework.views import APIView, Response, status
from django.core.paginator import Paginator
from django.utils import timezone
import os
import pandas as pd
import datetime

from .storage import is_allowed_file, is_duplicate_name, UPLOAD_FOLDER
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
        #     dumped_file = pd.read_csv(path)
        #     res = {
        #         "status": 200,
        #         "message": "查询成功",
        #         "content": {
        #             "columns": [],
        #             "data": [],
        #         }
        #     }
        #     columns = dumped_file.columns.to_list()
        #     for column in columns:
        #         res["content"]["columns"].append({
        #             "name": column,
        #             "value": column
        #         })
        #     for _, row in dumped_file.iterrows():
        #         added = {}
        #         for col in columns:
        #             added[col] = row[col]
        #         res["content"]["data"].append(added)
        #     print(res)
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
        path = UPLOAD_FOLDER+"/"+file.name
        with open(path, "wb+") as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        dumped_file = pd.read_csv(path)      
        row, column = dumped_file.shape

        File.objects.create(
            path=path,
            description="test",
            name=file.name,
            row=row,
            column=column,
            created=timezone.now()
        ).save()

        res = {
            "status": 200,
            "message": "查询成功",
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
        file = File.objects.get(pk=request.data.get("id"))
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