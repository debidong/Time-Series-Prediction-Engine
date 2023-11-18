import pandas as pd
import numpy as np
from rest_framework.views import APIView, Response, status

class GetAlgorithmView(APIView):
    def post(request):
        pass

class AlgorithmView(APIView):
    def post(request):
        pass
    def delete(request, pk):
        pass


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