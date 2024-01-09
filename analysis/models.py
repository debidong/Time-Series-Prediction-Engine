from django.db import models
from file.models import File

# Page1：新增算法
# 模型名称
# 模型描述

# 网络选择：LSTM/ MLP
# 优化算法：Adam/ RMSProp/ SGD
# 网络层数
# 神经元数目
# 学习率
# batch size

# Page2： 数据选择
# 验证保留集
# 数据集：choice：全部文件
# 目标：choice：字段名称

STATUS = [
    ("INI", "initialized"),
    ("ING", "training"),
    ("FIN", "finished"),
    ("ERR", "error")
]

class Algorithm(models.Model):
    '''算法模型
    '''
    description = models.CharField(max_length=128)
    name = models.CharField(max_length=128, unique=True)
    
    neuralNetwork = models.CharField(max_length=16)
    layers = models.IntegerField()
    learningRate = models.FloatField()
    neurons = models.CharField(max_length=256)
    epoch = models.IntegerField()
    batchSize = models.IntegerField()

    optimization = models.CharField(max_length=16)

    verificationRate = models.FloatField()
    dataset = models.ForeignKey(to=File, related_name="dataset" ,on_delete=models.CASCADE)
    selected = models.JSONField()
    target = models.CharField(max_length=128)

    window = models.IntegerField()
    step = models.IntegerField()

    status = models.CharField(max_length=11, choices=STATUS, default="INI")

class Result(models.Model):
    algo = models.ForeignKey(to=Algorithm, on_delete=models.CASCADE)
    # 展示前100个预测数据与真实数据的差异的图片路径
    difference = models.CharField(max_length=32)
    # 展示训练损失的图片路径
    loss = models.CharField(max_length=32)
    
    mse = models.FloatField()
    rmse = models.FloatField()
    mae = models.FloatField()
    # 训练好的模型
    model = models.CharField(max_length=128)