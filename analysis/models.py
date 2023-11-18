from django.db import models
from file.models import File


# Create your models here.

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
    ("FIN", "finished")
]

class Algorithm(models.Model):
    '''算法模型
    '''
    description = models.CharField(max_length=128)
    name = models.CharField(max_length=128)
    
    neuralNetwork = models.CharField(max_length=16)
    layers = models.IntegerField()
    learningRate = models.FloatField()
    neurons = models.IntegerField()
    rounds = models.IntegerField()
    batchSize = models.FloatField()

    optimization = models.CharField(max_length=16)

    verificationRate = models.FloatField()
    dataset = models.ForeignKey(to=File, related_name="dataset" ,on_delete=models.CASCADE)
    target = models.CharField(max_length=128)

    status = models.CharField(max_length=11, choices=STATUS, default="INI")