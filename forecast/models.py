from django.db import models

STATUS = [
    "AR",
    "ARIMA",
    "Fbprophet",
    "DL" # deep learning
]

class File(models.Model):
    '''文件模型
    '''
    path = models.URLField()
    name = models.CharField(max_length=128)
    row = models.IntegerField()
    column = models.IntegerField()
    created = models.DateTimeField()

# class Result(models.Model):
#     file = models.ForeignKey(to=File, on_delete=models.CASCADE)
#     forecast = models.JSONField()
#     figure = models.CharField(max_length=128)
#     model = models.CharField(max_length=11, choices=STATUS)
    

from django.db import models