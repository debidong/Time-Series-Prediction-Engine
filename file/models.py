from django.db import models

class File(models.Model):
    '''文件模型
    '''
    path = models.URLField()
    description = models.CharField(max_length=128)
    name = models.CharField(max_length=128)
    row = models.IntegerField()
    column = models.IntegerField()
    created = models.DateTimeField()