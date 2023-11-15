from django.db import models
from file.models import File


# Create your models here.
class File(models.Model):
    '''算法模型
    '''
    description = models.CharField(max_length=128)
    name = models.CharField(max_length=128)
    
    neuralNetwork = models.CharField(max_length=16)
    layers = models.IntegerField()
    learningRate = models.FloatField()
    neurons = models.IntegerField()
    round = models.IntegerField()
    batchSize = models.FloatField()

    optimization = models.CharField(max_length=16)

    verificationRate = models.FloatField()
    dataset = models.ForeignKey(to=File, related_name="dataset" ,on_delete=models.CASCADE)
    target = models.ForeignKey(to=File, related_name="target" ,on_delete=models.CASCADE)