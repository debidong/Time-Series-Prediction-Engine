from .views import *
from django.urls import path

urlpatterns = [
    path('algos', GetAlgorithmView.as_view()),
    path('algo/insert', AlgorithmView.as_view()),
    path('algo/delete', AlgorithmView.as_view()),
    path('train', TrainingView.as_view()),
    path('result', ResultView.as_view()),
]