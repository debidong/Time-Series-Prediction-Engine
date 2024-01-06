from django.urls import path, include

from .views import *

urlpatterns = [
    path('regression/ar', ARView.as_view()),
    path('regression/arima', ARIMAView.as_view()),
    path('regression/fbprophet', FbprophetView.as_view()),
    path('regression/nn', InferView.as_view()),
    # path('regression/result', ResultView.as_view()),
    path('file/file', GetFileView.as_view()),
    path('file/upload', FileView.as_view()),
    path('file/insert', InsertView.as_view()),
    path('file/delete', FileView.as_view())
]