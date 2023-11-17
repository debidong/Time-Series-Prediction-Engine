from .views import *
from django.urls import path

urlpatterns = [
    path('file', GetFileView.as_view()),
    path('upload', FileView.as_view()),
    path('insert', FileView.as_view()),
    path('delete', FileView.as_view())
]