from django.urls import path, include

from .views import ARView, ARIMAView
from analysis.views import ResultView

urlpatterns = [
    path('regression/ar', ARView.as_view()),
    path('regression/arima', ARIMAView.as_view()),
]