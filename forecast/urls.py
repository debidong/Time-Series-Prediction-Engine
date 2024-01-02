from django.urls import path, include

from .views import ARView, ARIMAView, FbprophetView
from analysis.views import ResultView

urlpatterns = [
    path('regression/ar', ARView.as_view()),
    path('regression/arima', ARIMAView.as_view()),
    path('regression/fbprophet', FbprophetView.as_view())
]