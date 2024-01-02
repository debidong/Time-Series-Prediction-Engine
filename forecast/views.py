from rest_framework.views import APIView, Response, status
from .lib.regression import AR, ARIMA_model, Fbprophet

class ARView(APIView):
    def post(self, request):
        try:
            pk = request.data.get('pk')
            order = request.data.get('order')
            target = request.data.get('target')
            window = request.data.get('window')
            step = request.data.get('step')
            AR(pk, target, order, window, step)
            res = {
                "status": 200,
                "content": "提交成功"
            }
            return Response(res, status=status.HTTP_200_OK)
        except:
            res = {
                "status": 500,
                "content": "参数有误"
            }
            return Response(res, status=status.HTTP_400_BAD_REQUEST)

class ARIMAView(APIView):
    def post(self, request):
        pk = request.data.get('pk')
        order = request.data.get('order')
        target = request.data.get('target')
        window = request.data.get('window')
        step = request.data.get('step')
        ARIMA_model(pk, target, order, window, step)
        res = {
            "status": 200,
            "content": "提交成功"
        }
        return Response(res, status=status.HTTP_200_OK)
    
class FbprophetView(APIView):
    def post(self, request):
        pk = request.data.get('pk')
        target = request.data.get('target')
        window = request.data.get('window')
        step = request.data.get('step')
        Fbprophet(pk, target, window, step)
        res = {
            "status": 200,
            "content": "提交成功"
        }
        return Response(res, status=status.HTTP_200_OK)