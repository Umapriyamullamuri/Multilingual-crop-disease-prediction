from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_disease, name='precictor'),       # Home page
    path('predict/', views.predict_disease, name='predict'), # ✅ Handles /predict/ POST requests
    path('voice/', views.v_input, name='voice_input'),       # Voice input page
]
