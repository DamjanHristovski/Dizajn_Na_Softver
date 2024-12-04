from django.urls import path
from . import views

urlpatterns = [
    path('generate-paragraph/', views.generate_paragraph, name='generate_paragraph'),
    path('list-companies/', views.list_companies, name='list_companies'),
]