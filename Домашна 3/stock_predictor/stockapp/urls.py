from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home Page
    path('companies/', views.companies, name='companies'),  # Companies List Page
    path('company/<str:company_code>/technical/', views.technical_analysis, name='technical_analysis'), # Company tech analysis
    path('company/<str:company_code>/fundamental-analysis/', views.fundamental_analysis, name='fundamental_analysis'), # Company tech analysis
    path('company/<str:company_code>/', views.company_details, name='company_detail'),  # Company Details Page
    path('about/', views.about, name='about'),  # About Us Page
]