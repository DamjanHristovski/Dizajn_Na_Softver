from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home Page
    path('companies/', views.companies, name='companies'),  # Companies List Page
#    path('company/<str:company_code>/technical/', views.technical_analysis, name='technical_analysis'), # Company tech analysis
    path('run-streamlit/<str:company_code>/', views.run_streamlit, name='run_streamlit'),
    path('fundamental-analysis/<str:company_code>/', views.fundamental_analysis, name='fundamental_analysis'),
    path('company/<str:company_code>/', views.company_details, name='company_detail'),  # Company Details Page
    path('about/', views.about, name='about'),  # About Us Page
]