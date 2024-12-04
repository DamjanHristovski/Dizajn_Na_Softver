from django.http import HttpResponse
from django.shortcuts import render
from .models import Company

# Create your views here.
def generate_paragraph(request):
    return render(request, 'generate_paragraph.html')

def list_companies(request):
    companies = Company.objects.all()
    total_companies = companies.count()
    return render(request, 'list_companies.html', {'companies': companies, 'total_companies': total_companies})
