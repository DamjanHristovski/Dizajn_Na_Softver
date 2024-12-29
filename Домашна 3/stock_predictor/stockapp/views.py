from django.shortcuts import render
import pandas as pd
import os
from django.conf import settings
from datetime import datetime

# Create your views here.
from django.shortcuts import render
import csv

from .LSTM_3 import predict_stock_prices
from .scraper import scrape_company_names
from .utils import load_csv_data
from django.http import HttpResponse

# Home page
def home(request):
    return render(request, 'home.html')

# List of companies
def companies(request):
    # File path to the CSV
    csv_file_path = os.path.join(settings.BASE_DIR, 'stockapp', 'data', 'company_codes_and_names.csv')

    # Get the last modified time of the CSV file
    try:
        last_modified_timestamp = os.path.getmtime(csv_file_path)
        last_modified_date = datetime.fromtimestamp(last_modified_timestamp).strftime('%d/%m/%Y %H:%M:%S')
    except FileNotFoundError:
        last_modified_date = 'File not found'


    # If the button is clicked to scrape data
    if request.method == "POST":
        # Call the scrape function and update the CSV file
        scrape_company_names(
            "https://www.mse.mk/mk/stats/symbolhistory/kmb",  # URL to scrape
            csv_file_path  # Path to save CSV
        )
        # After scraping, load the new data from the CSV
        companies_df = pd.read_csv(csv_file_path)

    # Load company data
    companies_df = pd.read_csv(csv_file_path)

    # Convert data to a list of dictionaries for easy rendering in the template
    companies_data = companies_df.to_dict(orient='records')

    #print(companies_data[0].keys() if companies_data else "No data") - To check the keys, they will be needed in the html
    return render(request, 'companies.html', {'companies': companies_data, 'last_modified_date': last_modified_date})


# Company details
def company_details(request, company_code):
    # Paths to your CSV files
    company_codes_and_names_path = os.path.join(settings.BASE_DIR, 'stockapp', 'data', 'company_codes_and_names.csv')
    daily_info_path = os.path.join(settings.BASE_DIR, 'stockapp', 'data', 'data_for_showcase.csv')

    # Load the data
    company_codes_and_names = pd.read_csv(company_codes_and_names_path)
    daily_info = pd.read_csv(daily_info_path)

    # Look up Company_Name and Company_ID in company_codes_and_names.csv using the given Company_Code
    company_info = company_codes_and_names[company_codes_and_names['Company_Code'] == company_code]
    if company_info.empty:
        return render(
            request,
            'company_detail.html',
            {
                'company_code': company_code,
                'company_name': None,
                'records': [],
                'company_found': False,
                'prediction_image': None,
            }
        )

    company_id = company_info.iloc[0]['Company_ID']
    company_name = company_info.iloc[0]['Company_Name']

    # Filter daily_info for the found Company_ID
    filtered_records = daily_info[daily_info['company_id'] == company_id]

    # Convert the 'date' column to datetime format (YYYY-MM-DD)
    filtered_records.loc[:, 'date'] = pd.to_datetime(
        filtered_records['date'], format='%Y-%m-%d', errors='coerce'
    )

    # Handle missing values (if any) in other columns
    filtered_records = filtered_records.fillna({
        'last_transaction': 0.0,
        'max_price': 0.0,
        'min_price': 0.0,
        'average_price': 0.0,
        'volume': 0.0,
        'BEST_profit': 0.0,
        'total_profit': 0.0,
    })

    # Convert the filtered records to a dictionary for rendering
    records = filtered_records.to_dict(orient='records')

    # Generate stock price prediction plot
    try:
        prediction_image = predict_stock_prices(company_code)
    except ValueError as e:
        prediction_image = None  # Handle cases where prediction cannot be generated
        print(f"Prediction error: {e}")

    # Render the template with the filtered data
    return render(
        request,
        'company_detail.html',
        {
            'company_code': company_code,
            'company_name': company_name,
            'records': records,
            'company_found': True,
            'prediction_image': prediction_image,
        }
    )


# About us
def about(request):
    return render(request, 'about.html')

