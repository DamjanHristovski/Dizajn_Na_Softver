from django.shortcuts import render
import pandas as pd
import os
from django.conf import settings
from datetime import datetime
import numpy as np
# Create your views here.
from django.shortcuts import render
import csv
from .LSTM_3 import predict_stock_prices
from .fundamental_analysis_csv import scrapeFilteredLinksForCode, scrapeNewsText, nlpPreProcessTheNews, createModel
from .scraper import scrape_company_names
from .utils import load_csv_data
import subprocess
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
            'prediction_image': prediction_image
        }
    )

# Technical analysis view
#def technical_analysis(request, company_code):
#    # Placeholder logic for technical analysis
#    return HttpResponse(f"Technical Analysis for {company_code}")

def run_streamlit(request, company_code):
    if company_code:
        # Start the Streamlit app
        subprocess.Popen([
            "streamlit",
            "run",
            "C:/Users/matej/Desktop/dias_project/stock_predictor/stockapp/tehnical_analysis_csv.py",
            "--server.headless"
        ])

        # Return an HTTP response with a clickable link including the query parameter
        streamlit_url = f"http://localhost:8501/?company_code={company_code}"
        response_html = f"<a href='{streamlit_url}' target='_blank'>Click here to view the Streamlit app for {company_code}</a>"
        return HttpResponse(response_html)
    else:
        return HttpResponse("No company code provided.")


# Fundamental analysis view
def fundamental_analysis(request, company_code):
    links = scrapeFilteredLinksForCode(company_code)
    news = scrapeNewsText(links)

    html_output = """
    <!DOCTYPE html>
    <html>
    <head lang="mk">
    <meta charset="utf-8">
    <title>Фундаментална анализа</title>
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px; }
        h1 { color: #333; text-align: center; padding: 20px; }
        p { color: #666; text-align: justify; padding: 10px; border-bottom: 1px solid #ddd; }
        a { color: #007bff; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
    </head>
    <body>
    <h1>Фундаментална анализа</h1>
    """

    if len(news) == 0:
        html_output += '<p>Не се најдени вести за компанијата</p>'
    else:
        nlpPreProcessedNews = nlpPreProcessTheNews(news)
        model = createModel()
        sentimentOfNews = model.predict(nlpPreProcessedNews)

        numOfPositiveNews = np.count_nonzero(sentimentOfNews == 1)
        numOfNegativeNews = np.count_nonzero(sentimentOfNews == 0)
        numOfNeutralNews = len(sentimentOfNews) - (numOfPositiveNews + numOfNegativeNews)

        html_output += '<h2>НАПОМЕНА: Имајте на ум дека фундаменталната анализа на вестите не е секогаш точна.</h2>'
        html_output += '<h4>Ова може да се случи поради тоа што вестите што ги објавуваат компаниите не се секогаш релевантни за предвидување на цените на акциите.</h4>'
        html_output += '<h4>Исто така, при прегледување на објавените вести од различни компании на англиски јазик, забележано е дека некои од вестите насочуваат кон македонската верзија на вестите за да можете да ги прочитате.</h4>'
        html_output += '<p>Вестите кои беа објавени на англиски јазик се: </p>'

        for n in range(len(news)):
            html_output += f'<h4>{news[n]}</h4>'
            html_output += f'<p>Линк: <a href="{links[n]}">{links[n]}</a></p>'
            if sentimentOfNews[n] == 1:
                html_output += '<p>Ова е позитивна вест.</p>'
            elif sentimentOfNews[n] == 0:
                html_output += '<p>Ова е негативна вест.</p>'
            else:
                html_output += '<p>Ова е неутрална вест.</p>'

        html_output += f'<h4>Бројот на позитивни вести е: {str(numOfPositiveNews)}</h4>'
        html_output += f'<h4>Бројот на негативни вести е: {str(numOfNegativeNews)}</h4>'
        html_output += f'<h4>Бројот на неутрални вести е: {str(numOfNeutralNews)}</h4>'

        if numOfPositiveNews > numOfNegativeNews:
            html_output += '<h4>Се препорачува да се купат акции за оваа компанија.</h4>'
        elif numOfNegativeNews > numOfPositiveNews:
            html_output += '<h4>Не е препорачливо да се купат акции за оваа компанија, а ако доколку имате, треба да размислите за продажба.</h4>'
        else:
            html_output += '<h4>Бројот на позитивни и негативни вести е ист. Во овој случај не треба да се купуваат или продаваат акции.</h4>'

    html_output += '</body></html>'
    return HttpResponse(html_output)

# About us
def about(request):
    return render(request, 'about.html')

