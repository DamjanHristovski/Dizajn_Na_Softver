#!/bin/bash

# Django
python manage.py runserver 0.0.0.0:8000 &

#  Streamlit
streamlit run /app/stock_predictor/stockapp/tehnical_analysis_csv.py
