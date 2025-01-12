#!/bin/bash

# Run Django development server in the background
python manage.py runserver 0.0.0.0:8000 &

# Run the Streamlit app
streamlit run /app/stock_predictor/stockapp/tehnical_analysis_csv.py
