#!/bin/bash

# Django
python manage.py runserver 0.0.0.0:8000 &

#  Streamlit
streamlit run /app/stockapp/tehnical_analysis_csv.py
