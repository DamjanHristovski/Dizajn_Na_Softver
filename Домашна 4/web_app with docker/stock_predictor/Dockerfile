FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Django, Streamlit
EXPOSE 8000
EXPOSE 8501

ENV DJANGO_SETTINGS_MODULE=stock_predictor.settings

CMD ["/bin/bash", "/app/start.sh"]
