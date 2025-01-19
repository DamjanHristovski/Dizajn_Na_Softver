FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    default-libmysqlclient-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install tensorflow==2.18.0

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Django, Streamlit
EXPOSE 8000
EXPOSE 8501

ENV DJANGO_SETTINGS_MODULE=stock_predictor.settings

CMD ["/bin/bash", "/app/start.sh"]
