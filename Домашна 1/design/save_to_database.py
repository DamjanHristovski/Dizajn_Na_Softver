import mysql
import re
import concurrent.futures
import pandas as pd
import sqlite3
import time
from joblib import Parallel, delayed
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.chrome.options import Options
from datetime import datetime, timedelta
from mysql.connector import (connection)
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, DECIMAL

# Postepeno resenie

start_year = 2014
end_year = 2024


def company_names():  # Filter 1
    url = "https://www.mse.mk/mk/stats/symbolhistory/kmb"
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    select_element = Select(driver.find_element(By.ID, "Code"))
    company_codes = [option.get_attribute("value") for option in select_element.options if
                     option.get_attribute("value") and not re.search(r'\d', option.get_attribute("value"))]

    driver.quit()
    return company_codes


def connect_mysql():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='#Sedi_Madro_Da_Ne$BudeModro69',
        database='berza_data'
    )


def insert_company_name(names):  # 12 secs exe time
    conn = connect_mysql()
    cursor = conn.cursor()
    sql = "INSERT IGNORE INTO company (name) VALUES (%s)"
    cursor.executemany(sql, [(name,) for name in names])
    conn.commit()
    cursor.close()
    conn.close()


def insert_one_company_name(name):
    conn = connect_mysql()
    cursor = conn.cursor()
    sql = "INSERT IGNORE INTO company (name) VALUES (%s)"
    cursor.execute(sql, (name,))
    conn.commit()
    cursor.close()
    conn.close()


def get_company_id(company_name: str):
    conn = connect_mysql()
    cursor = conn.cursor()
    sql = "SELECT company_id from company WHERE name=%s"
    cursor.execute(sql, (company_name,))
    result = cursor.fetchone()
    conn.close()
    cursor.close()
    return result[0]


def find_company_max_date(company_name: str):  # Filter 2
    conn = connect_mysql()
    cursor = conn.cursor()
    sql = "SELECT MAX(date) from daily_data WHERE company_id = (SELECT company_id FROM company WHERE name = %s LIMIT 1)"
    cursor.execute(sql, (company_name,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result[0] else None


def clean_numeric_column(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name].replace({',': ''}, regex=True)
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df


def insert_daily_data(df, company_name):
    numeric_columns = [
        'last_transaction', 'max_price', 'min_price', 'average_price',
        'volume', 'BEST_profit', 'total_profit'
    ]

    for col in numeric_columns:
        df = clean_numeric_column(df, col)

    engine = create_engine('mysql+mysqldb://root:#Sedi_Madro_Da_Ne$BudeModro69@localhost/berza_data')

    df.to_sql(
        name='daily_data',
        con=engine,
        if_exists='append',
        index=False,
        dtype={
            'last_transaction': DECIMAL(10, 2),
            'max_price': DECIMAL(10, 2),
            'min_price': DECIMAL(10, 2),
            'average_price': DECIMAL(10, 2),
            'volume': DECIMAL(10, 2),
            'BEST_profit': DECIMAL(20, 2),
            'total_profit': DECIMAL(20, 2)
        }
    )
    engine.dispose()


def scrape_data_for_company(company_code: str):
    base_url = "https://www.mse.mk/mk/stats/symbolhistory/"
    url = f"{base_url}{company_code}"
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    data = []
    company_id = get_company_id(company_code)
    for year in range(start_year, end_year + 1):
        from_date = driver.find_element(By.ID, "FromDate")
        from_date.clear()
        from_date.send_keys(f"01.01.{year}")

        to_date = driver.find_element(By.ID, "ToDate")
        to_date.clear()
        to_date.send_keys(f"31.12.{year}")

        button = driver.find_element(By.CLASS_NAME, "btn-primary-sm")
        driver.execute_script("arguments[0].scrollIntoView();", button)
        button.click()
        time.sleep(1)
        request = driver.page_source
        soup = BeautifulSoup(request, 'html.parser')
        body = soup.select('tbody tr')
        for b in body:
            cols = b.find_all('td')
            date_str = cols[0].text.strip()
            date_obj = datetime.strptime(date_str, "%d.%m.%Y")
            formatted_date = date_obj.strftime("%Y-%m-%d")
            if len(cols) >= 8:
                data.append({
                    "company_id": company_id,
                    "date": formatted_date,
                    "last_transaction": cols[1].text.strip().replace(".", "c").replace(",", ".").replace("c", ","),
                    "max_price": cols[2].text.strip().replace(".", "c").replace(",", ".").replace("c", ","),
                    "min_price": cols[3].text.strip().replace(".", "c").replace(",", ".").replace("c", ","),
                    "average_price": cols[4].text.strip().replace(".", "c").replace(",", ".").replace("c", ","),
                    "volume": cols[6].text.strip().replace(".", "c").replace(",", ".").replace("c", ","),
                    "BEST_profit": cols[7].text.strip().replace(".", "c").replace(",", ".").replace("c", ","),
                    "total_profit": cols[8].text.strip().replace(".", "c").replace(",", ".").replace("c", ",")
                })

    df = pd.DataFrame(data)
    driver.quit()
    return df


def scrape_and_insert(company_name):
    df = scrape_data_for_company(company_name)
    latest_date = find_company_max_date(company_name)

    if latest_date:
        df['date'] = pd.to_datetime(df['date'])
        latest_date = pd.to_datetime(latest_date)
        df = df[df['date'] > latest_date]

    if not df.empty:
        insert_daily_data(df, get_company_id(company_name))


def save_to_MYSQL_data_threads(names):
    Parallel(n_jobs=10)(delayed(scrape_and_insert)(name) for name in names)


def save_to_MYSQL_data_tester(name):
    scrape_and_insert(name)


def main():
    start_time = time.time()
    names = company_names()
    insert_company_name(names)
    save_to_MYSQL_data_threads(names)

    end_time = time.time()

    print(f'Time taken : {end_time - start_time} seconds')


if __name__ == '__main__':
    main()
