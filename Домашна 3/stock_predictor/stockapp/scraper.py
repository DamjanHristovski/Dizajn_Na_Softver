import os
import csv
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By


def scrape_company_names(url, output_csv_path):
    """
    Scrapes company codes and names from the provided URL and saves them into a CSV file.
    Only new company codes will be added if they don't already exist in the CSV.

    Args:
        url (str): The URL to scrape the company codes from.
        output_csv_path (str): The path where the CSV file will be saved.

    Returns:
        list: A list of company data (ID, code, name).
    """
    print(f"Starting scraping from: {url}")

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Set up the Chrome WebDriver with headless option
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)
    print("WebDriver initialized.")

    try:
        # Open the page
        driver.get(url)
        print("Page loaded successfully.")

        # Find the dropdown and get the company codes (excluding those with digits)
        print("Extracting company codes...")
        select_element = Select(driver.find_element(By.ID, "Code"))
        company_codes = [
            option.get_attribute("value")
            for option in select_element.options
            if option.get_attribute("value") and not re.search(r'\d', option.get_attribute("value"))
        ]
        print(f"Found {len(company_codes)} company codes: {company_codes}")

        # Prepare a list to store company data (ID, code, name)
        companies = []

        # Check if the CSV already exists and load existing company codes to avoid duplicates
        if os.path.exists(output_csv_path):
            existing_codes = set()
            with open(output_csv_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header
                for row in reader:
                    existing_codes.add(row[1])  # Add company codes to the set

            print(f"Existing company codes loaded: {len(existing_codes)}")

        else:
            existing_codes = set()

        # Loop through each company code
        for index, code in enumerate(company_codes, start=1):
            if code in existing_codes:
                print(f"Skipping existing company code {code}")
                continue  # Skip if the company code is already in the CSV

            print(f"Processing new company code {code} (ID: {index})...")
            company_url = f"https://www.mse.mk/mk/symbol/{code}"
            driver.get(company_url)

            # Check if the page redirects to a company page
            if "issuer" in driver.current_url:
                print("Detected 'issuer' in URL structure.")
                try:
                    company_name = driver.find_element(By.CSS_SELECTOR, "div#izdavach div.col-md-8.title").text
                except Exception as e:
                    print(f"Error extracting name (structure 1): {e}")
                    company_name = "Name not found"
            else:
                print("No 'issuer' in URL. Attempting alternative structure...")
                try:
                    raw_name = driver.find_element(By.ID, "titleKonf2011").text
                    print(f"Raw company name: {raw_name}")
                    company_name = raw_name.split(" - ", 2)[-1] if " - " in raw_name else raw_name
                except Exception as e:
                    print(f"Error extracting name (structure 2): {e}")
                    company_name = "Name not found"

            print(f"Extracted company name: {company_name}")
            companies.append([index, code, company_name])

        # Save the new company data to the CSV file (append mode)
        if companies:
            print(f"Saving new data to: {output_csv_path}")
            with open(output_csv_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Write header only if the file is empty
                if file.tell() == 0:  # Check if the file is empty
                    writer.writerow(["Company_ID", "Company_Code", "Company_Name"])
                writer.writerows(companies)

        print(f"Scraping completed. Data saved to {output_csv_path}.")
        return companies

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    finally:
        driver.quit()
        print("Web driver closed.")
