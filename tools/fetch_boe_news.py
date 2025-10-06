import os
import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://www.bankofengland.co.uk/news/news?page={}"
INCLUDE_KEYWORDS = ["minutes", "bank rate", "monetary policy", "mpc"]

def setup_driver():
    chrome_options = Options()
    # REMOVE HEADLESS so we see what’s happening
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_window_size(1920, 1080)
    return driver

def is_relevant(title):
    return any(k in title.lower() for k in INCLUDE_KEYWORDS)

def scrape_articles(driver, max_pages=5):
    all_articles = []

    for page in range(1, max_pages + 1):
        url = BASE_URL.format(page)
        print(f"📥 Scraping: {url}")
        driver.get(url)

        # Scroll down to load dynamic content
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        items = driver.find_elements(By.CSS_SELECTOR, "li.listingItem")

        if not items:
            print(f"⚠️ No articles loaded on page {page}")
            continue

        for item in items:
            try:
                link_elem = item.find_element(By.TAG_NAME, "a")
                title = item.find_element(By.CLASS_NAME, "listingItem-title").text.strip()
                date = item.find_element(By.CLASS_NAME, "listingItem-date").text.strip()
                href = link_elem.get_attribute("href")

                if is_relevant(title):
                    all_articles.append({
                        "title": title,
                        "date": date,
                        "url": href
                    })
            except Exception as e:
                print(f"❌ Skipping item: {e}")
                continue

    return all_articles

def save_to_csv(data, filename="data/boe_filtered_news.csv"):
    os.makedirs("data", exist_ok=True)
    if not data:
        print("⚠️ No relevant items found.")
        return
    print(f"💾 Saving {len(data)} articles to {filename}")
    with open(filename, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def run():
    driver = setup_driver()
    try:
        articles = scrape_articles(driver, max_pages=5)
    finally:
        driver.quit()
    save_to_csv(articles)

if __name__ == "__main__":
    run()
