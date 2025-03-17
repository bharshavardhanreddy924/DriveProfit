from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup

# URL for Mercedes-Benz GLA user reviews on Cardekho
url = 'https://www.cardekho.com/mercedes-benz/gla/user-reviews'

# Set up Selenium in headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in the background
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=chrome_options)

try:
    # Load the page
    driver.get(url)
    time.sleep(5)  # Wait for JavaScript to load content
    
    # Scroll down to ensure all reviews are loaded
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)
    
    # Get the fully loaded page source
    page_source = driver.page_source
finally:
    driver.quit()

# Parse with BeautifulSoup
soup = BeautifulSoup(page_source, 'lxml')

# Extract reviews using the correct selector (div with class 'contentspace')
reviews = soup.select('div.contentspace')

if not reviews:
    print("No reviews found. Check if the page structure has changed.")
else:
    print("\nUser Reviews:\n")
    for idx, review in enumerate(reviews, start=1):
        print(f"Review {idx}:\n{review.get_text(strip=True)}\n{'='*80}")
