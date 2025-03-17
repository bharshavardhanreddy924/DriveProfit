from flask import Flask, render_template, abort, url_for
import requests
from bs4 import BeautifulSoup
import re

app = Flask(__name__)

def get_models():
    """
    Scrapes the Mercedes‑Benz models from Cardekho.
    Returns a dictionary mapping model names to their detail page URLs.
    """
    url = "https://www.cardekho.com/cars/Mercedes-Benz"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except Exception as e:
        print("Error fetching URL:", e)
        return {}
        
    if response.status_code != 200:
        print("Unexpected status code:", response.status_code)
        return {}
    
    soup = BeautifulSoup(response.text, "html.parser")
    models = {}
    
    # Use regex to find hrefs that match the model page format.
    # We assume model URLs are of the format: /cars/Mercedes-Benz/<model-name>
    model_pattern = re.compile(r"^/cars/Mercedes-Benz/[^/]+$")
    
    # Inspect all <a> tags and filter based on the href pattern.
    for a in soup.find_all("a", href=True):
        href = a['href']
        if model_pattern.match(href):
            # Get the text of the link (the model name)
            model_name = a.get_text(strip=True)
            # Sometimes there might be empty strings or unrelated links
            if model_name and model_name not in models:
                # Build the full URL if necessary.
                full_url = href if href.startswith("http") else "https://www.cardekho.com/mercedes-benz/glc/user-reviews" + href
                models[model_name] = full_url

    if not models:
        print("No models found. The page structure might have changed.")
    return models

def get_reviews(model_url):
    """
    Scrapes user reviews from the given model detail page.
    Returns a list of review texts.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(model_url, headers=headers, timeout=10)
    except Exception as e:
        print("Error fetching model details:", e)
        return ["Unable to fetch model details."]
    
    if response.status_code != 200:
        return ["Unable to fetch model details."]

    soup = BeautifulSoup(response.text, "html.parser")
    reviews = []
    
    # --- Adjust the selectors below based on the current structure of the model page ---
    # For example, we assume that reviews may be contained in a <div> with id "reviews"
    review_section = soup.find("div", id="reviews")
    if review_section:
        # In our example, we assume each review is contained within a <div> with class "review-box"
        review_boxes = review_section.find_all("div", class_="review-box")
        for review in review_boxes:
            review_text = review.get_text(strip=True)
            if review_text:
                reviews.append(review_text)
    
    # If no reviews are found by the above method, add a placeholder message.
    if not reviews:
        reviews.append("No reviews found for this model.")
    return reviews

@app.route("/")
def index():
    """
    Home page that shows the list of Mercedes‑Benz models scraped from Cardekho.
    Each model name is a link to its reviews page.
    """
    models = get_models()
    if not models:
        return "Could not fetch models from Cardekho at this time."
    return render_template("index.html", models=models)

@app.route("/model/<model_name>")
def model_reviews(model_name):
    """
    For the given model name, fetch its detail page URL, scrape reviews,
    and render them.
    """
    models = get_models()
    if model_name not in models:
        abort(404, description="Model not found")
    model_url = models[model_name]
    reviews = get_reviews(model_url)
    return render_template("model.html", model_name=model_name, reviews=reviews)

if __name__ == "__main__":
    app.run(debug=True)
