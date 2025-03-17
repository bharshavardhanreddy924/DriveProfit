import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('your_data.csv')  # Update with the actual filename

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis
data['sentiment_score'] = data['Col4'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

# Normalize sentiment score to range 0-10
scaler = MinMaxScaler(feature_range=(1, 10))
data['normalized_sentiment'] = scaler.fit_transform(data[['sentiment_score']])

# Count number of reviews
total_reviews = len(data)
data['review_weight'] = np.log1p(total_reviews)

# Compute final score (weighted sentiment + review impact)
data['final_score'] = (data['normalized_sentiment'] * 0.7) + (data['review_weight'] * 0.3)
data['final_score'] = data['final_score'].round(1)

# Categorize performance in terms of sales
def classify_sales(score):
    if score >= 8:
        return "High Demand"
    elif score >= 5:
        return "Moderate Demand"
    else:
        return "Low Demand"

data['sales_performance'] = data['final_score'].apply(classify_sales)

# Save the processed data
data.to_csv('car_sentiment_analysis.csv', index=False)

# Display results
print(data[['Col1', 'final_score', 'sales_performance']])