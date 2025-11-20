"""
Simple Prometheus Exporter for Sentiment Analysis Metrics
Reads from CSV and exposes metrics for Grafana
"""

import time
import pandas as pd
from prometheus_client import start_http_server, Gauge, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define metrics
total_reviews = Gauge('sentiment_total_reviews', 'Total number of reviews')
positive_reviews = Gauge('sentiment_positive_reviews', 'Number of positive reviews')
negative_reviews = Gauge('sentiment_negative_reviews', 'Number of negative reviews')
neutral_reviews = Gauge('sentiment_neutral_reviews', 'Number of neutral reviews')
average_rating = Gauge('sentiment_average_rating', 'Average rating')
model_accuracy = Gauge('sentiment_model_accuracy', 'Model accuracy')

def update_metrics():
    """Update metrics from CSV files"""
    try:
        # Load processed reviews
        df = pd.read_csv('data/processed/processed_reviews.csv')
        
        # Update total reviews
        total_reviews.set(len(df))
        
        # Update sentiment distribution
        if 'sentiment_label' in df.columns:
            sentiment_counts = df['sentiment_label'].value_counts()
            positive_reviews.set(sentiment_counts.get('positive', 0))
            negative_reviews.set(sentiment_counts.get('negative', 0))
            neutral_reviews.set(sentiment_counts.get('neutral', 0))
        
        # Update average rating
        if 'rating' in df.columns:
            avg_rating = df['rating'].mean()
            average_rating.set(avg_rating)
        
        # Load model metrics
        import json
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
            model_accuracy.set(metrics.get('accuracy', 0))
        
        logger.info(f"✅ Metrics updated: {len(df)} reviews")
        
    except Exception as e:
        logger.error(f"❌ Error updating metrics: {e}")

if __name__ == '__main__':
    # Start prometheus server on port 8000
    start_http_server(8000)
    logger.info("🚀 Prometheus exporter started on port 8000")
    logger.info("📊 Metrics available at http://localhost:8000/metrics")
    
    # Update metrics every 30 seconds
    while True:
        update_metrics()
        time.sleep(30)
