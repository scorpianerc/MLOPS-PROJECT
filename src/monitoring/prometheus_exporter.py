"""
Prometheus Exporter for Sentiment Analysis Metrics
Reads from PostgreSQL database and exposes metrics for Grafana
"""

import time
import os
from prometheus_client import start_http_server, Gauge, Info
import logging
from sqlalchemy import create_engine, text
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define metrics
total_reviews = Gauge('sentiment_total_reviews', 'Total number of reviews')
predicted_reviews = Gauge('sentiment_predicted_reviews', 'Number of reviews with predictions')
unpredicted_reviews = Gauge('sentiment_unpredicted_reviews', 'Number of reviews without predictions')

positive_reviews = Gauge('sentiment_positive_reviews', 'Number of positive reviews')
negative_reviews = Gauge('sentiment_negative_reviews', 'Number of negative reviews')
neutral_reviews = Gauge('sentiment_neutral_reviews', 'Number of neutral reviews')

positive_pct = Gauge('sentiment_positive_percentage', 'Percentage of positive reviews')
negative_pct = Gauge('sentiment_negative_percentage', 'Percentage of negative reviews')
neutral_pct = Gauge('sentiment_neutral_percentage', 'Percentage of neutral reviews')

average_rating = Gauge('sentiment_average_rating', 'Average rating of all reviews')
average_thumbs_up = Gauge('sentiment_average_thumbs_up', 'Average thumbs up count')

# Model info
model_info = Info('sentiment_model', 'Sentiment analysis model information')

def get_db_connection():
    """Get PostgreSQL connection"""
    user = os.getenv('POSTGRES_USER', 'sentiment_user')
    password = os.getenv('POSTGRES_PASSWORD', 'password')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB', 'sentiment_db')
    
    url = f'postgresql://{user}:{password}@{host}:{port}/{database}'
    return create_engine(url)

def update_metrics():
    """Update metrics from PostgreSQL database"""
    try:
        engine = get_db_connection()
        
        with engine.connect() as conn:
            # Total reviews
            result = conn.execute(text("SELECT COUNT(*) FROM reviews"))
            total = result.scalar()
            total_reviews.set(total)
            
            # Predicted vs unpredicted
            result = conn.execute(text("SELECT COUNT(*) FROM reviews WHERE sentiment IS NOT NULL"))
            predicted = result.scalar()
            predicted_reviews.set(predicted)
            unpredicted_reviews.set(total - predicted)
            
            # Sentiment distribution
            result = conn.execute(text("""
                SELECT sentiment, COUNT(*) as count
                FROM reviews
                WHERE sentiment IS NOT NULL
                GROUP BY sentiment
            """))
            
            sentiment_counts = {row[0]: row[1] for row in result}
            pos_count = sentiment_counts.get('positive', 0)
            neg_count = sentiment_counts.get('negative', 0)
            neu_count = sentiment_counts.get('neutral', 0)
            
            positive_reviews.set(pos_count)
            negative_reviews.set(neg_count)
            neutral_reviews.set(neu_count)
            
            # Percentages
            if predicted > 0:
                positive_pct.set((pos_count / predicted) * 100)
                negative_pct.set((neg_count / predicted) * 100)
                neutral_pct.set((neu_count / predicted) * 100)
            
            # Average rating
            result = conn.execute(text("SELECT AVG(rating) FROM reviews"))
            avg_rating = result.scalar() or 0
            average_rating.set(float(avg_rating))
            
            # Average thumbs up
            result = conn.execute(text("SELECT AVG(thumbs_up) FROM reviews"))
            avg_thumbs = result.scalar() or 0
            average_thumbs_up.set(float(avg_thumbs))
        
        # Load model info
        try:
            metrics_path = Path('models/metrics.json')
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                    model_info.info({
                        'accuracy': str(metrics.get('accuracy', 'N/A')),
                        'precision': str(metrics.get('precision', 'N/A')),
                        'recall': str(metrics.get('recall', 'N/A')),
                        'f1_score': str(metrics.get('f1_score', 'N/A'))
                    })
        except Exception as e:
            logger.warning(f"Could not load model metrics: {e}")
        
        logger.info(f"✅ Metrics updated: {total} total, {predicted} predicted, {total-predicted} unpredicted")
        
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
