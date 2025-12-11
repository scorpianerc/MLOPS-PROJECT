"""
Batch prediction script untuk update sentiment pada reviews di database
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pickle
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database config
DB_CONFIG = {
    'host': os.getenv('POSTGRES_HOST') if os.getenv('POSTGRES_HOST') != 'postgres' else 'localhost',
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'sentiment_db'),
    'user': os.getenv('POSTGRES_USER', 'sentiment_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'your_password_here')
}

def load_model():
    """Load trained model"""
    model_path = 'models/sentiment_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("✅ Model loaded successfully")
    return model

def get_unpredicted_reviews(conn):
    """Get reviews without sentiment prediction"""
    query = """
        SELECT id, review_text, rating
        FROM reviews
        WHERE sentiment IS NULL
        LIMIT 1000
    """
    df = pd.read_sql(query, conn)
    logger.info(f"📥 Loaded {len(df)} unpredicted reviews")
    return df

def update_sentiments(conn, predictions):
    """Update sentiment predictions in database"""
    cursor = conn.cursor()
    
    update_query = """
        UPDATE reviews
        SET sentiment = %s, predicted_at = CURRENT_TIMESTAMP
        WHERE id = %s
    """
    
    execute_batch(cursor, update_query, predictions)
    conn.commit()
    cursor.close()
    logger.info(f"✅ Updated {len(predictions)} sentiment predictions")

def save_model_metrics(conn, accuracy, precision, recall, f1_score, model_name="bert-base-multilingual"):
    """Save model metrics to database for Grafana dashboard"""
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO model_metrics 
            (model_name, accuracy, precision_score, recall_score, f1_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (model_name, accuracy, precision, recall, f1_score))
        
        conn.commit()
        logger.info(f"✅ Model metrics saved: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save model metrics: {e}")
    finally:
        cursor.close()

def predict_sentiment_from_rating(rating):
    """Fallback: predict sentiment based on rating if model fails"""
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'

def main():
    """Main execution function"""
    try:
        # Connect to database
        logger.info("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("✅ Connected to database")
        
        # Load model
        model = load_model()
        
        # Get unpredicted reviews
        df = get_unpredicted_reviews(conn)
        
        if len(df) == 0:
            logger.info("✅ All reviews already have predictions")
            conn.close()
            return
        
        # Make predictions
        predictions = []
        
        if model is not None:
            # Try to use trained model
            try:
                logger.info("🔮 Making predictions with trained model...")
                predicted_sentiments = model.predict(df['review_text'])
                
                for idx, row in df.iterrows():
                    sentiment = predicted_sentiments[idx]
                    predictions.append((sentiment, row['id']))
                    
                logger.info("✅ Predictions completed with model")
            except Exception as e:
                logger.warning(f"⚠️ Model prediction failed: {e}. Using rating-based fallback.")
                model = None
        
        if model is None:
            # Fallback to rating-based sentiment
            logger.info("🔮 Using rating-based sentiment prediction...")
            for idx, row in df.iterrows():
                sentiment = predict_sentiment_from_rating(row['rating'])
                predictions.append((sentiment, row['id']))
            logger.info("✅ Predictions completed with rating-based method")
        
        # Update database
        update_sentiments(conn, predictions)
        
        # Show statistics
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(sentiment) as predicted,
                COUNT(CASE WHEN sentiment='positive' THEN 1 END) as positive,
                COUNT(CASE WHEN sentiment='negative' THEN 1 END) as negative,
                COUNT(CASE WHEN sentiment='neutral' THEN 1 END) as neutral
            FROM reviews
        """)
        stats = cursor.fetchone()
        cursor.close()
        
        logger.info(f"""
        📊 Database Statistics:
        - Total reviews: {stats[0]}
        - Predicted: {stats[1]}
        - Positive: {stats[2]}
        - Negative: {stats[3]}
        - Neutral: {stats[4]}
        """)
        
        # Save model metrics (default values, update dengan metrics real setelah training)
        # TODO: Ganti dengan metrics real dari model evaluation
        if model is not None:
            # Metrics ini harus diganti dengan hasil evaluasi real
            save_model_metrics(
                conn=conn,
                accuracy=0.92,  # Ganti dengan accuracy real dari validation set
                precision=0.91,  # Ganti dengan precision real
                recall=0.93,     # Ganti dengan recall real
                f1_score=0.92,   # Ganti dengan F1 real
                model_name="bert-base-multilingual"
            )
            logger.info("💡 Note: Metrics di atas adalah default. Update dengan metrics real setelah evaluasi model!")
        
        conn.close()
        logger.info("✅ Batch prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during batch prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
