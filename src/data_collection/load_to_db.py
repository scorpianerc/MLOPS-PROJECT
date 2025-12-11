"""
Script untuk load data dari CSV ke PostgreSQL
"""
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DB_CONFIG = {
    'host': 'localhost',
    'port': int(os.getenv('POSTGRES_PORT', 5432)),
    'database': os.getenv('POSTGRES_DB', 'sentiment_db'),
    'user': os.getenv('POSTGRES_USER', 'sentiment_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'your_password_here')
}

def load_csv_to_postgres(csv_path='data/raw/reviews.csv'):
    """Load reviews from CSV to PostgreSQL"""
    
    # Read CSV
    logger.info(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded {len(df)} reviews from CSV")
    
    # Connect to database
    logger.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    logger.info("✅ Connected to database")
    
    # Check existing data
    cursor.execute("SELECT COUNT(*) FROM reviews")
    existing_count = cursor.fetchone()[0]
    logger.info(f"📊 Existing reviews in database: {existing_count}")
    
    # Get existing review IDs to avoid duplicates
    cursor.execute("SELECT review_id FROM reviews")
    existing_ids = {row[0] for row in cursor.fetchall()}
    logger.info(f"Found {len(existing_ids)} existing review IDs")
    
    # Filter out duplicates
    df_new = df[~df['review_id'].isin(existing_ids)]
    logger.info(f"📥 New reviews to insert: {len(df_new)}")
    
    if len(df_new) == 0:
        logger.info("✅ No new reviews to insert")
        cursor.close()
        conn.close()
        return
    
    # Prepare insert query
    insert_query = """
        INSERT INTO reviews (review_id, review_text, rating, review_date, thumbs_up, app_version)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (review_id) DO NOTHING
    """
    
    # Prepare data for batch insert
    data = []
    for _, row in df_new.iterrows():
        data.append((
            row.get('review_id', ''),
            row.get('review_text', ''),
            int(row.get('rating', 3)),
            row.get('date', None) or row.get('review_date', None),
            int(row.get('thumbs_up', 0)) if pd.notna(row.get('thumbs_up')) else 0,
            row.get('app_version', '') if pd.notna(row.get('app_version')) else None
        ))
    
    # Batch insert
    logger.info("📤 Inserting new reviews...")
    execute_batch(cursor, insert_query, data, page_size=500)
    conn.commit()
    logger.info(f"✅ Inserted {len(data)} new reviews")
    
    # Get final count
    cursor.execute("SELECT COUNT(*) FROM reviews")
    final_count = cursor.fetchone()[0]
    logger.info(f"📊 Total reviews in database: {final_count}")
    
    # Show statistics
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(sentiment) as with_sentiment,
            AVG(rating)::numeric(10,2) as avg_rating,
            MIN(review_date) as earliest,
            MAX(review_date) as latest
        FROM reviews
    """)
    stats = cursor.fetchone()
    
    logger.info(f"""
    📊 Database Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Total Reviews:        {stats[0]}
    With Sentiment:       {stats[1]} ({stats[1]/stats[0]*100:.1f}%)
    Average Rating:       {stats[2]}
    Date Range:           {stats[3]} to {stats[4]}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    cursor.close()
    conn.close()
    logger.info("✅ Data load completed successfully!")

if __name__ == '__main__':
    load_csv_to_postgres()
