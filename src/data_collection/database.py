"""
Database utilities untuk menyimpan dan membaca data
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

Base = declarative_base()


class Review(Base):
    """Model untuk tabel reviews di PostgreSQL"""
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(255), unique=True, index=True)
    app_id = Column(String(100), index=True)
    user_name = Column(String(255))
    review_text = Column(Text)
    rating = Column(Integer)
    thumbs_up = Column(Integer)
    app_version = Column(String(50))
    review_date = Column(DateTime)
    scraped_at = Column(DateTime)
    
    # Sentiment analysis results
    sentiment = Column(String(20))  # positive, negative, neutral
    sentiment_score = Column(Float)
    predicted_at = Column(DateTime)


class DatabaseManager:
    """Manager untuk PostgreSQL dan MongoDB"""
    
    def __init__(self):
        """Initialize database connections"""
        # PostgreSQL
        postgres_url = self._get_postgres_url()
        self.pg_engine = create_engine(postgres_url)
        Base.metadata.create_all(self.pg_engine)
        self.Session = sessionmaker(bind=self.pg_engine)
        
        # MongoDB
        mongo_host = os.getenv('MONGO_HOST', 'localhost')
        mongo_port = int(os.getenv('MONGO_PORT', 27017))
        self.mongo_client = MongoClient(f'mongodb://{mongo_host}:{mongo_port}/')
        self.mongo_db = self.mongo_client[os.getenv('MONGO_DB', 'sentiment_reviews')]
        self.reviews_collection = self.mongo_db['reviews']
        self.predictions_collection = self.mongo_db['predictions']
        
    def _get_postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        user = os.getenv('POSTGRES_USER', 'sentiment_user')
        password = os.getenv('POSTGRES_PASSWORD', 'password')
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        database = os.getenv('POSTGRES_DB', 'sentiment_db')
        
        return f'postgresql://{user}:{password}@{host}:{port}/{database}'
    
    def save_reviews_to_postgres(self, df: pd.DataFrame) -> int:
        """
        Simpan reviews ke PostgreSQL
        
        Args:
            df: DataFrame berisi reviews
        
        Returns:
            Jumlah rows yang berhasil disimpan
        """
        try:
            session = self.Session()
            saved_count = 0
            
            for _, row in df.iterrows():
                # Check if review already exists
                existing = session.query(Review).filter_by(
                    review_id=row['review_id']
                ).first()
                
                if existing is None:
                    review = Review(
                        review_id=row['review_id'],
                        app_id=row['app_id'],
                        user_name=row['user_name'],
                        review_text=row['review_text'],
                        rating=row['rating'],
                        thumbs_up=row['thumbs_up'],
                        app_version=row.get('app_version'),
                        review_date=row['review_date'],
                        scraped_at=row['scraped_at']
                    )
                    session.add(review)
                    saved_count += 1
            
            session.commit()
            logger.info(f"Saved {saved_count} new reviews to PostgreSQL")
            return saved_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving to PostgreSQL: {str(e)}")
            raise
        finally:
            session.close()
    
    def save_reviews_to_mongodb(self, df: pd.DataFrame) -> int:
        """
        Simpan reviews ke MongoDB
        
        Args:
            df: DataFrame berisi reviews
        
        Returns:
            Jumlah documents yang berhasil disimpan
        """
        try:
            # Convert DataFrame to dict
            records = df.to_dict('records')
            
            # Convert datetime to string for MongoDB
            for record in records:
                if 'review_date' in record:
                    record['review_date'] = record['review_date'].isoformat()
                if 'scraped_at' in record:
                    record['scraped_at'] = record['scraped_at'].isoformat()
            
            # Insert or update
            saved_count = 0
            for record in records:
                result = self.reviews_collection.update_one(
                    {'review_id': record['review_id']},
                    {'$set': record},
                    upsert=True
                )
                if result.upserted_id or result.modified_count > 0:
                    saved_count += 1
            
            logger.info(f"Saved {saved_count} reviews to MongoDB")
            return saved_count
            
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {str(e)}")
            raise
    
    def update_predictions(self, predictions: List[Dict]) -> int:
        """
        Update sentiment predictions di PostgreSQL
        
        Args:
            predictions: List of dicts dengan review_id, sentiment, sentiment_score
        
        Returns:
            Jumlah rows yang di-update
        """
        try:
            session = self.Session()
            updated_count = 0
            
            for pred in predictions:
                review = session.query(Review).filter_by(
                    review_id=pred['review_id']
                ).first()
                
                if review:
                    review.sentiment = pred['sentiment']
                    review.sentiment_score = pred['sentiment_score']
                    review.predicted_at = datetime.now()
                    updated_count += 1
            
            session.commit()
            logger.info(f"Updated {updated_count} predictions in PostgreSQL")
            return updated_count
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating predictions: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_unpredicted_reviews(self, limit: int = 100) -> pd.DataFrame:
        """
        Ambil reviews yang belum diprediksi
        
        Args:
            limit: Maksimal jumlah reviews
        
        Returns:
            DataFrame berisi reviews
        """
        try:
            session = self.Session()
            
            reviews = session.query(Review).filter(
                Review.sentiment.is_(None)
            ).order_by(Review.scraped_at.desc()).limit(limit).all()
            
            data = []
            for review in reviews:
                data.append({
                    'review_id': review.review_id,
                    'review_text': review.review_text,
                    'rating': review.rating
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error getting unpredicted reviews: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_sentiment_stats(self) -> Dict:
        """
        Ambil statistik sentiment
        
        Returns:
            Dictionary berisi statistik
        """
        try:
            session = self.Session()
            
            total = session.query(Review).count()
            predicted = session.query(Review).filter(
                Review.sentiment.isnot(None)
            ).count()
            
            positive = session.query(Review).filter_by(sentiment='positive').count()
            negative = session.query(Review).filter_by(sentiment='negative').count()
            neutral = session.query(Review).filter_by(sentiment='neutral').count()
            
            stats = {
                'total_reviews': total,
                'predicted_reviews': predicted,
                'unpredicted_reviews': total - predicted,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'positive_pct': (positive / predicted * 100) if predicted > 0 else 0,
                'negative_pct': (negative / predicted * 100) if predicted > 0 else 0,
                'neutral_pct': (neutral / predicted * 100) if predicted > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting sentiment stats: {str(e)}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connections"""
        self.pg_engine.dispose()
        self.mongo_client.close()
