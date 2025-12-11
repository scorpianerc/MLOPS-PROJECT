"""
Feature Store untuk Sentiment Analysis
Menjaga konsistensi features antara training, testing, dan serving
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
import psycopg2
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """
    Extract features dari text untuk sentiment analysis
    Ensures consistent preprocessing antara training dan serving
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize feature extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Indonesian text processors
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        
        # Feature extraction config
        self.max_length = self.config.get('max_length', 128)
        self.min_length = self.config.get('min_length', 3)
        
        logger.info("✅ TextFeatureExtractor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text dengan consistent rules
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove special characters (keep spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove Indonesian stopwords"""
        return self.stopword_remover.remove(text)
    
    def stem_text(self, text: str) -> str:
        """Stem Indonesian text"""
        return self.stemmer.stem(text)
    
    def extract_features(self, text: str, include_metadata: bool = True) -> Dict:
        """
        Extract all features dari text
        
        Args:
            text: Raw text
            include_metadata: Include metadata features
        
        Returns:
            Dictionary dengan features
        """
        features = {}
        
        # Original text
        features['original_text'] = text
        
        # Cleaned text
        cleaned = self.clean_text(text)
        features['cleaned_text'] = cleaned
        
        # Text without stopwords
        no_stopwords = self.remove_stopwords(cleaned)
        features['text_no_stopwords'] = no_stopwords
        
        # Stemmed text
        stemmed = self.stem_text(no_stopwords)
        features['stemmed_text'] = stemmed
        
        if include_metadata:
            # Text length features
            features['original_length'] = len(text)
            features['cleaned_length'] = len(cleaned)
            features['word_count'] = len(cleaned.split())
            features['char_count'] = len(cleaned)
            features['avg_word_length'] = np.mean([len(w) for w in cleaned.split()]) if cleaned else 0
            
            # Text quality features
            features['has_emoji'] = bool(re.search(r'[^\w\s,]', text))
            features['has_url'] = bool(re.search(r'http\S+|www\S+', text))
            features['exclamation_count'] = text.count('!')
            features['question_count'] = text.count('?')
            features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return features
    
    def batch_extract_features(
        self, 
        texts: List[str], 
        include_metadata: bool = True
    ) -> pd.DataFrame:
        """
        Extract features untuk batch texts
        
        Args:
            texts: List of texts
            include_metadata: Include metadata features
        
        Returns:
            DataFrame dengan features
        """
        features_list = []
        
        for text in texts:
            features = self.extract_features(text, include_metadata)
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def save_config(self, path: str):
        """Save feature extractor config"""
        config_data = {
            'max_length': self.max_length,
            'min_length': self.min_length,
            'version': '1.0.0',
            'created_at': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"✅ Feature extractor config saved to {path}")
    
    @classmethod
    def load_config(cls, path: str):
        """Load feature extractor from config"""
        with open(path) as f:
            config = json.load(f)
        
        return cls(config=config)


class FeatureStore:
    """
    Feature Store untuk manage features
    """
    
    def __init__(self, db_config: Dict, feature_extractor: TextFeatureExtractor = None):
        """
        Initialize feature store
        
        Args:
            db_config: Database configuration
            feature_extractor: TextFeatureExtractor instance
        """
        self.db_config = db_config
        self.feature_extractor = feature_extractor or TextFeatureExtractor()
        
        # Create feature tables
        self.create_feature_tables()
    
    def create_feature_tables(self):
        """Create feature store tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Feature metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_metadata (
                    id SERIAL PRIMARY KEY,
                    feature_name VARCHAR(100) UNIQUE NOT NULL,
                    feature_type VARCHAR(50),
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Text features table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS text_features (
                    id SERIAL PRIMARY KEY,
                    review_id INTEGER REFERENCES reviews(id),
                    original_text TEXT,
                    cleaned_text TEXT,
                    text_no_stopwords TEXT,
                    stemmed_text TEXT,
                    original_length INTEGER,
                    cleaned_length INTEGER,
                    word_count INTEGER,
                    char_count INTEGER,
                    avg_word_length FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(review_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_text_features_review 
                ON text_features(review_id);
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("✅ Feature store tables created/verified")
            
        except Exception as e:
            logger.error(f"Error creating feature tables: {e}")
    
    def store_features(self, review_id: int, text: str) -> Dict:
        """
        Extract dan store features untuk review
        
        Args:
            review_id: Review ID
            text: Review text
        
        Returns:
            Extracted features
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(text)
            
            # Store to database
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO text_features 
                (review_id, original_text, cleaned_text, text_no_stopwords, stemmed_text,
                 original_length, cleaned_length, word_count, char_count, avg_word_length)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (review_id) 
                DO UPDATE SET
                    cleaned_text = EXCLUDED.cleaned_text,
                    text_no_stopwords = EXCLUDED.text_no_stopwords,
                    stemmed_text = EXCLUDED.stemmed_text,
                    original_length = EXCLUDED.original_length,
                    cleaned_length = EXCLUDED.cleaned_length,
                    word_count = EXCLUDED.word_count,
                    char_count = EXCLUDED.char_count,
                    avg_word_length = EXCLUDED.avg_word_length;
            """, (
                review_id,
                features['original_text'],
                features['cleaned_text'],
                features['text_no_stopwords'],
                features['stemmed_text'],
                features['original_length'],
                features['cleaned_length'],
                features['word_count'],
                features['char_count'],
                features['avg_word_length']
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return features
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return {}
    
    def get_features(self, review_id: int) -> Optional[Dict]:
        """
        Get stored features untuk review
        
        Args:
            review_id: Review ID
        
        Returns:
            Features dictionary atau None
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    original_text, cleaned_text, text_no_stopwords, stemmed_text,
                    original_length, cleaned_length, word_count, char_count, avg_word_length
                FROM text_features
                WHERE review_id = %s;
            """, (review_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return {
                    'original_text': result[0],
                    'cleaned_text': result[1],
                    'text_no_stopwords': result[2],
                    'stemmed_text': result[3],
                    'original_length': result[4],
                    'cleaned_length': result[5],
                    'word_count': result[6],
                    'char_count': result[7],
                    'avg_word_length': result[8]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting features: {e}")
            return None
    
    def get_training_features(self, limit: int = None) -> pd.DataFrame:
        """
        Get features untuk training
        
        Args:
            limit: Maximum number of samples
        
        Returns:
            DataFrame dengan features dan labels
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            
            query = """
                SELECT 
                    tf.cleaned_text,
                    tf.stemmed_text,
                    tf.word_count,
                    tf.avg_word_length,
                    r.sentiment as sentiment_label,
                    r.rating
                FROM text_features tf
                JOIN reviews r ON tf.review_id = r.id
                WHERE r.sentiment IS NOT NULL
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            logger.info(f"✅ Loaded {len(df)} training features")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting training features: {e}")
            return pd.DataFrame()
    
    def batch_store_features(self, reviews_df: pd.DataFrame):
        """
        Batch store features untuk multiple reviews
        
        Args:
            reviews_df: DataFrame dengan 'id' dan 'review_text' columns
        """
        logger.info(f"Storing features untuk {len(reviews_df)} reviews...")
        
        stored = 0
        for idx, row in reviews_df.iterrows():
            try:
                self.store_features(row['id'], row['review_text'])
                stored += 1
                
                if (stored % 100) == 0:
                    logger.info(f"  Processed {stored}/{len(reviews_df)} reviews")
                    
            except Exception as e:
                logger.error(f"Error storing features for review {row['id']}: {e}")
        
        logger.info(f"✅ Stored features untuk {stored} reviews")


def initialize_feature_store(db_config: Dict) -> FeatureStore:
    """
    Initialize feature store dan populate features
    
    Args:
        db_config: Database configuration
    
    Returns:
        FeatureStore instance
    """
    logger.info("🚀 Initializing feature store...")
    
    # Create feature extractor
    feature_extractor = TextFeatureExtractor()
    
    # Save config
    config_dir = Path("models/feature_store")
    config_dir.mkdir(parents=True, exist_ok=True)
    feature_extractor.save_config(config_dir / "feature_config.json")
    
    # Create feature store
    feature_store = FeatureStore(db_config, feature_extractor)
    
    # Load reviews and populate features
    try:
        conn = psycopg2.connect(**db_config)
        df = pd.read_sql("SELECT id, review_text FROM reviews WHERE review_text IS NOT NULL", conn)
        conn.close()
        
        logger.info(f"Found {len(df)} reviews to process")
        
        # Batch store features
        feature_store.batch_store_features(df)
        
    except Exception as e:
        logger.error(f"Error populating feature store: {e}")
    
    logger.info("✅ Feature store initialized!")
    
    return feature_store


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', 5432)),
        'database': os.getenv('POSTGRES_DB', 'sentiment_db'),
        'user': os.getenv('POSTGRES_USER', 'sentiment_user'),
        'password': os.getenv('POSTGRES_PASSWORD', 'password')
    }
    
    # Initialize feature store
    feature_store = initialize_feature_store(db_config)
    
    # Get training features
    train_features = feature_store.get_training_features(limit=10)
    print(train_features.head())
